"""Build and compare the small question/query corpus used by this prototype."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
from sqlglot import exp

from .database import describe_schema
from .llm import OpenAIClient
from .retrieval import validate_read_only_select


def collect_mistakes(
    questions_path: Path,
    output_path: Path,
    database_path: Path,
    api_client: OpenAIClient,
) -> tuple[int, int]:
    """Generate SQL for each question and save candidates that differ from reference results."""
    schema = describe_schema(database_path)
    connection = duckdb.connect(str(database_path), read_only=True)
    incorrect_candidates = []
    correct_candidates = 0

    try:
        question_rows = _read_jsonl(questions_path)
        for question_row in question_rows:
            question = question_row["question"]
            reference_sql = question_row["reference_sql"]
            candidate_sql = api_client.generate_sql(question, schema)

            equivalent, evidence = _queries_are_equivalent(
                connection,
                candidate_sql,
                reference_sql,
            )
            if equivalent:
                correct_candidates += 1
            else:
                incorrect_candidates.append(
                    _make_mistake_record(question_row, candidate_sql, evidence)
                )
    finally:
        connection.close()

    _write_jsonl(output_path, incorrect_candidates)
    return len(incorrect_candidates), correct_candidates


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return each non-empty line as a dictionary."""
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def _make_mistake_record(
    question_row: dict[str, Any],
    candidate_sql: str,
    evidence: str,
) -> dict[str, Any]:
    """Combine one generated error with its question and reference correction."""
    return {
        "id": str(question_row["id"]),
        "database_id": question_row.get("database_id", "concert_singer"),
        "question": question_row["question"],
        "incorrect_sql": candidate_sql,
        "correct_sql": question_row["reference_sql"],
        "source": question_row.get("source", "Spider 1.0"),
        "error_evidence": evidence,
        "split": question_row.get("split", "train"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file, creating its parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            handle.write(line + "\n")


def _queries_are_equivalent(
    connection: Any,
    candidate_sql: str,
    reference_sql: str,
) -> tuple[bool, str]:
    """Compare two safe SELECT queries by their results on the DuckDB snapshot."""
    candidate_tree, candidate_error = validate_read_only_select(candidate_sql)
    if candidate_error:
        return False, f"Candidate rejected: {candidate_error}"

    reference_tree, reference_error = validate_read_only_select(reference_sql)
    if reference_error:
        return False, f"Reference rejected: {reference_error}"

    try:
        candidate_rows = connection.execute(candidate_sql).fetchall()
        reference_rows = connection.execute(reference_sql).fetchall()
    except Exception as exc:
        return False, f"Execution error: {exc}"

    reference_is_ordered = reference_tree.args.get("order") is not None
    if reference_is_ordered:
        equivalent = candidate_rows == reference_rows
    else:
        candidate_counts = Counter(repr(row) for row in candidate_rows)
        reference_counts = Counter(repr(row) for row in reference_rows)
        equivalent = candidate_counts == reference_counts

    if equivalent:
        evidence = "Execution results match."
    else:
        evidence = "Candidate results differ from the reference results."
    return equivalent, evidence
