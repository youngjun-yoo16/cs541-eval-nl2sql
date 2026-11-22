"""Turn verified SQL mistakes into reusable, structured check records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

from .llm import OpenAIClient
from .models import CheckRecord
from .retrieval import run_rule, validate_read_only_select
from .store import CheckStore

ALLOWED_RULES = {
    "order_direction_intent",
    "count_distinct_intent",
    "count_nullable_column",
    "cartesian_join",
}


def learn_from_jsonl(
    examples_path: Path,
    database_path: Path,
    check_store: CheckStore,
    api_client: OpenAIClient,
) -> tuple[int, int]:
    """Ask the model to generalize each error and store only validated rule proposals."""
    learned_count = 0
    rejected_count = 0
    connection = duckdb.connect(str(database_path), read_only=True)

    try:
        with examples_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                example = _parse_example(line, examples_path, line_number)
                proposal = api_client.propose_check(example)
                record = _make_check_record(example, proposal)

                if record is None:
                    rejected_count += 1
                    continue

                if not _check_fires_only_on_the_mistake(record, example, connection):
                    rejected_count += 1
                    continue

                check_store.upsert(record)
                learned_count += 1
    finally:
        connection.close()

    return learned_count, rejected_count


def _parse_example(line: str, path: Path, line_number: int) -> dict[str, Any]:
    """Parse one JSONL line and include its file location in any error."""
    try:
        example = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path} on line {line_number}: {exc}") from exc

    if not isinstance(example, dict):
        raise ValueError(f"Expected a JSON object in {path} on line {line_number}.")
    return example


def _make_check_record(
    example: dict[str, Any],
    proposal: dict[str, Any],
) -> CheckRecord | None:
    """Build a check record only when the model selected a supported rule."""
    rule_id = str(proposal.get("rule_id", "unknown"))
    if rule_id not in ALLOWED_RULES:
        return None

    terms = proposal.get("applicability_terms", [])
    tables = proposal.get("tables", [])
    columns = proposal.get("columns", [])
    if not _all_strings(terms, tables, columns):
        return None

    confidence = float(proposal.get("confidence", 0.5))
    confidence = min(1.0, max(0.0, confidence))
    source_id = _get_source_id(example)
    check_id = _create_check_id(rule_id, terms, tables, columns)

    return CheckRecord(
        id=check_id,
        version=1,
        check_type="static_rule",
        rule_id=rule_id,
        dialect="duckdb",
        applicability={
            "description": "Applicability features proposed by the LLM.",
            "database_ids": [example.get("database_id", "concert_singer")],
            "terms": terms,
            "tables": tables,
            "columns": columns,
        },
        diagnostic={
            "procedure": rule_id,
            "parameters": {"columns": columns},
        },
        explanation=str(proposal.get("explanation", "")),
        suggested_correction=str(proposal.get("suggested_correction", "")),
        confidence=confidence,
        provenance={
            "source_example_ids": [source_id],
            "created_by": "openai-api",
            "validation_summary": (
                "The allowlisted rule fires on the supplied mistake "
                "but not on its correction."
            ),
        },
    )


def _all_strings(*values: Any) -> bool:
    """Return true when every provided value is a list of strings."""
    for value in values:
        if not isinstance(value, list):
            return False
        if any(not isinstance(item, str) for item in value):
            return False
    return True


def _get_source_id(example: dict[str, Any]) -> str:
    """Return the example's ID or create a stable ID from its contents."""
    if example.get("id"):
        return str(example["id"])

    serialized = json.dumps(example, sort_keys=True)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
    return digest[:12]


def _create_check_id(
    rule_id: str,
    terms: list[str],
    tables: list[str],
    columns: list[str],
) -> str:
    """Create a repeatable check ID from its rule and applicability features."""
    parts = [
        rule_id,
        ",".join(sorted(term.lower() for term in terms)),
        ",".join(sorted(table.lower() for table in tables)),
        ",".join(sorted(column.lower() for column in columns)),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return f"check-{digest[:12]}"


def _check_fires_only_on_the_mistake(
    record: CheckRecord,
    example: dict[str, Any],
    connection: Any,
) -> bool:
    """Require the proposed rule to flag the bad SQL and pass the correction."""
    incorrect_tree, incorrect_error = validate_read_only_select(example["incorrect_sql"])
    corrected_tree, corrected_error = validate_read_only_select(example["correct_sql"])
    if incorrect_error or corrected_error:
        return False

    parameters = record.diagnostic["parameters"]
    incorrect_finding = run_rule(
        record.rule_id,
        example["question"],
        incorrect_tree,
        connection,
        parameters,
    )
    corrected_finding = run_rule(
        record.rule_id,
        example["question"],
        corrected_tree,
        connection,
        parameters,
    )
    return incorrect_finding is not None and corrected_finding is None
