"""Apply retrieved SQL checks and return evidence plus an optional suggested fix."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb
from sqlglot import exp

from .database import describe_schema
from .llm import OpenAIClient
from .models import CheckRecord
from .retrieval import retrieve, run_rule, validate_read_only_select
from .store import CheckStore


def verify_query(
    database_path: Path,
    question: str,
    sql: str,
    check_store: CheckStore,
    api_client: OpenAIClient,
    top_k: int = 5,
    database_id: str = "concert_singer",
) -> dict[str, Any]:
    """Validate a query, run relevant checks, and return findings and a repair suggestion."""
    query_tree, validation_error = validate_read_only_select(sql)
    if validation_error:
        return _rejected_query_result(sql, validation_error)

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        execution_result = _execute_candidate(connection, sql)
        matching_checks = retrieve(
            question,
            sql,
            database_id,
            check_store.load(),
            top_k,
        )
        findings = _run_checks(
            matching_checks,
            question,
            query_tree,
            connection,
        )
    finally:
        connection.close()

    if not execution_result["ok"]:
        findings.insert(
            0,
            {
                "rule_id": "execution_error",
                "severity": "error",
                "evidence": execution_result["error"],
            },
        )

    suggested_sql, findings = _suggest_correction_if_needed(
        database_path,
        question,
        sql,
        findings,
        api_client,
    )

    if findings:
        status = "issues_found"
    else:
        status = "no_known_issue_detected"

    return {
        "status": status,
        "candidate_sql": sql,
        "execution": execution_result,
        "checks_considered": [check.id for check in matching_checks],
        "findings": findings,
        "suggested_sql": suggested_sql,
        "note": "No known check firing is not proof of correctness.",
    }


def _rejected_query_result(sql: str, error: str) -> dict[str, Any]:
    """Build a consistent result when a candidate fails the SQL safety checks."""
    return {
        "status": "rejected",
        "candidate_sql": sql,
        "findings": [
            {
                "rule_id": "read_only_guard",
                "severity": "error",
                "evidence": error,
            }
        ],
        "suggested_sql": None,
    }


def _execute_candidate(connection: Any, sql: str) -> dict[str, Any]:
    """Run a candidate query with a short interrupt timeout and report any error."""
    timeout_seconds = 5
    timer = threading.Timer(timeout_seconds, connection.interrupt)

    try:
        timer.start()
        connection.execute("SET threads=1")
        connection.execute(sql).fetchmany(1000)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        timer.cancel()


def _run_checks(
    checks: list[CheckRecord],
    question: str,
    query_tree: exp.Expression,
    connection: Any,
) -> list[dict[str, Any]]:
    """Run each retrieved rule and attach its explanation to any finding."""
    findings = []
    for check in checks:
        parameters = check.diagnostic.get("parameters", {})
        result = run_rule(
            check.rule_id,
            question,
            query_tree,
            connection,
            parameters,
        )
        if result is None:
            continue

        finding = {
            "check_id": check.id,
            "explanation": check.explanation,
            "suggested_correction": check.suggested_correction,
        }
        finding.update(result)
        findings.append(finding)

    return findings


def _suggest_correction_if_needed(
    database_path: Path,
    question: str,
    sql: str,
    findings: list[dict[str, Any]],
    api_client: OpenAIClient,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Ask the model for a fix when checks report issues and validate its SQL shape."""
    if not findings:
        return None, findings

    try:
        schema = describe_schema(database_path)
        suggested_sql = api_client.suggest_fix(question, sql, findings, schema)
    except Exception as exc:
        findings.append(
            {
                "rule_id": "repair_unavailable",
                "severity": "warning",
                "evidence": str(exc),
            }
        )
        return None, findings

    _, validation_error = validate_read_only_select(suggested_sql)
    if validation_error:
        findings.append(
            {
                "rule_id": "repair_rejected",
                "severity": "warning",
                "evidence": validation_error,
            }
        )
        return None, findings

    return suggested_sql, findings
