"""Find relevant learned checks and run their deterministic SQL rules."""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp

from .models import CheckRecord

WORD_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def sql_features(sql: str) -> tuple[set[str], set[str]]:
    """Extract the table and column names referenced by a SQL query."""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return set(), set()

    table_names = set()
    for table in tree.find_all(exp.Table):
        table_names.add(table.name.lower())

    column_names = set()
    for column in tree.find_all(exp.Column):
        if column.name:
            column_names.add(column.name.lower())

    return table_names, column_names


def retrieve(
    question: str,
    sql: str,
    database_id: str,
    checks: list[CheckRecord],
    top_k: int = 5,
) -> list[CheckRecord]:
    """Rank stored checks by database, question terms, tables, and columns."""
    combined_text = f"{question} {sql}".lower()
    query_terms = set(WORD_PATTERN.findall(combined_text))
    query_tables, query_columns = sql_features(sql)
    ranked_checks: list[tuple[float, CheckRecord]] = []

    for check in checks:
        applicability = check.applicability
        allowed_databases = {
            str(value).lower()
            for value in applicability.get("database_ids", [])
        }
        if allowed_databases and database_id.lower() not in allowed_databases:
            continue

        required_tables = {
            str(value).lower() for value in applicability.get("tables", [])
        }
        required_columns = {
            str(value).lower() for value in applicability.get("columns", [])
        }
        required_terms = {
            str(value).lower() for value in applicability.get("terms", [])
        }

        score = 0.0
        if not _features_match(required_tables, query_tables):
            continue
        if not _features_match(required_columns, query_columns):
            continue
        if not _features_match(required_terms, query_terms):
            continue

        score += _overlap_score(required_tables, query_tables)
        score += _overlap_score(required_columns, query_columns)
        score += _overlap_score(required_terms, query_terms)
        ranked_checks.append((score, check))

    ranked_checks.sort(key=lambda item: (-item[0], item[1].id))
    selected = ranked_checks[: max(0, top_k)]
    return [check for _, check in selected]


def _features_match(required: set[str], available: set[str]) -> bool:
    """Return true when an empty feature list or at least one feature matches."""
    if not required:
        return True
    return bool(required & available)


def _overlap_score(required: set[str], available: set[str]) -> float:
    """Return the fraction of applicability features present in the query."""
    if not required:
        return 0.0
    return len(required & available) / len(required)


def validate_read_only_select(sql: str) -> tuple[exp.Expression | None, str | None]:
    """Parse one SELECT and reject statements or tables outside this prototype."""
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as exc:
        return None, f"SQL parse error: {exc}"

    if len(statements) != 1 or statements[0] is None:
        return None, "Expected exactly one SQL statement."

    tree = statements[0]
    allowed_query_types = (exp.Select, exp.Union, exp.Intersect, exp.Except)
    if not isinstance(tree, allowed_query_types):
        return None, "Only one read-only SELECT query is allowed."

    allowed_tables = {
        "stadium",
        "singer",
        "concert",
        "singer_in_concert",
    }
    for common_table_expression in tree.find_all(exp.CTE):
        allowed_tables.add(common_table_expression.alias_or_name.lower())

    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return None, "Table functions are not allowed."
        if table.db or table.catalog:
            return None, "Qualified external schemas are not allowed."
        if table.name.lower() not in allowed_tables:
            message = f"Unknown table {table.name!r}; only concert_singer tables are allowed."
            return None, message

    unsafe_functions = {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_text",
        "read_blob",
        "sqlite_scan",
        "postgres_scan",
        "parquet_scan",
        "glob",
        "query",
        "query_table",
        "nextval",
        "setseed",
    }
    for node in tree.walk():
        if not isinstance(node, exp.Anonymous):
            continue
        if node.name.lower() in unsafe_functions:
            return None, f"External or dynamic function {node.name!r} is not allowed."

    return tree, None


def run_rule(
    rule_id: str,
    question: str,
    tree: exp.Expression,
    connection: Any,
    parameters: dict[str, Any],
) -> dict[str, Any] | None:
    """Run one allowlisted rule and return evidence when it detects a problem."""
    if rule_id == "order_direction_intent":
        return _check_order_direction(question, tree, parameters)
    if rule_id == "count_distinct_intent":
        return _check_distinct_count(question, tree)
    if rule_id == "count_nullable_column":
        return _check_nullable_count(question, tree, connection)
    if rule_id == "cartesian_join":
        return _check_implicit_join(tree)
    return None


def _check_order_direction(
    question: str,
    tree: exp.Expression,
    parameters: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare words like youngest/oldest with the query's first sort direction."""
    question_text = question.lower()
    expected_direction = None

    ascending_words = ("youngest", "smallest", "lowest", "earliest", "least")
    descending_words = ("oldest", "largest", "highest", "latest", "greatest")

    if any(re.search(rf"\b{word}\b", question_text) for word in ascending_words):
        expected_direction = "ASC"
    if any(re.search(rf"\b{word}\b", question_text) for word in descending_words):
        expected_direction = "DESC"
    if "ascending" in question_text or "low to high" in question_text:
        expected_direction = "ASC"
    if "descending" in question_text or "high to low" in question_text:
        expected_direction = "DESC"

    if expected_direction is None:
        return None

    order_clause = tree.args.get("order")
    if not order_clause or not order_clause.expressions:
        evidence = f"The question implies {expected_direction} ordering, but SQL has no ORDER BY."
        return {"rule_id": "order_direction_intent", "evidence": evidence}

    first_sort = order_clause.expressions[0]
    sort_expression = first_sort.this if isinstance(first_sort, exp.Ordered) else first_sort
    if isinstance(first_sort, exp.Ordered) and first_sort.args.get("desc"):
        actual_direction = "DESC"
    else:
        actual_direction = "ASC"

    if isinstance(sort_expression, exp.Column):
        column_name = sort_expression.name.lower()
    else:
        column_name = ""

    expected_columns = {
        str(column).lower() for column in parameters.get("columns", [])
    }
    if expected_columns and column_name not in expected_columns:
        return None

    if expected_direction == actual_direction:
        return None

    evidence = (
        f"The question implies {expected_direction} order, but SQL orders "
        f"{column_name or 'results'} {actual_direction}."
    )
    return {
        "rule_id": "order_direction_intent",
        "column": column_name,
        "expected_direction": expected_direction,
        "observed_direction": actual_direction,
        "evidence": evidence,
    }


def _check_distinct_count(
    question: str,
    tree: exp.Expression,
) -> dict[str, Any] | None:
    """Check whether a distinct-value question uses COUNT without DISTINCT."""
    distinct_words = ("different", "distinct", "unique")
    question_asks_for_distinct = any(
        re.search(rf"\b{word}\b", question.lower()) for word in distinct_words
    )
    if not question_asks_for_distinct:
        return None

    for count_expression in tree.find_all(exp.Count):
        counted_value = count_expression.this
        if isinstance(counted_value, exp.Distinct):
            continue

        if isinstance(counted_value, exp.Column):
            column_name = counted_value.name
        elif isinstance(counted_value, exp.Star):
            column_name = "*"
        else:
            column_name = str(counted_value)

        evidence = (
            f"The question asks for distinct values, but "
            f"COUNT({column_name}) does not use DISTINCT."
        )
        return {
            "rule_id": "count_distinct_intent",
            "column": column_name,
            "evidence": evidence,
        }

    return None


def _check_nullable_count(
    question: str,
    tree: exp.Expression,
    connection: Any,
) -> dict[str, Any] | None:
    """Find COUNT(column) queries that omit NULL rows from a requested row count."""
    count_phrases = ("how many", "number of", "count of", "total number")
    question_asks_for_count = any(
        phrase in question.lower() for phrase in count_phrases
    )
    if not question_asks_for_count:
        return None

    query_tables = list(tree.find_all(exp.Table))
    for count_expression in tree.find_all(exp.Count):
        counted_value = count_expression.this
        if isinstance(counted_value, exp.Star):
            continue
        if not isinstance(counted_value, exp.Column):
            continue

        if counted_value.table:
            table_name = counted_value.table
        elif len(query_tables) == 1:
            table_name = query_tables[0].name
        else:
            continue

        quoted_table = _quote_identifier(table_name)
        quoted_column = _quote_identifier(counted_value.name)
        null_query = (
            f"SELECT COUNT(*) FROM {quoted_table} "
            f"WHERE {quoted_column} IS NULL"
        )

        try:
            null_count = connection.execute(null_query).fetchone()[0]
        except Exception:
            continue

        if null_count == 0:
            continue

        evidence = (
            f"COUNT({counted_value.name}) omits {null_count} NULL rows, "
            "but the question asks for a row or entity count."
        )
        return {
            "rule_id": "count_nullable_column",
            "table": table_name,
            "column": counted_value.name,
            "null_rows": int(null_count),
            "evidence": evidence,
        }

    return None


def _check_implicit_join(tree: exp.Expression) -> dict[str, Any] | None:
    """Flag an implicit join that has neither an ON nor a USING condition."""
    for join in tree.find_all(exp.Join):
        join_method = str(join.args.get("method") or "").upper()
        if join_method == "NATURAL":
            continue

        join_kind = join.args.get("kind")
        has_on_condition = join.args.get("on") is not None
        has_using_condition = join.args.get("using") is not None
        if join_kind or has_on_condition or has_using_condition:
            continue

        joined_table = join.this
        if isinstance(joined_table, exp.Table):
            table_name = joined_table.name
        else:
            table_name = str(joined_table)

        evidence = (
            f"The join to {table_name} has no ON or USING condition "
            "and may multiply rows."
        )
        return {
            "rule_id": "cartesian_join",
            "table": table_name,
            "evidence": evidence,
        }

    return None


def _quote_identifier(name: str) -> str:
    """Quote a table or column name used in a diagnostic SQL statement."""
    escaped_name = name.replace('"', '""')
    return f'"{escaped_name}"'
