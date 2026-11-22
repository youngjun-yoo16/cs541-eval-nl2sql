"""Functions for creating and describing the prototype's DuckDB database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb


def initialize_database(sqlite_path: Path, duckdb_path: Path) -> None:
    """Copy each table in the read-only SQLite source into a new DuckDB file."""
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite source not found: {sqlite_path}")

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    if duckdb_path.exists():
        duckdb_path.unlink()

    sqlite_uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    source = sqlite3.connect(sqlite_uri, uri=True)
    destination = duckdb.connect(str(duckdb_path))

    try:
        table_names = _get_sqlite_table_names(source)
        for table_name in table_names:
            _copy_table(source, destination, table_name)
    finally:
        destination.close()
        source.close()


def _get_sqlite_table_names(connection: sqlite3.Connection) -> list[str]:
    """Return user-created table names from the SQLite source in sorted order."""
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """
    rows = connection.execute(query).fetchall()
    return [row[0] for row in rows]


def _copy_table(
    source: sqlite3.Connection,
    destination: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    """Create one DuckDB table and copy its rows from the SQLite source."""
    quoted_table = _quote_identifier(table_name)
    column_info = source.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    columns = [(row[1], row[2] or "VARCHAR") for row in column_info]
    boolean_columns = [
        index
        for index, (_, sqlite_type) in enumerate(columns)
        if "BOOL" in sqlite_type.upper()
    ]

    definitions = []
    for column_name, sqlite_type in columns:
        quoted_column = _quote_identifier(column_name)
        duckdb_type = _duckdb_type(sqlite_type)
        definitions.append(f"{quoted_column} {duckdb_type}")

    create_statement = f"CREATE TABLE {quoted_table} ({', '.join(definitions)})"
    destination.execute(create_statement)

    if not columns:
        return

    quoted_columns = [_quote_identifier(column_name) for column_name, _ in columns]
    column_list = ", ".join(quoted_columns)
    placeholders = ", ".join("?" for _ in columns)
    select_statement = f"SELECT {column_list} FROM {quoted_table}"
    insert_statement = (
        f"INSERT INTO {quoted_table} ({column_list}) VALUES ({placeholders})"
    )

    cursor = source.execute(select_statement)
    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        rows = _convert_boolean_values(rows, boolean_columns)
        destination.executemany(insert_statement, rows)


def _convert_boolean_values(
    rows: list[tuple],
    boolean_columns: list[int],
) -> list[tuple]:
    """Convert SQLite's common T/F boolean strings into Python booleans."""
    if not boolean_columns:
        return rows

    converted_rows = []
    true_values = {"t", "true", "1", "yes"}

    for row in rows:
        values = list(row)
        for column_index in boolean_columns:
            value = values[column_index]
            if value is not None:
                values[column_index] = str(value).strip().lower() in true_values
        converted_rows.append(tuple(values))

    return converted_rows


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier so its spelling cannot change the generated SQL."""
    escaped_name = name.replace('"', '""')
    return f'"{escaped_name}"'


def _duckdb_type(sqlite_type: str) -> str:
    """Map a SQLite type name to the closest simple DuckDB type used here."""
    kind = sqlite_type.upper()

    if "BOOL" in kind:
        return "BOOLEAN"
    if "INT" in kind:
        return "BIGINT"
    if any(marker in kind for marker in ("REAL", "FLOA", "DOUB", "NUM")):
        return "DOUBLE"
    if "BLOB" in kind:
        return "BLOB"
    return "VARCHAR"


def describe_schema(database_path: Path) -> str:
    """Return table columns and known relationships as text for the SQL model."""
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        query = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
        """
        columns = connection.execute(query).fetchall()
    finally:
        connection.close()

    columns_by_table: dict[str, list[str]] = {}
    for table_name, column_name, data_type in columns:
        description = f"{column_name} {data_type}"
        columns_by_table.setdefault(table_name, []).append(description)

    table_descriptions = []
    for table_name, descriptions in columns_by_table.items():
        joined_columns = ", ".join(descriptions)
        table_descriptions.append(f"{table_name}({joined_columns})")

    relationships = [
        "concert.Stadium_ID = stadium.Stadium_ID",
        "singer_in_concert.concert_ID = concert.concert_ID",
        "singer_in_concert.Singer_ID = singer.Singer_ID",
    ]
    table_text = "\n".join(table_descriptions)
    relationship_text = "; ".join(relationships)
    return f"{table_text}\nRelationships: {relationship_text}"
