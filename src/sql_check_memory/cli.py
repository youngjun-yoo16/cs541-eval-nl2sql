"""Command-line interface for setting up and using SQL Check Memory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .corpus import collect_mistakes
from .database import describe_schema, initialize_database
from .learning import learn_from_jsonl
from .llm import OpenAIClient
from .store import CheckStore
from .verification import verify_query


def _project_root() -> Path:
    """Return the project directory, based on this module's location."""
    return Path(__file__).resolve().parents[2]


def _get_database_and_store_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Choose paths from command-line options or use the project's default paths."""
    root = _project_root()

    database_option = getattr(args, "database", None)
    store_option = getattr(args, "store", None)

    if database_option:
        database_path = Path(database_option)
    else:
        database_path = root / "data" / "duckdb" / "concert_singer.duckdb"

    if store_option:
        store_path = Path(store_option)
    else:
        store_path = root / "data" / "memory" / "checks.jsonl"

    return database_path, store_path


def _add_api_options(command: argparse.ArgumentParser) -> None:
    """Add model and API endpoint options to a command that calls OpenAI."""
    command.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    command.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser and define all available CLI commands."""
    parser = argparse.ArgumentParser(
        prog="sql-check-memory",
        description="Learn and apply SQL checks with the OpenAI API and DuckDB.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init_command = commands.add_parser(
        "init-db",
        help="Convert the included Spider SQLite database to DuckDB.",
    )
    init_command.add_argument("--source")
    init_command.add_argument("--database")

    collect_command = commands.add_parser(
        "collect",
        help="Generate SQL for Spider questions and save incorrect candidates.",
    )
    collect_command.add_argument("--database")
    collect_command.add_argument("--questions")
    collect_command.add_argument("--output")
    _add_api_options(collect_command)

    learn_command = commands.add_parser(
        "learn",
        help="Learn checks from incorrect/correct SQL pairs.",
    )
    learn_command.add_argument("--database")
    learn_command.add_argument("--store")
    learn_command.add_argument("--examples")
    _add_api_options(learn_command)

    verify_command = commands.add_parser(
        "verify",
        help="Retrieve and run checks against a supplied SQL query.",
    )
    verify_command.add_argument("--database")
    verify_command.add_argument("--store")
    verify_command.add_argument("--question", required=True)
    verify_command.add_argument("--sql", required=True)
    verify_command.add_argument("--top-k", type=int, default=5)
    _add_api_options(verify_command)

    ask_command = commands.add_parser(
        "ask",
        help="Generate SQL with the OpenAI API, then verify it.",
    )
    ask_command.add_argument("--database")
    ask_command.add_argument("--store")
    ask_command.add_argument("--question", required=True)
    ask_command.add_argument("--top-k", type=int, default=5)
    _add_api_options(ask_command)

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the requested project operation."""
    parser = _build_parser()
    args = parser.parse_args()
    database_path, store_path = _get_database_and_store_paths(args)

    if args.command == "init-db":
        _initialize_database(args, database_path)
        return

    if not database_path.exists():
        parser.error(
            f"DuckDB database not found: {database_path}. "
            "Run `sql-check-memory init-db` first."
        )

    if args.command == "collect":
        _collect_errors(args, database_path)
        return

    api_client = OpenAIClient(args.base_url, args.model)
    check_store = CheckStore(store_path)

    if args.command == "learn":
        _learn_checks(args, database_path, check_store, api_client)
        return

    question, sql = _get_question_and_sql(args, database_path, api_client)
    result = verify_query(
        database_path,
        question,
        sql,
        check_store,
        api_client,
        args.top_k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _initialize_database(args: argparse.Namespace, database_path: Path) -> None:
    """Convert the chosen Spider SQLite file to the project's DuckDB database."""
    if args.source:
        sqlite_path = Path(args.source)
    else:
        sqlite_path = _project_root() / "data" / "raw" / "concert_singer.sqlite"

    initialize_database(sqlite_path, database_path)
    print(f"DuckDB database ready: {database_path}")


def _collect_errors(args: argparse.Namespace, database_path: Path) -> None:
    """Generate SQL for the starter questions and save candidates that fail evaluation."""
    root = _project_root()
    questions_path = Path(args.questions) if args.questions else root / "data" / "examples" / "questions.jsonl"
    output_option = getattr(args, "output", None)
    if output_option:
        output_path = Path(output_option)
    else:
        output_path = root / "data" / "memory" / "generated_mistakes.jsonl"

    api_client = OpenAIClient(args.base_url, args.model)
    error_count, correct_count = collect_mistakes(
        questions_path,
        output_path,
        database_path,
        api_client,
    )
    print(
        f"Collected {error_count} incorrect candidate(s); "
        f"{correct_count} matched the reference. Saved to: {output_path}"
    )


def _learn_checks(
    args: argparse.Namespace,
    database_path: Path,
    check_store: CheckStore,
    api_client: OpenAIClient,
) -> None:
    """Ask the API to generalize mistakes and save validated checks to memory."""
    if args.examples:
        examples_path = Path(args.examples)
    else:
        examples_path = _project_root() / "data" / "examples" / "mistakes.jsonl"

    learned_count, rejected_count = learn_from_jsonl(
        examples_path,
        database_path,
        check_store,
        api_client,
    )
    print(
        f"Learned {learned_count} check(s); rejected {rejected_count} proposal(s). "
        f"Memory: {check_store.path}"
    )


def _get_question_and_sql(
    args: argparse.Namespace,
    database_path: Path,
    api_client: OpenAIClient,
) -> tuple[str, str]:
    """Return the supplied SQL, or generate SQL when the user chose the ask command."""
    if args.command == "verify":
        return args.question, args.sql

    schema = describe_schema(database_path)
    sql = api_client.generate_sql(args.question, schema)
    print(f"Candidate SQL:\n{sql}\n")
    return args.question, sql


if __name__ == "__main__":
    main()
