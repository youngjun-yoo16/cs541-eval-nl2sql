# SQL Check Memory

Local research prototype for learning reusable SQL verification checks from NL2SQL mistakes, storing them as structured records, and applying them to new SQL.

## Database

The included source is Spider 1.0 `concert_singer` (four small tables; about 37 KB). The SQLite snapshot is converted to DuckDB locally. Attribution and source are in `data/raw/SOURCE.json`; Spider is distributed under CC BY-SA 4.0.

The starter corpus contains controlled mutations of Spider-style question/reference pairs. It demonstrates the pipeline, not naturally collected model errors or a benchmark result. Add JSONL pairs with `id`, `database_id`, `question`, `incorrect_sql`, `correct_sql`, and `source`. For the research comparison, use generated baseline-agent attempts and verify error labels by execution equivalence.

## OpenAI API setup

The SQL and check-generation calls use OpenAI's Responses API. Set `OPENAI_API_KEY` in the environment before running `collect`, `learn`, or `ask`. You can set `OPENAI_MODEL` (default `gpt-6.2-sol`) and `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) to override the model or endpoint. The key is never stored in project files. Questions, schema, SQL, and check diagnostics are sent to the API for inference.

```powershell
cd outputs/sql-check-memory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
$env:OPENAI_API_KEY = "your-key-here"
# Optional overrides:
# $env:OPENAI_MODEL = "gpt-6.2-sol"
# $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
sql-check-memory init-db
sql-check-memory collect
sql-check-memory learn --examples data/memory/generated_mistakes.jsonl
sql-check-memory learn
```

## Use

```powershell
sql-check-memory ask --question "Show the name and release year of the song by the youngest singer."
sql-check-memory verify --question "Show the name and release year of the song by the youngest singer." --sql "SELECT song_name, song_release_year FROM singer ORDER BY age DESC LIMIT 1"
sql-check-memory learn --examples data/examples/mistakes.jsonl
```

`ask` generates SQL through the API, retrieves checks, runs them, and requests a suggested correction. `verify` accepts SQL generated elsewhere; it needs an API key only when a finding requires a model-generated correction. Configure `--model`, `--base-url`, `--database`, and `--store` as needed.

## Architecture and safety

1. `collect` asks the fixed local SQL generator model to answer the seed Spider questions, then retains candidate/reference pairs whose execution results differ.
2. `learn` asks the local LLM to analyze incorrect/correct pairs and propose structured check metadata. Each proposal must fire on the incorrect SQL and not on the supplied correction before storage.
3. Proposals can select only an allowlist of deterministic static rules. No model-authored Python, `eval`, shell, or arbitrary SQL is executed.
4. JSONL memory stores rule ID, applicability features, explanation, correction hint, confidence, and source example IDs.
5. Retrieval uses question terms plus parsed table/column overlap.
6. SQLGlot permits one SELECT over the four known tables, DuckDB opens read-only, and matched deterministic checks produce findings. A suggested correction is returned for review; no automatic rewrite is executed.

Rules: `order_direction_intent`, `count_distinct_intent`, `count_nullable_column`, and `cartesian_join` (implicit joins without a predicate). A clean result means no known check fired, not proof of correctness. See [research plan](docs/research_plan.md) for the controlled comparison and limitations.
