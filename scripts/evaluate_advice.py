"""Compare frozen checks with LLM review of those same records' advice."""
import argparse
import hashlib
import json
from pathlib import Path

from sql_check_memory.database import describe_schema
from sql_check_memory.llm import OpenAIClient
from sql_check_memory.store import CheckStore

instructions = """Review the candidate DuckDB SQL against the question and schema using the retrieved advice.
Flag it only if the advice identifies a specific error in this query. Consider equivalent SQL formulations.
Return only a JSON object with "flag" (boolean) and "reason" (short explanation)."""
root = Path("data/runs/milestone3")
cache = root / "advice_cache"
cases = [json.loads(line) for line in (root / "detection_results.jsonl").read_text().splitlines()]
checks = {c.id: c for c in CheckStore(Path("data/memory/milestone3_checks.jsonl")).load()}
schema = describe_schema(Path("data/duckdb/concert_singer.duckdb"))
client = OpenAIClient(model="gpt-6.1-sol")
parser = argparse.ArgumentParser()
parser.add_argument("--run", action="store_true", help="Make API calls for uncached inputs")
live = parser.parse_args().run
results = []
needed = 0
for row in cases:
    advice = [{"explanation": checks[i].explanation, "hint": checks[i].suggested_correction}
              for i in row["retrieved"]]
    # Keep labels, references, and the static check findings out of the prompt.
    prompt = json.dumps(dict(schema=schema, question=row["question"], sql=row["sql"], advice=advice))
    key = hashlib.sha256(json.dumps([client.model, client.base_url, instructions, prompt]).encode()).hexdigest()
    path = cache / (key + ".json")
    if not advice:
        verdict = {"flag": False, "reason": "No retrieved advice."}
    else:
        if not path.exists():
            needed += 1
            if not live:
                continue
            cache.mkdir(parents=True, exist_ok=True)
            raw = client.chat(instructions, prompt)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({"response": raw}))
            temp.replace(path)
        raw = json.loads(path.read_text())["response"]
        verdict = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        if type(verdict.get("flag")) is not bool or not isinstance(verdict.get("reason"), str):
            raise ValueError(f"Invalid verdict in {path}")
    results.append({**row, "advice": verdict, "model": client.model})
if not live:
    print(f"{len(cases)} candidates; {needed} new API calls needed.")
else:
    dest = root / "advice_results.jsonl"
    dest.write_text("".join(json.dumps(r) + "\n" for r in results))
    for method in ["checks", "advice"]:
        flagged = [r for r in results if (bool(r["findings"]) if method == "checks" else r["advice"]["flag"])]
        hits = sum(not r["correct"] for r in flagged)
        print(f"{method}: {hits}/{sum(not r['correct'] for r in results)} incorrect flagged; "
              f"{len(flagged)-hits}/{sum(r['correct'] for r in results)} correct flagged.")
    print(f"Saved {len(results)} verdicts to {dest}")
