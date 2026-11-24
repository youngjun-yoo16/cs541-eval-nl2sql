"""Run from cs541-eval-nl2sql. --live adds measured API reviewers."""
import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import duckdb
from sql_check_memory.database import describe_schema
from sql_check_memory.llm import OpenAIClient, _extract_response_text, _make_api_request, _send_api_request
from sql_check_memory.retrieval import retrieve, run_rule, validate_read_only_select
from sql_check_memory.store import CheckStore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("data/runs/person3"))
parser.add_argument("--live", action="store_true")
parser.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-4.1-mini"])
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
dbpath = Path("data/duckdb/concert_singer.duckdb")
db = duckdb.connect(str(dbpath), read_only=True)
schema = describe_schema(dbpath)
memory = CheckStore(Path("data/memory/milestone3_checks.jsonl")).load()
cases = [json.loads(line) for line in Path("data/runs/milestone3/advice_results.jsonl").read_text().splitlines()]
instructions = """Review the candidate DuckDB SQL against the question and schema using the retrieved advice.
Flag it only if the advice identifies a specific error in this query. Consider equivalent SQL formulations.
Return only a JSON object with "flag" (boolean) and "reason" (short explanation)."""


def write_csv(name, rows):
    path = args.out / name
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def timed(operation):
    start = time.perf_counter_ns()
    result = operation()
    return result, (time.perf_counter_ns() - start) / 1e6


def median_ms(operation):
    operation()  # Warm up this statement and connection.
    return statistics.median(timed(operation)[1] for _ in range(11))


def query_ms(sql):
    tree, error = validate_read_only_select(sql)
    if error:
        raise ValueError(error)
    return median_ms(lambda: db.execute(sql).fetchall())


def review(model, payload):
    client = OpenAIClient(model=model)
    body = dict(model=model, instructions=instructions, input=json.dumps(payload), store=False)
    digest = hashlib.sha256(json.dumps([client.base_url, body]).encode()).hexdigest()
    path = args.out / "api_cache" / (digest + ".json")
    if not path.exists():
        if not client.api_key:
            raise SystemExit("Set OPENAI_API_KEY in this shell before --live.")
        request = _make_api_request(client.base_url, client.api_key, body)
        response, elapsed = timed(lambda: _send_api_request(request, client.timeout, client.base_url))
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(response=response, api_ms=elapsed)))
        tmp.replace(path)
    saved = json.loads(path.read_text())
    response = saved["response"]
    verdict = json.loads(_extract_response_text(response).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if type(verdict["flag"]) is not bool:
        raise ValueError(f"Unexpected verdict in {path}")
    usage = response["usage"]
    return verdict["flag"], verdict["reason"], saved["api_ms"], usage["input_tokens"], usage["output_tokens"]


def summarize(rows):
    summaries = []
    for arm in dict.fromkeys(r["configuration"] for r in rows):
        group = [r for r in rows if r["configuration"] == arm]
        tp = sum(r["flag"] and not r["correct"] for r in group)
        fp = sum(r["flag"] and r["correct"] for r in group)
        positives = sum(not r["correct"] for r in group)
        summary = dict(configuration=arm, candidates=len(group), TP=tp, FP=fp,
                       FN=positives-tp, TN=len(group)-positives-fp,
                       F1=2*tp/(positives+tp+fp), calls=sum(r["calls"] for r in group))
        for key in ["input_tokens", "output_tokens", "verification_ms"]:
            values = [r[key] for r in group]
            summary[key if key != "verification_ms" else "median_verification_ms"] = (
                None if None in values else statistics.median(values) if key == "verification_ms" else sum(values))
        summaries.append(summary)
    write_csv("candidates.csv", rows)
    write_csv("summary.csv", summaries)


scores = {r["id"]: r["rescored_match"] for r in json.loads(Path("data/runs/development_baseline_rescored.json").read_text())}
baseline = []
for path in sorted(Path("data/runs/development_baseline").glob("*.json")):
    r = json.loads(path.read_text())
    baseline.append(dict(id=r["id"], model=r["model"], strict_match=scores[r["id"]],
                         query_ms=query_ms(r["candidate_sql"]), original_generation_and_scoring_s=r["elapsed_seconds"]))
write_csv("baseline.csv", baseline)
rows = []
prepared = []
for r in cases:
    q, sql = r["question"], r["sql"]
    def retrieve_checks():
        return retrieve(q, sql, "concert_singer", memory, top_k=5)
    checks = retrieve_checks()
    assert [c.id for c in checks] == r["retrieved"]
    def detect():
        tree, error = validate_read_only_select(sql)
        if error:
            raise ValueError(error)
        return [finding for c in retrieve_checks()
                if (finding := run_rule(c.rule_id, q, tree, db, c.diagnostic["parameters"]))]
    findings = detect()
    assert findings == r["findings"]
    base = dict(case=r["case"], correct=r["correct"], sql=sql, question=q, query_ms=query_ms(sql))
    retrieval_ms = median_ms(retrieve_checks)
    advice = [dict(explanation=c.explanation, hint=c.suggested_correction) for c in checks]
    prepared.append((base, advice, retrieval_ms))
    rows.append(dict(**base, configuration="static", flag=bool(findings), reason=json.dumps(findings),
                     verification_ms=median_ms(detect), api_ms=0, calls=0, input_tokens=0, output_tokens=0))
    rows.append(dict(**base, configuration="original advice gpt-6.1-sol", **r["advice"],
                     verification_ms=None, api_ms=None if checks else 0, calls=int(bool(checks)),
                     input_tokens=None if checks else 0, output_tokens=None if checks else 0))
summarize(rows)
if args.live:
    for model in args.models:
        for base, advice, retrieval_ms in prepared:
            payload = dict(schema=schema, question=base["question"], sql=base["sql"], advice=advice)
            flag, reason, latency, inputs, outputs = (review(model, payload) if advice
                else (False, "No retrieved advice.", 0, 0, 0))
            rows.append(dict(**base, configuration="measured advice "+model, flag=flag, reason=reason,
                             verification_ms=retrieval_ms+latency, api_ms=latency, calls=int(bool(advice)),
                             input_tokens=inputs, output_tokens=outputs))
        summarize(rows)
db.close()
print((args.out/"summary.csv").read_text())
