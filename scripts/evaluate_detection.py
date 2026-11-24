"""Run from the project root. Outputs auditable per-candidate records as JSONL."""
import json
from pathlib import Path

import duckdb
from sql_check_memory.corpus import _queries_are_equivalent
from sql_check_memory.retrieval import retrieve, run_rule, validate_read_only_select
from sql_check_memory.store import CheckStore

# Each pair shares a question and reference. Alternate spellings test SQL or NL variation.
top = "SELECT Name FROM singer WHERE Country='France' ORDER BY Age DESC LIMIT 1"
bottom = top.replace("DESC", "ASC")
maximum = "SELECT Name FROM singer WHERE Country='France' AND Age=(SELECT MAX(Age) FROM singer WHERE Country='France')"
distinct = "SELECT COUNT(DISTINCT Country) FROM singer WHERE Age >= 30"
plain = distinct.replace("DISTINCT ", "")
grouped = "SELECT COUNT(*) FROM (SELECT Country FROM singer WHERE Age >= 30 GROUP BY Country) AS countries"
pairs = [
    ("age_order", "Among French singers ordered by age, give the name of the oldest.", top, top, bottom),
    ("age_alternative", "Among French singers ordered by age, give the name of the oldest.", top, maximum, maximum.replace("MAX(", "MIN(")),
    ("age_paraphrase", "Which French singer is oldest? Give their name.", top, top, bottom),
    ("country_count", "How many different countries are represented by singers aged 30 or above?", distinct, distinct, plain),
    ("country_alternative", "How many different countries are represented by singers aged 30 or above?", distinct, grouped, grouped.replace(" GROUP BY Country", "")),
    ("country_paraphrase", "How many unique countries are represented by singers aged 30 or above?", distinct, distinct, plain),
]

store = CheckStore(Path("data/memory/milestone3_checks.jsonl")).load()
db = duckdb.connect("data/duckdb/concert_singer.duckdb", read_only=True)
for case, question, reference, correct, incorrect in pairs:
    for intended_correct, sql in [(True, correct), (False, incorrect)]:
        tree, error = validate_read_only_select(sql)
        assert error is None, error
        matches, evidence = _queries_are_equivalent(db, sql, reference)
        assert matches == intended_correct, (case, evidence)
        checks = retrieve(question, sql, "concert_singer", store, top_k=5)
        findings = []
        for check in checks:
            finding = run_rule(check.rule_id, question, tree, db, check.diagnostic["parameters"])
            if finding:
                findings.append(finding)
        print(json.dumps(dict(case=case, question=question, reference=reference, sql=sql,
                              correct=matches, retrieved=[c.id for c in checks], findings=findings)))
db.close()
