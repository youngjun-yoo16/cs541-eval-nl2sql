# Research plan

## Motivation

Tk-Boost accumulates database-specific corrective knowledge from an NL2SQL agent's errors, associates knowledge with applicability conditions, and retrieves relevant knowledge for later queries. This project tests whether representing that knowledge as executable/static verification procedures improves detection of recurring errors over natural-language advice.

## Hypotheses

- **H1 (detection):** Check memory has higher error-detection F1 than NL memory at a matched retrieval budget.
- **H2 (precision):** Explicit diagnostic evidence improves precision, without an unacceptable increase in false positives on correct SQL.
- **H3 (repair utility):** Corrections suggested from a fired check improve execution accuracy over advice-only feedback.

Treat H1 as primary; H2/H3 explain the trade-offs. Report confidence intervals across task/database groups, not only a pooled point estimate.

## Unit and labels

A unit is `(database, question, candidate SQL, reference answer set)`. A candidate is incorrect only if it fails the benchmark's execution/equivalence protocol; syntax/runtime failures count as incorrect. Where multiple valid SQL answers exist, use the benchmark's accepted evaluation method. Keep exact-match accuracy as a secondary metric only.

For check evaluation, a fired check is a true positive when the candidate is incorrect and the diagnostic evidence is relevant to that incorrectness. It is a false positive when it fires on a correct candidate. Report per-query detection (any useful check fires), check-level precision, recall, F1, specificity, and false-positive rate. Measure correction outcome separately by executing the proposed candidate in the same evaluation sandbox.

## Dataset construction

1. Choose a public benchmark with questions, schemas, reference SQL, and locally runnable database snapshots (Spider is the initial target; this prototype uses only its small `concert_singer` database).
2. Partition by database/schema before generating attempts. Hold out complete databases for evaluation; keep their questions and references out of check induction.
3. Generate candidate SQL on training questions with a fixed baseline agent. Save prompts, outputs, model/version, schema context, and run metadata.
4. Compare candidates against references with the benchmark evaluator and execution protocol. Retain only verified failures as mistake examples; preserve valid candidates as negative examples for false-positive calibration.
5. Split the training mistakes into check induction and development sets. Never induce checks from test errors or test corrections.

A benchmark reference is a correction target, not necessarily the only equivalent SQL string. If licensing, benchmark evaluator behavior, or database setup prevents a clean split, document it before reporting results.

## Check induction and validation

For each training failure, ask an LLM to propose a generalized check and its conditions, diagnostic procedure, explanation, and suggested correction. Require structured output conforming to `check.schema.json`. Deduplicate checks and validate against held-out development examples. Admit a check only after its diagnostic is runnable/safe and its applicability can be evaluated. Store provenance to the source failure IDs, but never expose held-out or test outcomes during retrieval.

Allowed check form in the current prototype:

- `static_rule`: deterministic, allowlisted predicate over parsed SQL, the question, and (for NULL-sensitive count checks) database facts. LLM output selects and parameterizes a known implementation; no model-authored code or SQL executes.

Checks state dialect and required tables/columns. Candidate SQL is parsed as one SELECT, restricted to the four known tables, and executed read-only with a short interrupt timeout. Unsupported semantics should be treated as unknown, not proof of correctness. Read-only diagnostic SQL is a later extension once its safety and evaluation contract are specified.

## Current local prototype

The first implementation uses Spider 1.0 `concert_singer`, DuckDB, SQLGlot, JSONL memory, and the OpenAI Responses API. Starter mistake pairs are controlled mutations and are admitted only when the selected allowlisted rule fires on the incorrect SQL and not on its correction. Rules currently cover order direction, missing `DISTINCT` in distinct-count questions, counting nullable columns for row-count questions, and implicit joins with no predicate. This provides a runnable loop, not the controlled NL-advice-versus-check experiment described above. API prompts include schema, question, and SQL and are sent to OpenAI for processing.

## Online evaluation arms

Use the same baseline generator, task inputs, token/call budget, and retrieval candidates in every arm.

- **Baseline:** no memory retrieval.
- **NL memory:** natural-language advice learned from the same induction failures.
- **Check memory:** retrieved checks run against candidate SQL and the task database; send fired evidence plus the stored correction to the repair step.

Run both detection-only and repair-enabled evaluations. Detection-only isolates whether checks identify errors; repair-enabled measures downstream utility. Freeze the memory before evaluating held-out databases. If cross-database generalization is also tested, report it separately from same-database reuse.

## Retrieval and matching

Initial retrieval uses table/column/operator/clause features extracted from parsed SQL plus check applicability constraints and database identifiers. Apply top-k and token budgets equally to both memory arms. Record candidates retrieved, checks run, runtime, and reasons for applicability or rejection. Do not use gold SQL or correctness labels at inference time.

## Baselines and ablations

- No-memory baseline.
- Tk-Boost-style natural-language knowledge with matched retrieval budget.
- Check memory with static checks only.
- Check memory with diagnostic SQL only.
- Check memory without applicability filtering (retrieval ablation).
- Optional oracle retrieval upper bound, clearly labeled and excluded from the main comparison.

## Metrics and analysis

Primary: per-query error-detection F1 on initially incorrect held-out candidates, with false-positive rate on initially correct candidates reported alongside it.

Secondary:

- detection precision, recall, specificity, and error-family breakdown;
- execution accuracy before and after correction;
- repair success on incorrect candidates and regression rate on correct candidates;
- check coverage (fraction of tasks with an applicable check), runtime, and cost;
- confidence intervals via bootstrap resampling by database, plus paired comparisons across the same tasks.

A reliability claim requires better detection at a comparable false-positive rate, not simply more flags. Report no-result/unknown diagnostics and unsupported dialect cases.

## Risks and controls

- **Synthetic error distribution:** generated errors may not match human mistakes. Describe findings as baseline-agent error reuse; later validate on naturally occurring correction logs if available.
- **Reference incompleteness:** execution equivalence can be imperfect. Use benchmark evaluators and manually inspect a stratified sample of disagreements.
- **Overgeneralized checks:** validate on development data, keep explicit applicability and confidence, and test on schema-held-out data.
- **Unsafe diagnostics:** static validation, read-only connections, timeouts, resource limits, and query allowlisting.
- **Unfair baseline:** match retrieval and generation budgets; publish prompts/configuration and seed/model version.
- **Leakage:** split by database before induction; inspect check text for benchmark/test identifiers.

## Milestones

1. **Benchmark harness:** load one benchmark split, run evaluator, capture candidate SQL and outcomes.
2. **Memory format and validator:** parse records, validate static and diagnostic check safety.
3. **Offline induction:** generate NL advice and structured checks from training failures; deduplicate and measure coverage.
4. **Online retrieval/detection:** retrieve checks and produce evidence-backed flags.
5. **Controlled comparison:** run matched baseline/NL/check arms, analyze F1 and false-positive trade-offs.
6. **Robustness:** schema-held-out evaluation, ablations, and optional second benchmark.

## Reproducibility record

Every run should capture benchmark/version/license, split manifest, database hashes, SQL dialect, generator and LLM identifiers, prompt hashes, decoding settings, random seeds, check-memory snapshot hash, retrieval parameters, evaluator version, and execution-sandbox limits.
