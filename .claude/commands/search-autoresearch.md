# Search Autoresearch

You are an autonomous search methodology researcher for the Lattice project. Your job is to improve retrieval and re-ranking quality, measured on the LongMemEval benchmark. All improvements that work here transfer directly to Lattice production code.

## North Star

Identify **dataset-agnostic** retrieval and re-ranking improvements. Lattice is an agent-native enterprise search system — it cannot know question type at runtime. Every experiment must use techniques that work without `question_type` labels. Generation improvements (answer_generator.py) are secondary — they raise the measurement ceiling but don't transfer to Lattice's production retrieval pipeline.

## In-scope files (read these first for full context)

- eval/harness/run_eval.py         — retrieval strategy: top_k, deep_rerank, hypotheses, raw_query
- eval/harness/answer_generator.py — generation: atom formatting, system prompt
- eval/harness/search_results.tsv  — experiment log (append only, do not overwrite)

## Fixed files (do NOT modify)

- eval/harness/ingest.py       — atomize+distill is expensive; treat ingested atoms as fixed
- backend/serving/l3_search.py — shared Lattice production code; changes go through separate review
- eval/LongMemEval/            — external benchmark; read-only (except evaluate_qa.py model_zoo additions)

## Dataset-agnostic constraint

**NEVER** add logic conditioned on `question_type`. Lattice does not know the question type at query time. Disallowed patterns:
- `if question_type == "temporal-reasoning": ...`
- Type-specific top_k multipliers
- Type-specific generation hints (`_TYPE_HINTS`)
- Any post-processing that branches on question type

All techniques must apply uniformly to every query.

## Metric to optimize

Primary: QA accuracy (higher is better) — from evaluate_qa.py output, look for "Accuracy:" line
Secondary: Recall@Any, NDCG — printed by run_eval.py retrieval summary at end of run

## Baseline

Overall: TBD — run the baseline first before starting experiments.

## Two-stage evaluation

**Fast iteration (oracle_stratified 50q) — generation with Gemma:**
```bash
python eval/harness/run_eval.py --dataset oracle_stratified \
  --out_file eval/harness/results/oracle_temp.jsonl --no-resume --concurrency 5
```

**Switch LM Studio to Qwen for evaluation (automatic):**
```bash
python eval/harness/switch_lm_model.py nvidia/nemotron-3-nano-4b
```

**Scoring:**
```bash
python3 eval/LongMemEval/src/evaluation/evaluate_qa.py \
  nemotron-nano-4b eval/harness/results/oracle_temp.jsonl \
  eval/LongMemEval/data/longmemeval_oracle.json
```

Note: always score against the full `longmemeval_oracle.json` reference — it contains all ground-truth answers for the stratified subset too.

**Switch back to Gemma after evaluation:**
```bash
python eval/harness/switch_lm_model.py google/gemma-4-e2b
```

**Full validation (only for improvements > 0.02 on the 50q sample):**
```bash
python eval/harness/run_eval.py --dataset oracle --concurrency 5 \
  --out_file eval/harness/results/oracle_full.jsonl --no-resume
python eval/harness/switch_lm_model.py nvidia/nemotron-3-nano-4b
python3 eval/LongMemEval/src/evaluation/evaluate_qa.py \
  nemotron-nano-4b eval/harness/results/oracle_full.jsonl \
  eval/LongMemEval/data/longmemeval_oracle.json
python eval/harness/switch_lm_model.py google/gemma-4-e2b
```

## Experiment loop

LOOP FOREVER:

1. Read current state of in-scope files AND read search_results.tsv
2. Pick ONE change — generation OR retrieval OR re-ranking, not a mix.
   Before picking, check whether the idea is already implemented by:
   - Scanning the description column of search_results.tsv (previously tested experiments)
   - Reading the current code in run_eval.py and answer_generator.py (already-applied changes)
   If the idea is already present in either the TSV or the code: skip it and pick the next untested idea.
   If the priority list is exhausted: invent a new dataset-agnostic idea not yet in the TSV or code.
3. Save current file content to a backup:
   ```bash
   cp eval/harness/run_eval.py eval/harness/run_eval.py.bak
   # or
   cp eval/harness/answer_generator.py eval/harness/answer_generator.py.bak
   ```
4. Apply the change to the file
5. Run fast iteration (oracle_stratified 50q)
6. Switch to Qwen, score, switch back to Gemma
7. Parse accuracy from output (look for "Accuracy:" line)
8. If accuracy improved vs. best so far:
   - Keep the file change
   - For improvements > 0.02, run full 500q validation and score
9. If accuracy equal or worse:
   - Restore from backup:
     ```bash
     cp eval/harness/run_eval.py.bak eval/harness/run_eval.py
     # or
     cp eval/harness/answer_generator.py.bak eval/harness/answer_generator.py
     ```
10. Append to search_results.tsv (tab-separated, NOT comma-separated):
    ```
    run_id\taccuracy\trecall_any\tndcg\tstatus\tdescription
    ```
    status: keep / discard / crash
11. Check exit criteria: if the last 5 consecutive experiments are all `discard`, stop and print a summary.
12. Otherwise: NEVER STOP. Generate the next idea and repeat.

## Rollback without git

No git commits in this loop. Rollback = `cp <file>.bak <file>` using the backup saved in step 3.

## Ideas to try (dataset-agnostic only, in priority order)

### Generation (oracle accuracy ceiling)
1. Richer generation prompt — add universal date-reasoning instructions and "use the source" guidance (no type conditioning)
2. Richer atom formatting — include `[date]` and `(source)` for every atom regardless of question type

### Retrieval (transfers directly to Lattice)
3. HyDE via process_query() — 3 declarative hypotheses instead of raw question
4. BM25 hybrid — pass `raw_query=question` to search_atoms (adds lexical signal to dense search)
5. Uniform top_k multiplier — fetch 2× or 3× atoms, pass `atoms[:top_k]` to generator (wider net improves recall)

### Re-ranking (post-retrieval, in run_eval.py)
6. Deep LLM reranking — enable `deep_rerank=True` for all queries (uniform, no type branching)
7. Source diversity reranking — round-robin atoms across source sessions for all queries (no single session monopolizes top_k)
8. Listwise LLM re-ranker — rank all top_k candidates jointly via LLM call (use concurrency 1–2 to avoid OOM)

## Crashes

If a run crashes (Python error, OOM), read the last 30 lines of output, fix or skip. Log status as "crash" in the TSV. For OOM: reduce `--concurrency` to 2 or 1.

## Exit criteria

After each experiment, check: have the last 5 consecutive experiments all resulted in `discard`?
- If yes: stop the loop, print a summary of all kept experiments and their accuracy gains, and report that the search methodology has plateaued.
- If no: continue.

Do NOT stop for any other reason unless manually interrupted.
