# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Benchmarks the **LLM-Wiki memory pattern** against [LongMemEval](https://github.com/xiaowu0162/LongMemEval), a 500-question benchmark for long-term memory in chat assistants. Instead of RAG (rediscovers knowledge from scratch), it uses an incremental wiki: an ingest agent extracts atomic facts into a persistent structured wiki; at query time a selection agent retrieves relevant atoms and a synthesis agent reasons over them.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Set env vars in `.env`:
```
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=<ollama-model-tag>
OLLAMA_TIMEOUT=120.0
```

Start Ollama: `ollama serve &` then pull your model: `ollama pull <model>`.

Download benchmark data:
```bash
mkdir -p eval/longmemeval/data && cd eval/longmemeval/data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

## Running Evaluations

```bash
# Quick sanity check
python run_eval.py --dataset oracle --limit 5

# Full oracle split
python run_eval.py --dataset oracle --concurrency 4

# Resume interrupted run
python run_eval.py --dataset oracle --concurrency 4 --resume

# Debug a single question
python diagnose.py --index 0
python diagnose.py --question_id <question_id>
```

Score results:
```bash
python3 eval/longmemeval/evaluate_qa.py gpt-4o results/oracle_predictions.jsonl \
    eval/longmemeval/data/longmemeval_oracle.json
```

Monitor traces:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001  # → http://localhost:5001
```

## Linting / Tests

```bash
ruff check .          # lint
ruff check --fix .    # auto-fix
pytest                # run tests (pytest-asyncio configured in pyproject.toml)
```

## Architecture

```
LongMemEval sessions
    ↓
ingest.py       — extracts atomic facts per session, handles supersession
    ↓
db.py (WikiDB)  — markdown atom store + subject/word indexes + append-only log
    ↓
selection.py    — navigates wiki via tools to find relevant atom_ids
    ↓
synthesis.py    — reasons over atoms, produces final answer
    ↓
results/<dataset>_predictions.jsonl
```

**Orchestration entry points:**
- `run_eval.py` — full benchmark loop with concurrency, MLflow, resume
- `answer_generator.py` — single-question pipeline (selection → as_of filter → synthesis)
- `diagnose.py` — single-question debug with colored output

**WikiDB on-disk layout** (`wiki/<question_id>/`):
```
index.md           — atom catalog
log.md             — append-only event log
subjects.json      — subject → atom_id map
word_index.json    — inverted word → [atom_ids] index
atoms/<atom_id>.md — YAML frontmatter + content per atom
```

**Atom frontmatter fields:** `atom_id`, `kind` (fact/preference/event/belief), `source` (user/assistant), `subject`, `valid_from`, `valid_until`, `is_superseded`, `superseded_by`, `supersedes`.

**Agent tools** (defined in `tools.py`): `search_atoms`, `read_atom`, `list_subjects`, `list_all_atoms` — these are what selection and ingest agents call to navigate the wiki.

**LLM client** (`llm_client.py`) uses the OpenAI-compatible LM Studio API; `models.py` holds Pydantic models for structured LLM outputs.

## Key Design Decisions

- **Unified indexes** — `word_index.json` and `subjects.json` are not split by source (user vs assistant); the selection agent reads full atom content and judges relevance contextually.
- **Subject naming convention** — user atoms use general subjects (`"user's location"`); assistant atoms use specific subjects (`"user's flight itinerary"`) to prevent incorrect supersession across voices.
- **Assistant atoms** — only extract personalized artifacts (itineraries, bookings, recommendations), not general advice.
- **Temporal metadata** — `valid_until` is stamped when an atom is superseded, enabling before/after interval queries.

## Submodules

- `LongMemEval/` — benchmark dataset and scoring scripts (ICLR 2025)
- `autoresearch/` — Karpathy's autonomous AI research agent (reference implementation)
