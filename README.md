# LongMemEval — LLM-Wiki Memory Harness

An evaluation harness that benchmarks the **[LLM-Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** memory pattern against [LongMemEval](https://github.com/xiaowu0162/LongMemEval), a rigorous benchmark for long-term memory in chat assistants.

---

## What this is

Standard RAG rediscovers knowledge from scratch on every query. This project implements a different approach: as chat sessions arrive, an **ingest agent** compiles them into a persistent, structured wiki of atomic facts. At query time, a **selection agent** finds the relevant atoms and a **synthesis agent** reasons over them to produce an answer.

The wiki is a compounding artifact — each new session updates it, contradictions are flagged, and superseded facts are tracked. The knowledge is compiled once, not re-derived per query.

---

## Architecture

```
LongMemEval sessions
        │
        ▼
  Ingest Agent          ← extracts atoms from each user-assistant turn pair
        │                 uses search_atoms / read_atom tools to detect superseding
        ▼
     WikiDB              ← persistent store: atom files, subject index, word index, log
        │
        ▼
  Selection Agent       ← given a question, finds relevant atom_ids from the wiki
        │
        ▼
  Synthesis Agent       ← reasons over selected atoms, produces a short answer
        │
        ▼
    hypothesis.jsonl    ← evaluated by LongMemEval's evaluate_qa.py
```

### Key files

| File                  | Role                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `ingest.py`           | Ingest agent: LongMemEval sessions → wiki atoms                   |
| `db.py`               | WikiDB: atom CRUD, subject/word index, log                        |
| `selection.py`        | Selection agent: question → relevant atom_ids                     |
| `synthesis.py`        | Synthesis agent: atoms + question → answer                        |
| `answer_generator.py` | Orchestrates selection → as_of filter → synthesis                 |
| `run_eval.py`         | Full eval loop over LongMemEval with concurrency + MLflow tracing |
| `models.py`           | Pydantic models for all LLM calls                                 |

---

## Submodules

### [LongMemEval](https://github.com/xiaowu0162/LongMemEval)

The evaluation benchmark (ICLR 2025). 500 high-quality questions across five memory ability categories:

- Information Extraction
- Multi-Session Reasoning
- Knowledge Updates
- Temporal Reasoning
- Abstention

Each instance includes a long timestamped chat history. The system must parse and memorize sessions online, then answer a question posed after all sessions.

### [autoresearch](https://github.com/karpathy/autoresearch)

Karpathy's autonomous AI research agent. An agent is given a training codebase (`train.py`) and a research program (`program.md`) and runs experiments overnight — modifying code, training for 5 minutes, checking if the result improved, and repeating. The metric is `val_bpb` (validation bits per byte). Included here as a reference for autonomous agent-driven experimentation loops.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/amulyahwr/longevalmem-llmwiki-autoresearch
cd longevalmem-llmwiki-autoresearch

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

Download the LongMemEval dataset from Hugging Face and place it under `eval/longmemeval/data/`:

```bash
mkdir -p eval/longmemeval/data && cd eval/longmemeval/data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ../../..
```

### Ollama configuration

This project uses a **local LLM served via [Ollama](https://ollama.com/)** through its OpenAI-compatible API. Start Ollama with `ollama serve` and pull your model with `ollama pull <model>`.

Configure via environment variables or a `.env` file at the project root:

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434/v1   # default
OLLAMA_MODEL=<ollama-model-tag>             # match the model you pulled
OLLAMA_TIMEOUT=120.0                        # seconds, increase for slower hardware
```

No API key is required — Ollama uses `"ollama"` as a placeholder.

---

## Running the eval

```bash
# Quick test: oracle split, 5 questions
python run_eval.py --dataset oracle --limit 5

# Full oracle eval with concurrency
python run_eval.py --dataset oracle --concurrency 4

# LongMemEval_S (115k-token histories)
python run_eval.py --dataset s --concurrency 2
```

Results are written to `results/<dataset>_predictions.jsonl`. To score:

```bash
python3 eval/longmemeval/evaluate_qa.py gpt-4o results/oracle_predictions.jsonl \
    eval/longmemeval/data/longmemeval_oracle.json
```

MLflow traces are available at `http://localhost:5001` (start with `mlflow ui --port 5001`).

---

## How the LLM-Wiki pattern works

Rather than retrieving from raw documents at query time, the system **incrementally compiles** knowledge into a wiki of atomic markdown facts. Three operations:

- **Ingest** — a new session arrives; the agent extracts atoms, detects superseding facts, updates the index and log.
- **Query** — the selection agent reads the index, picks relevant atoms, the synthesis agent reasons over them.
- **Lint** — periodic health check: orphan atoms, stale facts, contradictions, missing cross-references.

The wiki is the intelligence layer. Cross-references are pre-built, contradictions already flagged, synthesis already reflects all prior sessions. Nothing is re-derived at query time.
