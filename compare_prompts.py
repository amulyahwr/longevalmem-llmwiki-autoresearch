"""A/B comparison of named prompt variants on the 50-question stratified sample.

Usage:
    python compare_prompts.py --baseline v0 --variant v1
    python compare_prompts.py --baseline v0 --variant v1 --concurrency 2

Add new variants to VARIANTS below. Each variant overrides the defaults in VariantDefaults.
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
from agents import Agent
from tqdm import tqdm

import selection as _sel
import tools as _tools
import answer_generator as _ag
from db import WikiDB
from ingest import ingest_session
from llm_client import get_model, get_model_settings
from selection import SELECTION_SYSTEM

SAMPLE_FILE = Path("eval_sample_50.json")
DATASET_FILE = Path("eval/longmemeval/data/longmemeval_oracle.json")
RESULTS_DIR = Path("results/ab")
WIKI_BASE = Path("wiki/ab")
EVALUATOR_MODEL = "deepseek-r1-7b"
GROUND_TRUTH = Path("eval/longmemeval/data/longmemeval_oracle.json")
EVAL_SCRIPT = Path("eval/longmemeval/evaluate_qa.py")


@dataclass
class VariantConfig:
    name: str
    selection_system: str = field(default_factory=lambda: SELECTION_SYSTEM)
    max_search_results: int = 5
    max_snippet_chars: int = 120
    max_atom_chars_tool: int = 400
    max_list_chars: int = 1500
    max_atoms: int = 15
    max_atom_chars_answer: int = 500


# --- Named variants -----------------------------------------------------------
# Add entries here. Keys become the --baseline / --variant argument values.
# Unspecified fields inherit VariantConfig defaults (which match current code).

VARIANTS: dict[str, VariantConfig] = {
    "v0": VariantConfig(
        name="v0",
        # original single-angle selection prompt (before the multi-angle patch)
        selection_system="""\
You are a memory retrieval agent. Your job: find all atom_ids relevant to answering a question.

You have tools to actively navigate the wiki:
  search_atoms(query)      — grep atom contents for a keyword; returns atom_ids + snippets
  read_atom(atom_id)       — read a specific atom's full content
  list_all_atoms()         — list every atom_id currently in the wiki

Strategy:
1. Identify the key entities and concepts in the question.
2. For each entity, call search_atoms. Include ALL atom_ids returned in your final answer —
   do not filter them out just because you haven't read them in full.
3. Optionally call read_atom on a few atoms to understand dates or context better.
   Reading an atom is NOT required to include it — if search returned it, include it.
4. For temporal questions ("after X", "before Y", "first", "last"):
   — Find the anchor event and read it to get its date.
   — Search for events relative to that anchor.
   — Include BOTH the anchor atom and the answer atom.
5. Err heavily on the side of inclusion — every atom_id from search_atoms belongs in your answer.
   Synthesis will handle filtering and reasoning.

You MUST call at least one tool before producing your final answer.
Your final answer MUST be ONLY valid JSON. Do NOT write any prose or explanation.
Final answer format (no code fences):
{"relevant_atom_ids": ["id1", "id2", ...]}\
""",
    ),
    "v1": VariantConfig(
        name="v1",
        # multi-angle prompt (the current patched version)
        # selection_system left as default = SELECTION_SYSTEM (current code)
    ),
    "v2": VariantConfig(
        name="v2",
        # size-aware strategy: return everything for small wikis, multi-angle for large ones
        selection_system="""\
You are a memory retrieval agent. Your job: find all atom_ids relevant to answering a question.

You have tools to actively navigate the wiki:
  search_atoms(query)      — grep atom contents for a keyword; returns atom_ids + snippets
  read_atom(atom_id)       — read a specific atom's full content
  list_subjects()          — show all subject → atom_id mappings (browse by topic)
  list_all_atoms()         — list every atom_id currently in the wiki

Strategy:
  Step 1 — call list_all_atoms() first to see how many atoms exist.
  Step 2 — if the wiki has 12 or fewer atoms, return ALL atom_ids immediately. Do not search further.
  Step 3 — if the wiki is larger, use multi-angle search:
    Angle 1 — Exact terms: search_atoms with key nouns and verbs from the question.
    Angle 2 — Synonyms and related concepts: search_atoms with alternate phrasings.
    Angle 3 — Subject browse: list_subjects() and read atoms under relevant subjects.
    Angle 4 (if still empty) — list_all_atoms() for a full sweep.

Additional rules:
- Include ALL atom_ids returned by search_atoms — do not filter before synthesis.
- For temporal questions ("after X", "before Y", "first", "last"):
   — Find the anchor event and read it to get its date.
   — Search for events relative to that anchor.
   — Include BOTH the anchor atom and the answer atom.
- Err on the side of inclusion. Synthesis handles filtering and reasoning.

Your final answer MUST be ONLY valid JSON. Do NOT write any prose or explanation.
Final answer format (no code fences):
{"relevant_atom_ids": ["id1", "id2", ...]}\
""",
    ),
}
# ------------------------------------------------------------------------------


def _apply_variant(v: VariantConfig) -> None:
    """Monkey-patch module globals to match variant config."""
    _tools._MAX_SEARCH_RESULTS = v.max_search_results
    _tools._MAX_SNIPPET_CHARS = v.max_snippet_chars
    _tools._MAX_ATOM_CHARS = v.max_atom_chars_tool
    _tools._MAX_LIST_CHARS = v.max_list_chars
    _ag._MAX_ATOMS = v.max_atoms
    _ag._MAX_ATOM_CHARS = v.max_atom_chars_answer
    _sel._SELECTION_AGENT = Agent(
        name="selection",
        instructions=v.selection_system,
        tools=_sel.SELECTION_TOOLS,
        model=get_model(),
        model_settings=get_model_settings(reasoning=True),
    )


async def _run_question(
    q: dict, sem: asyncio.Semaphore, wiki_base: Path
) -> dict | None:
    async with sem:
        question_id = q["question_id"]
        wiki = WikiDB(wiki_base / question_id)
        try:
            wiki.reset()
            for session_id, turns, date in sorted(
                zip(
                    q["haystack_session_ids"],
                    q["haystack_sessions"],
                    q["haystack_dates"],
                ),
                key=lambda x: x[2],
            ):
                await ingest_session(session_id, turns, date, wiki)
            hypothesis = await _ag.generate(
                question=q["question"],
                wiki=wiki,
                as_of=q.get("question_date"),
            )
            return {
                "question_id": question_id,
                "hypothesis": hypothesis,
                "question_type": q["question_type"],
            }
        except Exception as e:
            print(f"\n[ERROR] {question_id}: {e}")
            return None


async def _run_variant(
    variant: VariantConfig, questions: list[dict], concurrency: int
) -> Path:
    out_path = RESULTS_DIR / f"{variant.name}_predictions.jsonl"
    wiki_base = WIKI_BASE / variant.name
    wiki_base.mkdir(parents=True, exist_ok=True)

    _apply_variant(variant)
    mlflow.set_experiment(f"llmwiki-ab-{variant.name}")

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    results = []

    with out_path.open("w") as f:
        with tqdm(total=len(questions), desc=variant.name, unit="q") as pbar:
            tasks = [_run_question(q, sem, wiki_base) for q in questions]
            for coro in asyncio.as_completed(tasks):
                row = await coro
                if row:
                    async with write_lock:
                        print(json.dumps(row), file=f, flush=True)
                    results.append(row)
                pbar.update(1)

    return out_path


def _score(predictions_path: Path) -> dict[str, float]:
    """Run evaluate_qa.py and parse per-category accuracy."""
    result = subprocess.run(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            EVALUATOR_MODEL,
            str(predictions_path),
            str(GROUND_TRUTH),
        ],
        capture_output=True,
        text=True,
    )
    scores: dict[str, float] = {}
    for line in result.stdout.splitlines():
        m = re.match(r"\s+(\S[\w-]+):\s+([\d.]+)", line)
        if m:
            scores[m.group(1)] = float(m.group(2))
        m2 = re.match(r"\s*[Aa]ccuracy:\s+([\d.]+)", line)
        if m2:
            scores["overall"] = float(m2.group(1))
    if not scores and result.stderr:
        print(f"[scoring error]\n{result.stderr[:1000]}", file=sys.stderr)
    return scores


def _print_delta_table(
    name_a: str, name_b: str, scores_a: dict, scores_b: dict
) -> None:
    cats = sorted(set(scores_a) | set(scores_b))
    if not cats:
        print("\n[scoring produced no results — check stderr above]")
        return
    col_w = max(len(c) for c in cats) + 2
    header = f"{'category':<{col_w}}  {name_a:>8}  {name_b:>8}  {'delta':>8}"
    print("\n" + header)
    print("-" * len(header))
    for cat in cats:
        a = scores_a.get(cat, float("nan"))
        b = scores_b.get(cat, float("nan"))
        delta = b - a
        sign = "+" if delta > 0 else ""
        print(f"{cat:<{col_w}}  {a:>8.3f}  {b:>8.3f}  {sign}{delta:>7.3f}")
    print()


async def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _db = Path(__file__).parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{_db}")
    mlflow.openai.autolog()

    sample = json.loads(SAMPLE_FILE.read_text())
    all_questions = json.loads(DATASET_FILE.read_text())
    sample_ids = set(sample["question_ids"])
    questions = [q for q in all_questions if q["question_id"] in sample_ids]
    print(
        f"Running A/B on {len(questions)} questions: {args.baseline} vs {args.variant}"
    )

    path_a = RESULTS_DIR / f"{args.baseline}_predictions.jsonl"
    path_b = RESULTS_DIR / f"{args.variant}_predictions.jsonl"

    if not args.score_only:
        var_a = VARIANTS[args.baseline]
        var_b = VARIANTS[args.variant]

        print(f"\n[1/2] Running baseline: {args.baseline}")
        path_a = await _run_variant(var_a, questions, args.concurrency)

        print(f"\n[2/2] Running variant: {args.variant}")
        path_b = await _run_variant(var_b, questions, args.concurrency)

    print("\nScoring...")
    scores_a = _score(path_a)
    scores_b = _score(path_b)

    _print_delta_table(args.baseline, args.variant, scores_a, scores_b)

    print(f"Predictions: {path_a}  |  {path_b}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B prompt comparison on the stratified eval sample."
    )
    parser.add_argument("--baseline", default="v0", choices=list(VARIANTS))
    parser.add_argument("--variant", default="v1", choices=list(VARIANTS))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--score-only", action="store_true", help="skip inference, score existing results")
    asyncio.run(main(parser.parse_args()))
