"""Evaluate the LLM-Wiki agent pipeline on LongMemEval.

Usage:
    python run_eval.py --dataset oracle --limit 5
    python run_eval.py --dataset oracle
    python run_eval.py --dataset s --concurrency 3
"""
import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import mlflow
from tqdm import tqdm

from answer_generator import generate
from db import WikiDB
from ingest import ingest_session

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

WIKI_BASE = Path(__file__).parent / "wiki"


async def _eval_instance(
    q: dict,
    sem: asyncio.Semaphore,
    out_f,
    write_lock: asyncio.Lock,
    pbar: tqdm,
) -> dict | None:
    async with sem:
        question_id = q["question_id"]
        wiki = WikiDB(WIKI_BASE / question_id)
        with mlflow.start_span(
            name=f"question/{question_id}",
            attributes={
                "question_id": question_id,
                "question_type": q.get("question_type", ""),
                "question": q["question"][:200],
            },
        ):
            try:
                wiki.reset()

                for session_id, turns, date in sorted(
                    zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]),
                    key=lambda x: x[2],
                ):
                    await ingest_session(session_id, turns, date, wiki)

                hypothesis = await generate(
                    question=q["question"],
                    wiki=wiki,
                    as_of=q.get("question_date"),
                )
                row = {
                    "question_id": question_id,
                    "hypothesis": hypothesis,
                    "question_type": q["question_type"],
                }
                async with write_lock:
                    print(json.dumps(row), file=out_f, flush=True)
                return row
            except Exception as e:
                print(f"\n[ERROR] {question_id}: {e}")
                return None
            finally:
                pbar.update(1)


def _print_summary(results: list[dict]) -> None:
    if not results:
        return
    by_type: dict[str, int] = defaultdict(int)
    for r in results:
        by_type[r["question_type"]] += 1
    print(f"\n--- Generated {len(results)} hypotheses ---")
    for qtype, n in sorted(by_type.items()):
        print(f"  {qtype:<35} n={n}")


async def main(args: argparse.Namespace) -> None:
    _db = Path(__file__).parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{_db}")
    mlflow.set_experiment(f"llmwiki-eval-{args.dataset}")
    mlflow.openai.autolog()

    data_file = Path(args.data_dir) / DATASET_FILES[args.dataset]
    questions = json.loads(data_file.read_text())
    if args.limit:
        questions = questions[: args.limit]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    WIKI_BASE.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} questions already done, skipping.")

    pending = [q for q in questions if q["question_id"] not in done_ids]
    print(f"Processing {len(pending)} questions with concurrency={args.concurrency}...")
    print(f"Traces: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 → experiment 'llmwiki-eval-{args.dataset}'")

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    file_mode = "a" if args.resume else "w"
    with out_path.open(file_mode) as out_f:
        with tqdm(total=len(pending), desc=f"Questions ({args.dataset})", unit="q") as pbar:
            tasks = [_eval_instance(q, sem, out_f, write_lock, pbar) for q in pending]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

    results = [r for r in gathered if isinstance(r, dict)]
    _print_summary(results)
    print(f"\nOutput written to: {out_path}")
    print(f"\nTo score QA accuracy:")
    print(f"  python3 eval/longmemeval/evaluate_qa.py gpt-4o {out_path} eval/longmemeval/data/{DATASET_FILES[args.dataset]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent-based wiki on LongMemEval.")
    parser.add_argument("--dataset", default="oracle", choices=list(DATASET_FILES.keys()))
    parser.add_argument("--data_dir", default="eval/longmemeval/data")
    parser.add_argument("--out_file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = f"results/{args.dataset}_predictions.jsonl"
    asyncio.run(main(args))
