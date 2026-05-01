"""Diagnostic runner: full pipeline on a single question.

Shows wiki state after each ingest and final hypothesis vs ground truth.
LLM internals (tool calls, reasoning, token counts) are visible in MLflow UI:
    mlflow server --port 5001
    open http://localhost:5001 → experiment 'llmwiki-eval' → Traces tab

Usage:
    python diagnose.py                       # first oracle question
    python diagnose.py --index 4
    python diagnose.py --question_id <id>
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("llmwiki-eval")
mlflow.openai.autolog()

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_MAGENTA= "\033[95m"
_BLUE   = "\033[94m"
_RED    = "\033[91m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _hr(char: str = "-", label: str = "", color: str = "") -> None:
    width = 70
    color = color or {
        "#": _BOLD + _GREEN,
        "=": _BOLD + _MAGENTA,
        "-": _BOLD + _BLUE,
    }.get(char, "")
    if label:
        side = (width - len(label) - 2) // 2
        print(f"\n{color}{char * side} {label} {char * side}{_RESET}")
    else:
        print(f"{color}{char * width}{_RESET}")


from db import WikiDB
from ingest import ingest_session
from answer_generator import generate
from session_formatter import format_session

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

WIKI_DIR = Path(__file__).parent / "wiki" / "_diagnose"


async def main(args: argparse.Namespace) -> None:
    data_file = Path(args.data_dir) / DATASET_FILES[args.dataset]
    questions = json.loads(data_file.read_text())

    if args.question_id:
        q = next((q for q in questions if q["question_id"] == args.question_id), None)
        if not q:
            print(_c(_RED, f"question_id '{args.question_id}' not found in {args.dataset}"))
            sys.exit(1)
    else:
        q = questions[args.index]

    _hr("#", "QUESTION")
    print(f"  id:       {_c(_BOLD, q['question_id'])}")
    print(f"  type:     {q['question_type']}")
    print(f"  date:     {q.get('question_date', 'N/A')}")
    print(f"  Q:        {_c(_BOLD, q['question'])}")
    print(f"  answer:   {_c(_GREEN, q['answer'])}")
    print(f"  sessions: {len(q['haystack_sessions'])}")

    wiki = WikiDB(WIKI_DIR)
    wiki.reset()

    sessions = sorted(
        zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]),
        key=lambda x: x[2],
    )

    for session_id, turns, date in sessions:
        _hr("#", f"INGEST  {session_id}  |  {date}  |  {len(turns)} turns")
        print(_c(_DIM, "[SESSION TEXT]"))
        print(_c(_DIM, format_session(turns, date)[:4000]))

        n = await ingest_session(session_id, turns, date, wiki)

        print(_c(_GREEN, f"\n→ {n} atoms written"))
        _hr("-", "WIKI INDEX")
        print(_c(_CYAN, wiki.read_active_index()))
        _hr("-", "SUBJECTS")
        print(_c(_CYAN, json.dumps(wiki.list_subjects(), indent=2)))

    _hr("#", "ANSWER GENERATION")

    hypothesis = await generate(
        question=q["question"],
        wiki=wiki,
        as_of=q.get("question_date"),
    )

    _hr("#", "RESULT")
    print(f"  hypothesis:   {_c(_BOLD + _YELLOW, hypothesis)}")
    print(f"  ground truth: {_c(_BOLD + _GREEN, q['answer'])}")
    _hr("#")
    print(_c(_DIM, "Traces at: http://localhost:5001 → experiment 'llmwiki-eval' → Traces tab"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose the agent eval pipeline on a single LongMemEval question."
    )
    parser.add_argument("--dataset", default="oracle", choices=list(DATASET_FILES.keys()))
    parser.add_argument("--data_dir", default="LongMemEval/data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--question_id", type=str, default=None)
    asyncio.run(main(parser.parse_args()))
