"""Ingest agent: LongMemEval turns → wiki atoms.

Instead of one-shot extraction, the agent can inspect the existing wiki
before deciding what to extract — searching atom contents, reading specific
atoms — so superseding and deduplication are content-aware, not subject-name-dependent.
"""

import asyncio
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agents import Agent, Runner
from db import WikiDB
from llm_client import get_model, get_model_settings, clean_json
from models import TurnPairResult
from session_formatter import format_session
from tools import INGEST_TOOLS

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

PAIR_INGEST_SYSTEM = """\
You are a memory extraction agent. Read ONE user-assistant exchange and extract \
all durable, personalized facts, preferences, events, and beliefs — from BOTH the user and the assistant.

You have tools to inspect the existing wiki before extracting:
  search_atoms(query)  — search atom contents for a keyword; returns matching atom_ids + snippets
  read_atom(atom_id)   — read a specific atom's full content
  list_subjects()      — show all current subject → atom_id mappings

Two categories of atoms to extract:

1. USER-sourced (source: "user"):
   - Facts, events, or beliefs the user explicitly states about themselves.
   - Preferences the user expresses explicitly OR implicitly (e.g., the user follows up on a specific
     option, says "I'll try that", or accepts one recommendation over others — extract as a preference).

2. ASSISTANT-sourced (source: "assistant"):
   - Personalized information the assistant provides specifically for this user:
     itineraries, schedules, or plans it created; account/flight/booking details it looked up;
     recommendations it gave that the user acknowledged or accepted.
   - Do NOT extract general explanations, how-to advice, or facts that apply to anyone.

Rules for each atom:
  - Each atom must be a single self-contained statement about the user.
  - content    : a single self-contained statement about the user
  - kind       : one of fact, preference, event, belief
  - subject    : canonical "user's <noun>" — see subject rules below
  - source     : "user" or "assistant"
  - supersedes : atom_id this replaces, or null

Subject naming rules:
  - User atoms: use the most general term that still uniquely identifies the topic
    (e.g. "user's location", "user's job").
  - Assistant atoms: use a SPECIFIC subject that will not collide with a user-stated atom
    on the same broad topic (e.g. "user's flight itinerary" not "user's travel",
    "user's clinic appointment" not "user's location").
  - Before assigning a subject, call list_subjects() to check existing subjects.
    If an existing subject is a user-sourced atom and the new atom is assistant-sourced
    (or vice versa) and they are NOT contradictory updates, use a more specific subject
    rather than claiming the same key — claiming the same subject triggers supersession.

If nothing is worth extracting, output: {"atoms": []}

Output ONLY the JSON object — no code fences, no commentary.
Output exactly: {"atoms": [{"content": "...", "kind": "...", "subject": "...", "source": "user", "supersedes": null}, ...]}\
"""

_INGEST_AGENT = Agent(
    name="ingest",
    instructions=PAIR_INGEST_SYSTEM,
    tools=INGEST_TOOLS,
    model=get_model(),
    model_settings=get_model_settings(reasoning=False),
)


def _parse_date(date_str: str) -> datetime:
    for fmt in (
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromtimestamp(0, tz=timezone.utc)


async def ingest_turn_pair(
    pair_id: str,
    user_turn: dict,
    asst_turn: dict,
    date: str,
    wiki: WikiDB,
) -> int:
    """Extract atoms from a single user-assistant exchange. Returns count of atoms written."""
    pair_text = format_session([user_turn, asst_turn], date)
    try:
        result = await Runner.run(_INGEST_AGENT, pair_text, context=wiki)
        atoms = TurnPairResult.model_validate_json(
            clean_json(result.final_output)
        ).atoms
    except Exception:
        return 0

    valid_from = _parse_date(date)
    for j, atom in enumerate(atoms):
        atom_id = f"{pair_id}_{j:02d}"
        supersedes = atom.supersedes

        async with wiki.write_lock:
            if atom.subject:
                old_id = wiki.register_subject(atom.subject, atom_id)
                if old_id:
                    wiki.mark_superseded(old_id, superseded_by=atom_id, valid_until=valid_from)
                    supersedes = old_id
            elif atom.supersedes:
                wiki.mark_superseded(atom.supersedes, superseded_by=atom_id, valid_until=valid_from)

            wiki.write_atom(
                atom_id=atom_id,
                content=atom.content,
                kind=atom.kind,
                valid_from=valid_from,
                subject=atom.subject,
                supersedes=supersedes,
                source=atom.source,
            )
            wiki.update_word_index(atom_id, atom.content)
            wiki.update_index(
                atom_id,
                atom.content[:80].replace("\n", " "),
                atom.kind,
                valid_from,
                subject=atom.subject,
            )
    return len(atoms)


async def ingest_session(
    session_id: str,
    turns: list[dict],
    date: str,
    wiki: WikiDB,
) -> int:
    """Process each user-assistant turn pair sequentially. Returns total atom count."""
    user_turns = turns[::2]
    asst_turns = turns[1::2]
    count = 0
    for i, (u, a) in enumerate(zip(user_turns, asst_turns)):
        pair_id = f"{session_id}_{i:03d}"
        count += await ingest_turn_pair(pair_id, u, a, date, wiki)
    wiki.append_log("ingest", f"session={session_id} atoms={count}")
    return count


async def main(args: argparse.Namespace) -> None:
    data_file = Path(args.data_dir) / DATASET_FILES[args.dataset]
    questions = json.loads(data_file.read_text())
    if args.limit:
        questions = questions[: args.limit]

    wiki = WikiDB(Path(__file__).parent / "wiki" / "_standalone")
    for q in questions:
        wiki.reset()
        for session_id, turns, date in sorted(
            zip(q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]),
            key=lambda x: x[2],
        ):
            n = await ingest_session(session_id, turns, date, wiki)
            print(f"  [{session_id}] {n} atoms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest agent — LongMemEval sessions into wiki."
    )
    parser.add_argument(
        "--dataset", default="oracle", choices=list(DATASET_FILES.keys())
    )
    parser.add_argument("--data_dir", default="eval/longmemeval/data")
    parser.add_argument("--limit", type=int, default=None)
    asyncio.run(main(parser.parse_args()))
