"""Selection agent: find relevant atom_ids for a question by actively navigating the wiki.

Unlike one-shot selection (which pattern-matches subject names), the agent can:
  - grep atom contents for entities mentioned in the question
  - read specific atoms to understand context and temporal ordering
  - follow temporal leads (e.g. read service atom → find its date → search for subsequent events)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import Agent, Runner
from db import WikiDB
from llm_client import get_model, get_model_settings, clean_json
from models import AtomSelectionResult
from tools import SELECTION_TOOLS

SELECTION_SYSTEM = """\
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
"""

_SELECTION_AGENT = Agent(
    name="selection",
    instructions=SELECTION_SYSTEM,
    tools=SELECTION_TOOLS,
    model=get_model(),
    model_settings=get_model_settings(reasoning=True),
)


async def select(
    question: str,
    wiki: WikiDB,
    as_of: str | None = None,
) -> AtomSelectionResult:
    """Run the selection agent and return relevant atom_ids."""
    payload = f"Question: {question}"
    if as_of:
        payload += f"\n\nTemporal constraint: only consider atoms with event dates on or before {as_of}."

    try:
        result = await Runner.run(_SELECTION_AGENT, payload, context=wiki)
        return AtomSelectionResult.model_validate_json(clean_json(result.final_output))
    except Exception:
        return AtomSelectionResult(relevant_atom_ids=[], reasoning="")
