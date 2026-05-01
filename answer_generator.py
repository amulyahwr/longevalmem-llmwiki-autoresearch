"""Orchestrate selection + synthesis agents to generate a hypothesis."""
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import WikiDB
from selection import select
from synthesis import synthesize

_MAX_ATOMS = 15
_MAX_ATOM_CHARS = 500

_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _normalize_date(date_str: str) -> str:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str[:10]


def _atom_valid_at(atom_content: str, as_of: str) -> bool:
    """Return True if the atom's valid_from is on or before as_of (YYYY-MM-DD)."""
    match = re.search(r"^valid_from:\s*(.+)$", atom_content, re.MULTILINE)
    if not match:
        return True
    valid_from = match.group(1).strip()[:10]
    return valid_from <= as_of


async def generate(question: str, wiki: WikiDB, as_of: str | None = None) -> str:
    """Run selection → as_of filter → synthesis and return the hypothesis string."""
    as_of_norm = _normalize_date(as_of) if as_of else None

    if not wiki.list_atoms():
        return "I don't have this information"

    # Step 1: selection agent finds relevant atom_ids
    selection = await select(question, wiki, as_of=as_of_norm)

    # Step 2: read atoms, apply as_of filter in code (deterministic)
    atom_texts: list[str] = []
    for atom_id in selection.relevant_atom_ids[:_MAX_ATOMS]:
        content = wiki.read_atom(atom_id)
        if not content:
            continue
        if as_of_norm and not _atom_valid_at(content, as_of_norm):
            continue
        if len(content) > _MAX_ATOM_CHARS:
            content = content[:_MAX_ATOM_CHARS] + "\n… (truncated)"
        atom_texts.append(f"[{atom_id}]\n{content}")

    # Step 3: synthesis agent reasons over atoms, can fetch more if needed
    result = await synthesize(question, atom_texts, wiki)
    return result.answer
