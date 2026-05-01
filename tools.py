"""Wiki tool definitions for the agent harness.

Each tool receives the WikiDB instance via RunContextWrapper[WikiDB],
passed as `context=wiki` in Runner.run().
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import RunContextWrapper, function_tool
from db import WikiDB

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "this", "that",
}


def _query_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOPWORDS]


_MAX_SEARCH_RESULTS = 5
_MAX_SNIPPET_CHARS = 120
_MAX_ATOM_CHARS = 400
_MAX_LIST_CHARS = 1500


@function_tool(strict_mode=False)
def search_atoms(ctx: RunContextWrapper[WikiDB], query: str) -> str:
    """Search all atom contents for a keyword or phrase. Returns matching atom_ids and snippets."""
    wiki: WikiDB = ctx.context
    words = _query_words(query)
    if not words:
        return "No atoms found matching query."
    index = wiki.load_word_index()
    hits: dict[str, int] = {}
    for word in words:
        for atom_id in index.get(word, []):
            hits[atom_id] = hits.get(atom_id, 0) + 1
    ranked = sorted(hits.items(), key=lambda x: -x[1])[:_MAX_SEARCH_RESULTS]
    if not ranked:
        return "No atoms found matching query."
    results = []
    for atom_id, _ in ranked:
        content = wiki.read_atom(atom_id)
        first_word = next((w for w in words if w in content.lower()), words[0])
        idx = content.lower().find(first_word)
        snippet = content[max(0, idx - 20): idx + _MAX_SNIPPET_CHARS].replace("\n", " ")
        results.append(f"[{atom_id}]: ...{snippet}...")
    return "\n".join(results)


@function_tool(strict_mode=False)
def read_atom(ctx: RunContextWrapper[WikiDB], atom_id: str) -> str:
    """Read the full content of a specific atom, including its frontmatter (valid_from, kind, subject)."""
    wiki: WikiDB = ctx.context
    content = wiki.read_atom(atom_id)
    if not content:
        return "Atom not found."
    return content if len(content) <= _MAX_ATOM_CHARS else content[:_MAX_ATOM_CHARS] + "\n… (truncated)"


@function_tool(strict_mode=False)
def list_subjects(ctx: RunContextWrapper[WikiDB]) -> str:
    """Show all current subject → atom_id mappings in the wiki."""
    wiki: WikiDB = ctx.context
    subjects = wiki.list_subjects()
    if not subjects:
        return "No subjects yet."
    raw = "\n".join(f"  {k}: {v}" for k, v in subjects.items())
    return raw if len(raw) <= _MAX_LIST_CHARS else raw[:_MAX_LIST_CHARS] + "\n… (truncated)"


@function_tool(strict_mode=False)
def list_all_atoms(ctx: RunContextWrapper[WikiDB]) -> str:
    """List every atom_id in the wiki with its subject and kind. Use for a broad overview."""
    wiki: WikiDB = ctx.context
    index = wiki.read_active_index()
    if index.strip() == "# Index":
        return "Wiki is empty."
    return index if len(index) <= _MAX_LIST_CHARS else index[:_MAX_LIST_CHARS] + "\n… (truncated)"


INGEST_TOOLS = [search_atoms, read_atom, list_subjects]
SELECTION_TOOLS = [search_atoms, read_atom, list_all_atoms]
SYNTHESIS_TOOLS = [read_atom]
