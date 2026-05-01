"""Synthesis agent: produce a final answer from retrieved atoms.

Starts with the atoms provided by selection, but can call read_atom to fetch
additional context mid-reasoning if it discovers a gap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import Agent, Runner
from db import WikiDB
from llm_client import get_model, get_model_settings, clean_json
from models import SynthesisResult
from tools import SYNTHESIS_TOOLS

SYNTHESIS_SYSTEM = """\
You are a memory synthesis agent. Given a set of memory atoms and a question, produce a concise answer.

You have one tool:
  read_atom(atom_id) — fetch an additional atom if you need more context mid-reasoning

Workflow:
1. Consider temporal ordering carefully:
   - 'valid_from' is the date the SESSION was recorded, not when the event occurred.
     Use it to resolve relative time expressions in the atom content.
   - When content says "last Saturday", "two months ago", "yesterday", etc.,
     compute the actual event date by offsetting from that atom's valid_from.
   - For conflicting facts about the same subject, the atom with the later valid_from
     is the more recent and should take precedence.
   - For event ordering questions, compare resolved event dates, not valid_from dates.
3. If you realize mid-reasoning that you need another atom (e.g. to resolve a conflict or
   check a date), call read_atom — but only if genuinely needed.
4. Set 'answer' to a short, direct response (1-2 sentences) for automated evaluation.
5. Base your answer strictly on the atoms — do not hallucinate or add outside knowledge.
6. If the answer is not present in the atoms, set 'answer' to exactly:
   "I don't have this information"

Your final answer MUST be ONLY valid JSON. Do NOT write any prose or explanation.
Final answer format (no code fences):
{"synthesis": "brief summary", "answer": "...", "atom_ids_used": ["id1", ...]}\
"""

_SYNTHESIS_AGENT = Agent(
    name="synthesis",
    instructions=SYNTHESIS_SYSTEM,
    tools=SYNTHESIS_TOOLS,
    model=get_model(),
    model_settings=get_model_settings(reasoning=True),
)


async def synthesize(
    question: str,
    atom_texts: list[str],
    wiki: WikiDB,
) -> SynthesisResult:
    """Run the synthesis agent over the provided atoms and return a SynthesisResult."""
    atoms_block = "\n\n".join(atom_texts) if atom_texts else "No relevant atoms found."
    payload = f"Memory Atoms:\n{atoms_block}\n\nQuestion: {question}"

    try:
        result = await Runner.run(_SYNTHESIS_AGENT, payload, context=wiki)
        return SynthesisResult.model_validate_json(clean_json(result.final_output))
    except Exception:
        return SynthesisResult(answer="I don't have this information", synthesis="", atom_ids_used=[])
