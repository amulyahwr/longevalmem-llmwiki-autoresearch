"""Pydantic models for all LLM calls and eval output in the wiki-based harness."""
from pydantic import BaseModel


# --- Ingest models ---

class ExtractedAtom(BaseModel):
    content: str
    kind: str     # "fact" | "preference" | "event" | "belief"
    subject: str  # short noun phrase identifying the topic, e.g. "user's location"
    supersedes: str | None = None  # atom_id this contradicts/updates (LLM fallback)


class AtomExtractionResult(BaseModel):
    reasoning: str  # think through what is worth extracting before listing atoms
    atoms: list[ExtractedAtom]


class TurnPairResult(BaseModel):
    atoms: list[ExtractedAtom] = []


# --- Query models ---

class AtomSelectionResult(BaseModel):
    relevant_atom_ids: list[str]
    reasoning: str = ""


class SynthesisResult(BaseModel):
    synthesis: str       # brief 1-2 sentence reasoning trace
    answer: str          # short extractive answer for eval; "I don't have this information" if absent
    atom_ids_used: list[str]


# --- Eval output ---

class EvalOutput(BaseModel):
    question_id: str
    hypothesis: str
