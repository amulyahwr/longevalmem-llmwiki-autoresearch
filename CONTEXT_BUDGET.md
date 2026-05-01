# Context Budget — harness-agent

**Model:** Gemma (local, LM Studio)  
**Context window:** 8 192 tokens  
**Char → token ratio:** ~4 chars / token (English conversational text)  
**Rule of thumb:** fixed overhead + variable inputs + tool history + output ≤ 8 192

---

## Overview

| Stage | Fixed tokens | Variable tokens (max) | Tool history | Output | Headroom |
|-------|-------------|----------------------|--------------|--------|----------|
| **Ingest** (per pair) | ~325 | ~100 | ~200 / round | ~75 | ~7 067 |
| **Selection** | ~325 | ~55 | ~175 / round | ~150 | ~7 487 |
| **Synthesis** | ~200 | ~1 540 | ~100 / round | ~200 | ~6 252 |

*Tool history is per tool-call round (assistant message + result). Headroom = budget for tool rounds before overflow.*

---

## Stage 1 — Ingest Agent (turn-pair mode)

Each user-assistant turn pair is ingested as a separate agent call.  
A 12-turn session produces 6 agent calls instead of 1, but each call is far smaller.

```
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT WINDOW  (8 192 tokens)  — per turn pair            │
│                                                             │
│  ┌── FIXED ──────────────────────────────────────────────┐  │
│  │  System prompt (PAIR_INGEST_SYSTEM)   ~175 tok        │  │
│  │  Tool schemas × 3                     ~150 tok        │  │
│  │  ─────────────────────────────────────────────────    │  │
│  │  Fixed subtotal                        325 tok        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── VARIABLE (user turn) ───────────────────────────────┐  │
│  │  Turn pair text   0 – 400 chars       0 – 100 tok     │  │
│  │  (1 user msg + 1 assistant msg)                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── TOOL ROUNDS (accumulated per call) ─────────────────┐  │
│  │  search_atoms result   ≤ 600 chars    ≤ 150 tok/call  │  │
│  │  read_atom result      ≤ 400 chars    ≤ 100 tok/call  │  │
│  │  list_subjects result  ≤ 1 500 chars  ≤ 375 tok/call  │  │
│  │  assistant turn overhead               ~50 tok/call   │  │
│  │  Knobs: _MAX_SEARCH_RESULTS, _MAX_SNIPPET_CHARS,      │  │
│  │         _MAX_ATOM_CHARS, _MAX_LIST_CHARS in tools.py  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── OUTPUT ─────────────────────────────────────────────┐  │
│  │  TurnPairResult JSON                   ~75 tok        │  │
│  │  {"atom": {"content": "...", ...}} or {"atom": null}  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

| Slot | Chars | Tokens | Fixed / Variable |
|------|-------|--------|-----------------|
| PAIR_INGEST_SYSTEM | ~700 | ~175 | Fixed |
| Tool schemas (3) | ~600 | ~150 | Fixed |
| Turn pair text | 0–400 | 0–100 | **Variable** |
| Tool round (per call) | 0–2 500 | 0–625 | **Variable** |
| Output JSON | ~300 | ~75 | Variable (bounded) |
| **Total worst case** | | **~1 125** | |

> **Headroom at max pair + 1 tool call:** 8 192 − 325 − 100 − 625 − 75 = **7 067 tok**  
> Very safe — even a full list_subjects call leaves >6 000 tok headroom.  
> Trade-off: N/2 agent calls per session (e.g. 6 calls for a 12-turn session).

---

## Stage 2 — Selection Agent

```
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT WINDOW  (8 192 tokens)                             │
│                                                             │
│  ┌── FIXED ──────────────────────────────────────────────┐  │
│  │  System prompt (SELECTION_SYSTEM)     ~175 tok        │  │
│  │  Tool schemas × 3                     ~150 tok        │  │
│  │  ─────────────────────────────────────────────────    │  │
│  │  Fixed subtotal                        325 tok        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── VARIABLE (user turn) ───────────────────────────────┐  │
│  │  Question text        10–50 words      ~40 tok        │  │
│  │  as_of constraint (optional)           ~15 tok        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── TOOL ROUNDS (accumulated per call) ─────────────────┐  │
│  │  search_atoms result   ≤ 600 chars    ≤ 150 tok/call  │  │
│  │  read_atom result      ≤ 400 chars    ≤ 100 tok/call  │  │
│  │  list_all_atoms result ≤ 1 500 chars  ≤ 375 tok/call  │  │
│  │  assistant turn overhead               ~50 tok/call   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── OUTPUT ─────────────────────────────────────────────┐  │
│  │  AtomSelectionResult JSON              ~150 tok       │  │
│  │  {"relevant_atom_ids": [...], "reasoning": "..."}     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

| Slot | Chars | Tokens | Fixed / Variable |
|------|-------|--------|-----------------|
| SELECTION_SYSTEM | ~700 | ~175 | Fixed |
| Tool schemas (3) | ~600 | ~150 | Fixed |
| Question + as_of | ~220 | ~55 | **Variable** |
| Tool round (per call) | 0–2 500 | 0–625 | **Variable** |
| Output JSON | ~600 | ~150 | Variable (bounded) |
| **Total worst case** | | **~1 155** | |

> **Most comfortable stage.** Question is tiny; tool results dominate.  
> Headroom after question + 3 tool calls: 8 192 − 325 − 55 − 3 × 625 − 150 = **5 787 tok**

---

## Stage 3 — Synthesis Agent

```
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT WINDOW  (8 192 tokens)                             │
│                                                             │
│  ┌── FIXED ──────────────────────────────────────────────┐  │
│  │  System prompt (SYNTHESIS_SYSTEM)     ~150 tok        │  │
│  │  Tool schemas × 1 (read_atom)          ~50 tok        │  │
│  │  ─────────────────────────────────────────────────    │  │
│  │  Fixed subtotal                        200 tok        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── VARIABLE (user turn) ───────────────────────────────┐  │
│  │  Selected atoms  0–15 × ≤400 chars    0–1 500 tok     │  │
│  │  Question text                           ~40 tok      │  │
│  │  Knobs: _MAX_ATOMS, _MAX_ATOM_CHARS in                │  │
│  │         answer_generator.py                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── TOOL ROUNDS (rare — only if gap detected) ──────────┐  │
│  │  read_atom result      ≤ 400 chars    ≤ 100 tok/call  │  │
│  │  assistant turn overhead               ~50 tok/call   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌── OUTPUT ─────────────────────────────────────────────┐  │
│  │  SynthesisResult JSON                  ~200 tok       │  │
│  │  {"synthesis": "...", "answer": "...",                │  │
│  │   "atom_ids_used": [...]}                             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

| Slot | Chars | Tokens | Fixed / Variable |
|------|-------|--------|-----------------|
| SYNTHESIS_SYSTEM | ~600 | ~150 | Fixed |
| Tool schemas (1) | ~200 | ~50 | Fixed |
| Atoms (15 × 400 chars) | ~6 000 | ~1 500 | **Variable** |
| Question | ~160 | ~40 | **Variable** |
| Tool round (per call) | ~600 | ~150 | **Variable** |
| Output JSON | ~800 | ~200 | Variable (bounded) |
| **Total worst case (no tool calls)** | | **~1 940** | |

> **Pressure point:** atom payload. At 15 atoms × 400 chars each = 6 000 chars = 1 500 tokens.  
> Headroom: 8 192 − 200 − 1 540 − 200 = **6 252 tok** — still safe.  
> Risk: if `_MAX_ATOMS` or `_MAX_ATOM_CHARS` are raised significantly, this fills fast.

---

## All Knobs In One Place

| Knob | File | Current value | Effect |
|------|------|---------------|--------|
| `_MAX_SEARCH_RESULTS` | `tools.py` | 5 results | Caps search_atoms output per call |
| `_MAX_SNIPPET_CHARS` | `tools.py` | 120 chars | Caps each snippet in search results |
| `_MAX_ATOM_CHARS` | `tools.py` | 400 chars | Caps read_atom output |
| `_MAX_LIST_CHARS` | `tools.py` | 1 500 chars | Caps list_all_atoms / list_subjects output |
| `_MAX_ATOMS` | `answer_generator.py` | 15 atoms | Max atoms passed to synthesis |
| `_MAX_ATOM_CHARS` | `answer_generator.py` | 500 chars | Truncation of each atom in synthesis payload |
| `max_turns` | `llm_client.run_agent()` | 12 turns | Hard cap on tool call rounds |

---

## Overflow Risk by Stage

```
Ingest    █░░░░░░░░░░░░░░░░░░░  very low — turn pair is tiny; N/2 calls per session
Selection ██░░░░░░░░░░░░░░░░░░  very low — question is tiny
Synthesis █████░░░░░░░░░░░░░░░  medium   — atom payload grows with _MAX_ATOMS
```

Synthesis is the stage most sensitive to tuning — raising `_MAX_ATOMS` or `_MAX_ATOM_CHARS` 
directly compresses the headroom available for the output and any mid-reasoning `read_atom` calls.
