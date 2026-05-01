# Claude Instructions for This Python Repository

## Project Overview

This is a Python codebase. Claude is used for:

- Debugging issues
- Building new features
- Refactoring and improving existing code

Assume production-quality expectations unless stated otherwise.

---

## Core Principles

- Prioritize correctness over cleverness
- Make minimal, targeted changes
- Preserve existing behavior unless explicitly asked to change it
- Always understand the code before modifying it

---

## Python Code Style

- Follow PEP8 conventions
- Use clear, descriptive variable and function names
- Prefer small, single-purpose functions
- Avoid deeply nested logic
- Use type hints where possible

### Preferred Patterns

- Use list/dict comprehensions when readable
- Use context managers (`with` statements)
- Use `dataclasses` or typed models when appropriate

### Avoid

- Unnecessary abstraction
- Over-engineering
- Global mutable state

---

## When Debugging

- Identify the root cause before proposing a fix
- Do not guess — reason from the code
- Use logs, stack traces, and error messages
- Reproduce the issue mentally or via code flow

### Debugging Output

- Clearly explain:
  1. What is wrong
  2. Why it is happening
  3. The exact fix

---

## When Building Features

- Follow existing project structure and patterns
- Reuse existing utilities and modules when possible
- Keep changes isolated and modular
- Include error handling and edge cases

### Feature Code Should

- Be readable and maintainable
- Include docstrings where useful
- Not break existing functionality

---

## When Refactoring

- Do not change behavior
- Improve readability and structure
- Reduce duplication
- Keep diffs small and reviewable

---

## File & Change Rules

- Do NOT rewrite entire files unless necessary
- Only modify relevant sections
- Do NOT introduce unrelated changes
- Respect existing architecture
- When code changes, analyze affected files and suggest if updates or refactoring are needed in related files

---

## LLM Calls

All calls to `chat()` in `backend/compiler/llm_client.py` **must** pass a Pydantic model class as `response_format`. Never call `chat()` without it.

```python
# Correct
output = await chat(SYSTEM, payload, response_format=MyPydanticModel)

# Wrong — never do this
output = await chat(SYSTEM, payload)
```

- Define a `RootModel` or `BaseModel` subclass that exactly describes the expected JSON shape
- Pass the class itself (not an instance) — `chat()` extracts the JSON schema automatically
- The Pydantic model serves as the contract between the prompt and the parser; keep them in sync

---

## Dependencies

- Prefer standard library when possible
- Do NOT add new dependencies unless justified
- If adding one, explain why

---

## Testing Awareness

- Do not remove or break existing tests
- If tests fail, fix the code (or explain clearly)
- Suggest tests if missing but do not over-generate

---

## Response Style Rules

- Do NOT include summaries at the end of responses
- Do NOT restate what was changed unless explicitly asked
- Do NOT add “summary”, “in summary”, or recap sections
- Keep responses focused and minimal

When providing code changes:

- Output only the necessary explanation + code
- Avoid verbose descriptions

---

## Strict Output Control

- Never include a summary section
- Never explain obvious changes
- Do not add extra commentary beyond what is necessary
- Keep responses concise and to the point

---

## Summary

Claude should act as a careful, senior Python engineer:

- Debug precisely
- Build clean features
- Refactor safely
- Respect the existing codebase
- Analyze affected files when code changes and suggest required updates or refactoring
