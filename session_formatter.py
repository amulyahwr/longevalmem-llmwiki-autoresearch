"""Format a LongMemEval haystack session (list of turns + date) into a single text string."""


def format_session(turns: list[dict], date: str) -> str:
    """Convert a session's turns and date into a timestamped dialogue string.

    Args:
        turns: List of {"role": "user"|"assistant", "content": str} dicts.
        date: Session date string, e.g. "2023/05/20".

    Returns:
        A single string with date header followed by role-prefixed turns.
    """
    lines = [f"[Date: {date}]"]
    for turn in turns:
        role = turn["role"].capitalize()
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)
