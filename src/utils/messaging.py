"""Messaging helpers for parsing and sanitizing agent communications."""

import re
from typing import Dict


def parse_tags(payload: str) -> Dict[str, str]:
    """Parse XML-style tags (``<tag>value</tag>``) into a dictionary.

    Args:
        payload: Text containing XML-style tags

    Returns:
        Dictionary mapping tag names to their content
    """
    tags = re.findall(r"<(.*?)>(.*?)</\1>", payload, flags=re.DOTALL)
    return {key: value.strip() for key, value in tags}


def sanitize_action(raw_action: str) -> str:
    """Normalize an LLM response to a single TextWorld command.

    Args:
        raw_action: Raw text from LLM response

    Returns:
        Cleaned action string suitable for TextWorld environment
    """
    text = raw_action.strip()

    # Remove markdown code blocks if present
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`")

    # If multiple lines, find first non-comment line
    if "\n" in text:
        candidates = [line.strip() for line in text.splitlines() if line.strip()]
        for candidate in candidates:
            if not candidate.startswith(("#", "-", "*", "[", "//")):
                return candidate
        if candidates:
            return candidates[0]

    return text
