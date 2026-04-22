"""Validate and sanitize agent text before sending it to the client."""

from __future__ import annotations

import re

# Code fences that could leak agent scratch work.
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py|javascript|js|ts|tsx|bash|sh)\s*[\s\S]*?```",
    re.IGNORECASE,
)

_THOUGHT_ACTION_RE = re.compile(
    r"(?im)^\s*(Thought|Action|Action Input|Observation)\s*:\s*.*$",
    re.MULTILINE,
)

_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>[\s\S]*?</script>", re.IGNORECASE)
_JS_URL_RE = re.compile(r"javascript\s*:", re.IGNORECASE)
# Rough strip of inline event handlers (onclick=, onerror=, ...)
_EVENT_HANDLER_RE = re.compile(r"\s+on[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)

_NULL_RE = re.compile(r"\x00")


def sanitize_agent_text(text: str) -> str:
    """
    Remove noisy agent scaffolding and basic HTML/JS injection patterns.
    """
    if not text:
        return text
    cleaned = _NULL_RE.sub("", text)
    cleaned = _SCRIPT_BLOCK_RE.sub("", cleaned)
    cleaned = _JS_URL_RE.sub("", cleaned)
    cleaned = _EVENT_HANDLER_RE.sub("", cleaned)
    cleaned = _CODE_FENCE_RE.sub("", cleaned)
    cleaned = _THOUGHT_ACTION_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
