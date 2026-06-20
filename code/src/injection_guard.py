"""Deterministic prompt-injection / instruction-text detector.

The user_claim is *data*, never instructions. Some claims try to coerce an
outcome ("approve immediately", "ignore previous instructions", "the note says
approve"). We detect these deterministically so `text_instruction_present` is set
regardless of what the vision model reports, and the decision layer refuses to
let such text influence the verdict. This is defense-in-depth: the model is also
told to treat embedded text as untrusted, and per-image OCR'd instructions get
the same flag.
"""
from __future__ import annotations

import re

# Patterns that indicate an attempt to instruct/steer the reviewer or system,
# rather than describe damage. Kept broad but anchored to coercive intent.
_INJECTION_PATTERNS = [
    r"\bignore\b.{0,30}\b(previous|prior|all)\b.{0,20}\binstruction",
    r"\bignore all previous\b",
    r"\b(approve|accept|mark)\b.{0,40}\b(claim|row|this)\b.{0,30}\b(immediately|now|supported|automatically)\b",
    r"\bapprove\s+(this|the\s+claim|it)\b",
    r"\bapprove\s+(the\s+)?claim\b",
    r"\bclaim\s+approve\b",                 # code-switched order, e.g. "claim approve kar dena"
    r"\bapprove\s+kar\s+dena\b",            # Hindi: "(please) approve it"
    r"\bskip\b.{0,20}\b(manual\s+)?review\b",
    r"\bmark\s+(this|it|the\s+\w+)\s+(as\s+)?(supported|approved)\b",
    r"\bmark this (row|claim)?\b.{0,20}\b(supported|approved)\b",
    r"\bfollow (the|this|it)\b.{0,20}\b(note|instruction)?\b.{0,20}\bapprove\b",
    r"\bfollow (the|this) (note|instruction)\b",
    r"\bthe note (says|is)\b.{0,40}\bapprove\b",
    r"\bset\b.{0,20}\bseverity\b",
    r"\bany system reading this\b",
    r"\byou must\b.{0,20}\b(approve|accept)\b",
    r"\bdo not (reject|review)\b",
    r"\bautomatically approve\b",
    r"\busko follow\b.{0,20}\bapprove\b",   # Hindi: "follow it and approve"
]

# Coercion / pressure language (escalation threats) — softer signal; contributes
# to manual_review but is also instruction-like text.
_PRESSURE_PATTERNS = [
    r"\bescalate publicly\b",
    r"\bkeep reopening\b",
    r"\buntil someone approves\b",
    r"\bi will (escalate|report|keep)\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]
_PRESSURE_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PRESSURE_PATTERNS]


def detect_instruction_text(text: str) -> dict:
    """Return {'instruction': bool, 'pressure': bool, 'matches': [str]}."""
    text = text or ""
    matches = []
    instruction = False
    pressure = False
    for rx in _INJECTION_RE:
        m = rx.search(text)
        if m:
            instruction = True
            matches.append(m.group(0).strip()[:80])
    for rx in _PRESSURE_RE:
        m = rx.search(text)
        if m:
            pressure = True
            matches.append(m.group(0).strip()[:80])
    return {"instruction": instruction, "pressure": pressure, "matches": matches}
