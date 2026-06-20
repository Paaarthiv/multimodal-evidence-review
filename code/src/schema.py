"""Output schema: allowed enum values, column order, and validation/coercion.

The grader checks the exact 14-column schema and allowed values. Every value we
emit is clamped to the allowed set here, so a hallucinated label can never leak
into output.csv. Unknown/out-of-vocab values collapse to the safe defaults
(`unknown` / `none`) defined by the problem statement.
"""
from __future__ import annotations

from typing import Iterable

# Exact output column order (problem_statement.md "Required output").
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_OBJECTS = {"car", "laptop", "package"}

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}
SEVERITY_ORDER = ["none", "low", "medium", "high"]  # ordinal (unknown excluded)

# Canonical order for emitting risk_flags (matches problem statement list + gold rows).
RISK_FLAG_ORDER = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _norm(v) -> str:
    return str(v).strip().lower().replace(" ", "_") if v is not None else ""


def coerce_enum(value, allowed: set, default: str) -> str:
    v = _norm(value)
    return v if v in allowed else default


def coerce_claim_status(value) -> str:
    return coerce_enum(value, CLAIM_STATUS, "not_enough_information")


def coerce_issue_type(value) -> str:
    return coerce_enum(value, ISSUE_TYPES, "unknown")


def coerce_object_part(value, claim_object: str) -> str:
    allowed = OBJECT_PARTS.get(_norm(claim_object), set())
    return coerce_enum(value, allowed, "unknown")


def coerce_severity(value) -> str:
    return coerce_enum(value, SEVERITY, "unknown")


def coerce_bool(value) -> str:
    """Return the literal string 'true'/'false' as the CSV expects."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if _norm(value) in {"true", "1", "yes", "y"} else "false"


def coerce_risk_flags(values: Iterable[str]) -> str:
    """Dedupe + validate risk flags; join with ';'. 'none' if empty.

    'none' is mutually exclusive with real flags.
    """
    valid = set()
    for v in values or []:
        nv = _norm(v)
        if nv in RISK_FLAGS and nv != "none":
            valid.add(nv)
    ordered = [f for f in RISK_FLAG_ORDER if f in valid]
    return ";".join(ordered) if ordered else "none"


def coerce_image_ids(values: Iterable[str]) -> str:
    out = []
    for v in values or []:
        nv = str(v).strip()
        if nv and nv.lower() != "none" and nv not in out:
            out.append(nv)
    return ";".join(out) if out else "none"
