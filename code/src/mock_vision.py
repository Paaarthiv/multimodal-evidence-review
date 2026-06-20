"""Offline deterministic stand-in for the vision model.

This is NOT a real classifier — it cannot see images. It exists only so the full
pipeline (CSV I/O, evidence lookup, history fusion, schema clamping, output
writing) can be exercised and unit-tested without API keys. Real runs use the
OpenAI/Gemini providers. The mock infers a plausible claimed issue/part from the
conversation text and assumes the image supports it.
"""
from __future__ import annotations

import re

_ISSUE_KEYWORDS = [
    ("glass_shatter", r"shatter"),
    ("crack", r"crack|cracked|grieta|toot|phati"),
    ("dent", r"dent|dab|dented"),
    ("scratch", r"scratch|scrape|ray?on|mark"),
    ("broken_part", r"broken|broke|toot|missing or broken"),
    ("missing_part", r"missing|falta|nahi"),
    ("torn_packaging", r"torn|tear|phati|open"),
    ("crushed_packaging", r"crush|crushed|dab"),
    ("water_damage", r"water|wet|liquid|gela"),
    ("stain", r"stain|oil|mark"),
]

_PART_KEYWORDS = [
    ("front_bumper", r"front bumper|parachoques (delantero|frontal)"),
    ("rear_bumper", r"rear bumper|back bumper|parachoques (trasero|de atras)"),
    ("windshield", r"windshield|front glass"),
    ("side_mirror", r"side mirror|mirror|espejo"),
    ("headlight", r"headlight|front light"),
    ("taillight", r"taillight|back light|tail light"),
    ("hood", r"hood|bonnet"),
    ("door", r"door|puerta"),
    ("screen", r"screen|display|pantalla"),
    ("keyboard", r"keyboard|teclas|teclado|keys"),
    ("trackpad", r"trackpad|touchpad"),
    ("hinge", r"hinge"),
    ("lid", r"lid"),
    ("corner", r"corner"),
    ("seal", r"seal"),
    ("label", r"label"),
    ("contents", r"contents|item inside|product inside|inside"),
    ("package_corner", r"package corner|box corner|corner"),
    ("package_side", r"package (side|surface)|box surface"),
    ("body", r"body|panel"),
]


def _first_match(text, table, default):
    t = text.lower()
    for label, pat in table:
        if re.search(pat, t):
            return label
    return default


def mock_response(user_prompt: str, n_images: int) -> dict:
    issue = _first_match(user_prompt, _ISSUE_KEYWORDS, "unknown")
    part = _first_match(user_prompt, _PART_KEYWORDS, "unknown")
    ids = [f"img_{i+1}" for i in range(max(1, n_images))]
    return {
        "claimed_issue_type": issue,
        "claimed_object_part": part,
        "claimed_severity": "medium",
        "is_multi_part_claim": False,
        "conversation_language": "unknown",
        "model_detected_instruction_text": False,
        "per_image": [
            {
                "image_id": i, "visible_object": "unknown", "visible_part": part,
                "visible_issue_type": issue, "quality_issues": [], "authenticity_issues": [],
                "shows_claimed_part": True, "embedded_text_instruction": False,
                "supports_claim": True, "notes": "mock",
            } for i in ids
        ],
        "issue_type": issue,
        "object_part": part,
        "evidence_sufficient": True,
        "valid_image": True,
        "claim_status": "supported",
        "severity": "medium",
        "supporting_image_ids": ids[:1],
        "risk_flags": [],
        "evidence_reason": "[MOCK] assumed claimed part visible.",
        "justification": f"[MOCK] image {ids[0]} assumed to show {issue} on {part}.",
    }
