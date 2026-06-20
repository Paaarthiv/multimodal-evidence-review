"""Evidence-requirements lookup.

Maps a claim (object + issue family + part) to the relevant minimum-evidence
rules from evidence_requirements.csv. The selected rule text is injected into the
vision prompt so the model judges `evidence_standard_met` against the *stated*
standard rather than an ad-hoc one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from . import io_utils

# issue_type -> issue family keyword (matched against the `applies_to` column)
_FAMILY_BY_OBJECT_ISSUE = {
    "car": {
        "dent": "dent or scratch",
        "scratch": "dent or scratch",
        "crack": "crack, broken, or missing part",
        "glass_shatter": "crack, broken, or missing part",
        "broken_part": "crack, broken, or missing part",
        "missing_part": "crack, broken, or missing part",
    },
    "laptop": {
        "crack": "screen, keyboard, or trackpad",
        "glass_shatter": "screen, keyboard, or trackpad",
        "stain": "screen, keyboard, or trackpad",
        "missing_part": "screen, keyboard, or trackpad",
        "scratch": "screen, keyboard, or trackpad",
        "broken_part": "hinge, lid, corner, body, or port",
        "dent": "hinge, lid, corner, body, or port",
    },
    "package": {
        "crushed_packaging": "crushed, torn, or seal damage",
        "torn_packaging": "crushed, torn, or seal damage",
        "water_damage": "water, stain, or label damage",
        "stain": "water, stain, or label damage",
        "missing_part": "contents or inner item",
    },
}


class EvidenceRules:
    def __init__(self, rows: List[Dict[str, str]]):
        self.rows = rows
        self._by_key = {(r["claim_object"], r["applies_to"]): r for r in rows}

    @classmethod
    def load(cls, path: Path) -> "EvidenceRules":
        return cls(io_utils.read_csv(path))

    def relevant_text(self, claim_object: str, issue_type: str, part: str,
                      multi_image: bool) -> str:
        """Return a compact, newline-joined block of the applicable rules."""
        selected: List[str] = []

        def add(obj: str, applies: str):
            r = self._by_key.get((obj, applies))
            if r and r["minimum_image_evidence"] not in selected:
                selected.append(r["minimum_image_evidence"])

        # Always: general object/part visibility + reviewability/trust.
        add("all", "general claim review")
        add("all", "reviewability")
        if multi_image:
            add("all", "multi-image rows")

        # Object/issue-specific family.
        fam = _FAMILY_BY_OBJECT_ISSUE.get(claim_object, {}).get(issue_type)
        if fam:
            add(claim_object, fam)
        else:
            # Fall back to every rule for the object so the model still has a standard.
            for r in self.rows:
                if r["claim_object"] == claim_object and r["minimum_image_evidence"] not in selected:
                    selected.append(r["minimum_image_evidence"])

        # Package contents / car identity special cases worth surfacing.
        if claim_object == "package" and part in {"contents", "item"}:
            add("package", "contents or inner item")
        if claim_object == "car":
            add("car", "vehicle identity or orientation")

        return "\n".join(f"- {s}" for s in selected)
