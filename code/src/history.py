"""User-history risk context.

History can *add* risk flags and justification context, but per the problem
statement it must NOT override clear visual evidence by itself. So this module
only ever produces flags + a short context string; the decision layer (fusion)
is responsible for never flipping a clearly-supported visual verdict on history
alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from . import io_utils


class UserHistory:
    def __init__(self, rows):
        self._by_user: Dict[str, dict] = {r["user_id"]: r for r in rows}

    @classmethod
    def load(cls, path: Path) -> "UserHistory":
        return cls(io_utils.read_csv(path))

    def get(self, user_id: str) -> Optional[dict]:
        return self._by_user.get(user_id)

    def risk_context(self, user_id: str) -> dict:
        """Return {'flags': [..], 'summary': str, 'elevated': bool}.

        flags is a subset of {user_history_risk, manual_review_required}.
        """
        row = self._by_user.get(user_id)
        if not row:
            return {"flags": [], "summary": "No history on file for this user.", "elevated": False}

        raw_flags = (row.get("history_flags") or "none").strip()
        flags = []
        if "user_history_risk" in raw_flags:
            flags.append("user_history_risk")
        if "manual_review_required" in raw_flags:
            flags.append("manual_review_required")

        # Derive an elevated signal from counts too (heuristic, not an override).
        try:
            past = int(row.get("past_claim_count", "0") or 0)
            rejected = int(row.get("rejected_claim", "0") or 0)
        except ValueError:
            past, rejected = 0, 0
        reject_ratio = (rejected / past) if past else 0.0
        if reject_ratio >= 0.4 and "user_history_risk" not in flags:
            flags.append("user_history_risk")

        summary = (row.get("history_summary") or "").strip()
        return {
            "flags": flags,
            "summary": summary,
            "elevated": bool(flags),
            "reject_ratio": round(reject_ratio, 2),
        }
