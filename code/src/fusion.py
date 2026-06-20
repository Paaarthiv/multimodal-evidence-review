"""Decision fusion + schema guard.

Combines the vision model's structured output with the deterministic injection
guard and user-history risk context into the final 14-column row. Policy rules
here were validated against the 20 labelled gold rows in sample_claims.csv:

- evidence_standard_met is definitionally tied to the verdict:
  true unless claim_status == not_enough_information.
- not_enough_information forces severity=unknown, issue_type=unknown,
  supporting_image_ids=none (the images were not sufficient to decide).
- History can ADD risk flags but never flips a clear visual verdict by itself.
  Whenever user_history_risk applies, manual_review_required is added too
  (matches gold behaviour).
- text_instruction_present is set if EITHER the deterministic guard OR the model
  (conversation or in-image text) detected steering; such text never changes the
  verdict.
"""
from __future__ import annotations

from typing import Dict, List

from . import schema as S
from . import injection_guard


def fuse(row: Dict[str, str], vision: dict, hist_ctx: dict) -> Dict[str, str]:
    claim_object = (row.get("claim_object") or "").strip().lower()
    user_claim = row.get("user_claim") or ""
    submitted_ids = vision.get("_submitted_ids", [])

    # --- raw model fields (clamped) ---
    status = S.coerce_claim_status(vision.get("claim_status"))
    issue_type = S.coerce_issue_type(vision.get("issue_type"))
    object_part = S.coerce_object_part(vision.get("object_part"), claim_object)
    severity = S.coerce_severity(vision.get("severity"))
    # Terse local models often omit valid_image; default True when we have images
    # (the prior is that a submitted image IS usable) and override only on signals.
    _vi = vision.get("valid_image", None)
    valid_image = True if _vi is None else str(_vi).lower() in {"true", "1", "yes"}

    # --- risk flags: start from image-derived flags ---
    risk = set()
    for f in vision.get("risk_flags", []) or []:
        risk.add(str(f).strip().lower())
    # also harvest per-image quality/auth flags as backstop
    for im in vision.get("per_image", []) or []:
        for f in (im.get("quality_issues") or []) + (im.get("authenticity_issues") or []):
            risk.add(str(f).strip().lower())

    # --- injection / instruction text (deterministic OR model) ---
    det = injection_guard.detect_instruction_text(user_claim)
    model_instr = bool(vision.get("model_detected_instruction_text"))
    model_instr = model_instr or any(
        im.get("embedded_text_instruction") for im in (vision.get("per_image") or [])
    )
    if det["instruction"] or model_instr:
        risk.add("text_instruction_present")
    if det["pressure"]:
        risk.add("manual_review_required")

    # --- history risk (additive, never overrides visuals) ---
    for f in hist_ctx.get("flags", []):
        risk.add(f)
    if "user_history_risk" in risk:
        risk.add("manual_review_required")

    # --- deterministic reconciliation against verdict ---
    if status == "not_enough_information":
        evidence_met = False
        severity = "unknown"
        issue_type = "unknown"
        support_ids: List[str] = []
        # NEI keeps the claimed/relevant part (object_part) as-is.
        if "damage_not_visible" not in risk and not any(
            q in risk for q in ("blurry_image", "cropped_or_obstructed", "wrong_angle", "low_light_or_glare")
        ):
            risk.add("damage_not_visible")
    else:
        evidence_met = True
        support_ids = [i for i in (vision.get("supporting_image_ids") or []) if i in submitted_ids]
        if not support_ids:
            # supported/contradicted must point at the evidence used.
            support_ids = submitted_ids[:1]

    # valid_image safety: manipulated/non-original or zero usable -> not valid
    if status == "not_enough_information" and not valid_image:
        pass  # leave as model said
    if "non_original_image" in risk and status != "contradicted":
        # a fabricated image is not usable evidence unless it's the basis of a contradiction
        valid_image = valid_image and False

    out = {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": S.coerce_bool(evidence_met),
        "evidence_standard_met_reason": _clip(vision.get("evidence_reason")
                                              or _default_evidence_reason(status)),
        "risk_flags": S.coerce_risk_flags(risk),
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": status,
        "claim_status_justification": _clip(vision.get("justification")
                                            or "Decision based on submitted image evidence."),
        "supporting_image_ids": S.coerce_image_ids(support_ids),
        "valid_image": S.coerce_bool(valid_image),
        "severity": severity,
    }
    return out


def _default_evidence_reason(status: str) -> str:
    if status == "not_enough_information":
        return "The submitted images do not show the claimed part clearly enough to evaluate."
    return "The claimed part is visible clearly enough to evaluate the claim."


def _clip(text, n: int = 280) -> str:
    t = " ".join(str(text or "").split())
    return t[:n]
