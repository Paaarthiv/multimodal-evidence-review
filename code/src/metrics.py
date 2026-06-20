"""Scoring of predictions against gold labels (sample_claims.csv)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from . import schema as S

SCALAR_COLS = [
    "evidence_standard_met", "issue_type", "object_part", "claim_status",
    "valid_image", "severity",
]
SET_COLS = ["risk_flags", "supporting_image_ids"]


def _as_set(v: str) -> set:
    return {x.strip().lower() for x in (v or "").split(";") if x.strip() and x.strip().lower() != "none"}


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def score(pred: List[Dict[str, str]], gold: List[Dict[str, str]]) -> dict:
    n = min(len(pred), len(gold))
    out = {"n": n, "scalar_accuracy": {}, "multilabel": {}, "confusion": {}, "examples_wrong": {}}

    # scalar exact-match
    for col in SCALAR_COLS:
        correct = sum(1 for i in range(n)
                      if str(pred[i].get(col, "")).strip().lower() == str(gold[i].get(col, "")).strip().lower())
        out["scalar_accuracy"][col] = round(correct / n, 3) if n else 0.0

    # claim_status confusion matrix
    conf = defaultdict(lambda: defaultdict(int))
    wrong_status = []
    for i in range(n):
        g = str(gold[i].get("claim_status", "")).strip().lower()
        p = str(pred[i].get("claim_status", "")).strip().lower()
        conf[g][p] += 1
        if g != p:
            wrong_status.append({"row": i, "user_id": gold[i].get("user_id"), "gold": g, "pred": p})
    out["confusion"]["claim_status"] = {g: dict(v) for g, v in conf.items()}
    out["examples_wrong"]["claim_status"] = wrong_status

    # multilabel (risk_flags) + set match (supporting_image_ids)
    for col in SET_COLS:
        tp = fp = fn = 0
        exact = 0
        for i in range(n):
            gs, ps = _as_set(gold[i].get(col, "")), _as_set(pred[i].get(col, ""))
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
            if gs == ps:
                exact += 1
        p, r, f = _f1(tp, fp, fn)
        out["multilabel"][col] = {
            "precision": p, "recall": r, "f1": f,
            "exact_set_match": round(exact / n, 3) if n else 0.0,
        }

    # severity ordinal MAE (unknown treated as its own; counted only when both known)
    idx = {s: i for i, s in enumerate(S.SEVERITY_ORDER)}
    diffs, both_known = [], 0
    for i in range(n):
        g = str(gold[i].get("severity", "")).strip().lower()
        p = str(pred[i].get("severity", "")).strip().lower()
        if g in idx and p in idx:
            diffs.append(abs(idx[g] - idx[p]))
            both_known += 1
    out["severity_ordinal_mae"] = round(sum(diffs) / both_known, 3) if both_known else None

    # overall full-row exact match across the 6 scalar + 2 set cols
    full = 0
    for i in range(n):
        ok = all(str(pred[i].get(c, "")).strip().lower() == str(gold[i].get(c, "")).strip().lower()
                 for c in SCALAR_COLS)
        ok = ok and all(_as_set(pred[i].get(c, "")) == _as_set(gold[i].get(c, "")) for c in SET_COLS)
        if ok:
            full += 1
    out["full_row_exact_match"] = round(full / n, 3) if n else 0.0
    return out
