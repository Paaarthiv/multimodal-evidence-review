"""End-to-end per-claim pipeline and batch runner.

Stages: parse claim -> select evidence standard -> single vision call (all images
of the case batched into ONE call) -> fuse with history + injection guard ->
clamp to schema. One model call per claim keeps cost/latency/RPM low.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from . import config, fusion, io_utils, providers, schema
from .evidence import EvidenceRules
from .history import UserHistory
from .vision_schema import build_user_prompt


@dataclass
class Context:
    evidence: EvidenceRules
    history: UserHistory
    system_prompt: str
    dataset_dir: Path

    @classmethod
    def load(cls) -> "Context":
        system = (config.PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")
        return cls(
            evidence=EvidenceRules.load(config.EVIDENCE_REQ_CSV),
            history=UserHistory.load(config.USER_HISTORY_CSV),
            system_prompt=system,
            dataset_dir=config.DATASET_DIR,
        )


def _nei_fallback(row: Dict[str, str], reason: str, hist_ctx: dict = None) -> Dict[str, str]:
    """Safe row when no usable image / hard error: not_enough_information.

    hist_ctx is threaded through so a failed/no-image row still carries the
    user's history risk flags (instead of silently dropping them).
    """
    vision = {
        "claim_status": "not_enough_information", "issue_type": "unknown",
        "object_part": "unknown", "severity": "unknown", "valid_image": False,
        "risk_flags": ["damage_not_visible"], "supporting_image_ids": [],
        "evidence_reason": reason, "justification": reason,
        "per_image": [], "_submitted_ids": [],
    }
    hist = hist_ctx or {"flags": [], "summary": "", "elevated": False}
    return fusion.fuse(row, vision, hist)


def process_claim(row: Dict[str, str], ctx: Context,
                  stats: providers.CallStats) -> Tuple[Dict[str, str], providers.ModelResult]:
    claim_object = (row.get("claim_object") or "").strip().lower()
    rel_paths = io_utils.split_image_paths(row.get("image_paths", ""))
    image_ids = [io_utils.image_id_from_path(p) for p in rel_paths]

    hist_ctx = ctx.history.risk_context(row.get("user_id", ""))

    abs_paths = [io_utils.resolve_image_path(p, ctx.dataset_dir) for p in rel_paths]
    existing = [p for p in abs_paths if p.exists()]
    if not existing:
        return _nei_fallback(row, "No usable image file was found for this claim.", hist_ctx), \
            providers.ModelResult(data={}, provider="none", model="none")

    allowed_parts = schema.OBJECT_PARTS.get(claim_object, {"unknown"})
    evidence_standard = ctx.evidence.relevant_text(
        claim_object, "unknown", "unknown", multi_image=len(existing) > 1
    )
    user_prompt = build_user_prompt(
        claim_object, row.get("user_claim", ""), image_ids,
        evidence_standard, list(allowed_parts),
    )

    result = providers.analyze(ctx.system_prompt, user_prompt, existing)
    if result.error or not result.data:
        out = _nei_fallback(row, "Automated review unavailable; routed for manual review.", hist_ctx)
        out["risk_flags"] = schema.coerce_risk_flags(
            out["risk_flags"].split(";") + ["manual_review_required"]
        )
        stats.record(result, len(existing))
        return out, result

    vision = dict(result.data)
    vision["_submitted_ids"] = image_ids
    out = fusion.fuse(row, vision, hist_ctx)
    stats.record(result, len(existing))
    return out, result


def run_file(input_csv: Path, ctx: Context, limit: int = 0,
             concurrency: int = None) -> Tuple[List[Dict[str, str]], providers.CallStats]:
    import time as _time

    rows = io_utils.read_csv(input_csv)
    if limit:
        rows = rows[:limit]
    stats = providers.CallStats()
    outs: Dict[int, Dict[str, str]] = {}
    failed: set = set()
    concurrency = concurrency or config.MAX_CONCURRENCY

    def handle(i, out, res):
        outs[i] = out
        # provider == "none" means an infra failure (quota/overload), not a real NEI.
        if res is not None and res.provider == "none":
            failed.add(i)
        else:
            failed.discard(i)

    if concurrency <= 1:
        for i, row in enumerate(rows):
            out, res = process_claim(row, ctx, stats)
            handle(i, out, res)
            print(f"  [{i+1}/{len(rows)}] {row.get('user_id')} -> {out['claim_status']}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(process_claim, row, ctx, stats): i for i, row in enumerate(rows)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                out, res = fut.result()
                handle(i, out, res)
                done += 1
                print(f"  [{done}/{len(rows)}] row {i+1} {rows[i].get('user_id')} -> {out['claim_status']}")

    # Retry rows that failed for infra reasons (cache makes successes free).
    for attempt in range(config.ROW_RETRY_PASSES):
        if not failed:
            break
        retry = sorted(failed)
        print(f"  [retry pass {attempt+1}] {len(retry)} failed row(s): {retry}")
        _time.sleep(config.ROW_RETRY_DELAY)
        for i in retry:
            out, res = process_claim(rows[i], ctx, stats)
            handle(i, out, res)

    if failed:
        print(f"  WARNING: {len(failed)} row(s) still failed after retries "
              f"(emitted as manual-review NEI): {sorted(failed)}")

    return [outs[i] for i in range(len(rows))], stats
