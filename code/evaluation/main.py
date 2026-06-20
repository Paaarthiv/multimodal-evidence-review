"""Evaluation entry point (project contract: code/evaluation/main.py).

Runs the pipeline on dataset/sample_claims.csv (which carries gold labels),
scores predictions per column, and COMPARES multiple model configurations
(OpenAI vs Gemini, or mock when offline). Writes:
  - code/evaluation/sample_predictions_<provider>.csv
  - code/evaluation/metrics.json
  - code/evaluation/evaluation_report.md  (metrics + operational analysis)

Usage:
  python code/evaluation/main.py                 # compare all configured providers
  python code/evaluation/main.py --providers openai
  USE_MOCK=1 python code/evaluation/main.py      # offline smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

from src import config, io_utils, metrics  # noqa: E402
from src.cost import summarize_cost  # noqa: E402
from src.pipeline import Context, run_file  # noqa: E402
from report import write_report  # noqa: E402


def available_providers() -> list:
    """Default model configs to compare (>=2 satisfies the eval requirement)."""
    if config.USE_MOCK:
        return ["mock"]
    out = []
    if config.GROQ_API_KEY:
        out += ["groq:qwen/qwen3.6-27b"]
    if config.GEMINI_API_KEY and not out:
        out += ["gemini:gemini-2.5-flash", "gemini:gemini-2.5-flash-lite"]
    if config.OPENAI_API_KEY and not out:
        out += ["openai:gpt-4o"]
    return out or ["ollama:qwen2.5vl:7b"]


def run_for_provider(provider: str, ctx: Context, limit: int):
    # Force a single model config (no failover) so the comparison is clean.
    saved = config.PROVIDER_ORDER
    saved_mock = config.USE_MOCK
    config.PROVIDER_ORDER = [provider] if provider != "mock" else saved
    if provider == "mock":
        config.USE_MOCK = True
    try:
        t0 = time.time()
        preds, stats = run_file(config.SAMPLE_CLAIMS_CSV, ctx, limit=limit, concurrency=config.MAX_CONCURRENCY)
        elapsed = time.time() - t0
    finally:
        config.PROVIDER_ORDER = saved
        config.USE_MOCK = saved_mock

    gold = io_utils.read_csv(config.SAMPLE_CLAIMS_CSV)
    if limit:
        gold = gold[:limit]
    sc = metrics.score(preds, gold)

    safe = provider.replace(":", "_").replace("/", "_")
    out_csv = config.EVAL_DIR / f"sample_predictions_{safe}.csv"
    io_utils.write_output_csv(out_csv, preds)

    return {
        "provider": provider,
        "metrics": sc,
        "operational": {
            "model_calls": stats.calls, "cache_hits": stats.cache_hits,
            "images": stats.images, "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens, "elapsed_sec": round(elapsed, 1),
            "cost": summarize_cost(stats),
        },
        "predictions_csv": str(out_csv),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate evidence-review pipeline")
    ap.add_argument("--providers", default="", help="comma list, e.g. openai,gemini")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    provs = [p.strip() for p in args.providers.split(",") if p.strip()] or available_providers()
    print(f"Evaluating providers: {provs}")
    ctx = Context.load()

    results = []
    for prov in provs:
        print(f"\n--- {prov} ---")
        results.append(run_for_provider(prov, ctx, args.limit))

    (config.EVAL_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, config.EVAL_DIR / "evaluation_report.md")

    print("\n=== EVAL SUMMARY ===")
    for r in results:
        m = r["metrics"]
        print(f"{r['provider']:8s}  status_acc={m['scalar_accuracy']['claim_status']}  "
              f"full_row={m['full_row_exact_match']}  "
              f"riskF1={m['multilabel']['risk_flags']['f1']}  "
              f"calls={r['operational']['model_calls']}  cost=${r['operational']['cost']['total_cost']}")
    print(f"\nReport: {config.EVAL_DIR / 'evaluation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
