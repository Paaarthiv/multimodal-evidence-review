"""Terminal entry point (project contract: code/main.py).

Runs the multi-modal evidence-review pipeline over a claims CSV and writes
output.csv with the exact required 14-column schema.

Examples:
  python code/main.py                       # run dataset/claims.csv -> output.csv
  python code/main.py --input dataset/sample_claims.csv --out sample_output.csv
  python code/main.py --limit 5             # quick smoke test on first 5 rows
  USE_MOCK=1 python code/main.py --limit 3  # offline, no API keys needed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python code/main.py` from repo root or from inside code/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, io_utils  # noqa: E402
from src.pipeline import Context, run_file  # noqa: E402
from src.cost import summarize_cost  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-Modal Evidence Review")
    ap.add_argument("--input", default=str(config.TEST_CLAIMS_CSV),
                    help="claims CSV to process (default: dataset/claims.csv)")
    ap.add_argument("--out", default=str(config.REPO_ROOT / "output.csv"),
                    help="output CSV path (default: <repo>/output.csv)")
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (0 = all)")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    args = ap.parse_args()

    input_csv = Path(args.input)
    out_csv = Path(args.out)
    print(f"Input : {input_csv}")
    print(f"Output: {out_csv}")
    print(f"Provider order: {'mock' if config.USE_MOCK else config.PROVIDER_ORDER}")

    ctx = Context.load()
    t0 = time.time()
    rows, stats = run_file(input_csv, ctx, limit=args.limit, concurrency=args.concurrency)
    elapsed = time.time() - t0

    io_utils.write_output_csv(out_csv, rows)

    cost = summarize_cost(stats)
    runlog = {
        "input": str(input_csv), "output": str(out_csv), "rows": len(rows),
        "elapsed_sec": round(elapsed, 1), "model_calls": stats.calls,
        "cache_hits": stats.cache_hits, "images": stats.images,
        "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens,
        "by_provider": stats.by_provider, "cost_usd": cost,
    }
    (config.CODE_DIR / "logs").mkdir(exist_ok=True)
    (config.CODE_DIR / "logs" / "last_run.json").write_text(
        json.dumps(runlog, indent=2), encoding="utf-8")

    print("\n=== RUN SUMMARY ===")
    print(json.dumps(runlog, indent=2))
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
