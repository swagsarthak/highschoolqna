# eval/math_eval_full.py
"""
One-click math evaluation pipeline.

Steps:
1) Run generation eval (QA accuracy) via run_math_eval.py
2) Run retrieval eval (MAP / MRR / Recall@10 / nDCG@10) via math_retrieval_eval.py
3) Make plots via math_plot.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="qwen2.5:1.5b",
        help="LLM model name for QA (passed to run_math_eval.py)",
    )
    return ap.parse_args()


def run(cmd, cwd: Path | None = None):
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=cwd or ROOT)
    if proc.returncode != 0:
        print(f"[ERROR] Command failed with code {proc.returncode}: {' '.join(map(str, cmd))}")
        sys.exit(proc.returncode)


def main() -> None:
    args = parse_args()

    # 1) generation eval
    run([sys.executable, "eval/run_math_eval.py", "--model", args.model])

    # 2) retrieval eval (uses math_gold_with_chunks.jsonl + embeddings)
    run([sys.executable, "eval/math_retrieval_eval.py"])

    # 3) plots
    run([sys.executable, "eval/math_plot.py"])

    print("\n=== Math evaluation pipeline complete ===")
    print("Check:")
    print("  - eval/math_eval_results.jsonl")
    print("  - eval/math_retrieval_summary.json")
    print("  - eval/figs/*.png  (for PPT plots)")


if __name__ == "__main__":
    main()
