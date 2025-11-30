"""Plot evaluation metrics from eval/run_eval.py JSON output.

Usage:
  python eval/plot_eval.py --results eval/results.json --out-dir eval/plots
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_summary(results_path: Path) -> Dict[str, dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError("Results JSON missing 'summary'. Run eval/run_eval.py with --out first.")
    return data["summary"]


def find_ks(summary: Dict[str, dict], prefix: str) -> List[int]:
    ks = set()
    for bucket in summary.values():
        for key in bucket:
            m = re.match(rf"{prefix}@(\d+)", key)
            if m:
                ks.add(int(m.group(1)))
    return sorted(ks)


def plot_grouped(summary: Dict[str, dict], prefix: str, title: str, out_path: Path) -> None:
    ks = find_ks(summary, prefix)
    if not ks:
        return
    buckets = list(summary.keys())
    idx = np.arange(len(buckets))
    width = 0.8 / len(ks)

    plt.figure(figsize=(10, 5))
    for j, k in enumerate(ks):
        vals = [summary[b].get(f"{prefix}@{k}", 0.0) for b in buckets]
        plt.bar(idx + (j - (len(ks) - 1) / 2) * width, vals, width=width, label=f"{prefix}@{k}")

    plt.xticks(idx, buckets, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_single(summary: Dict[str, dict], metric: str, title: str, out_path: Path, ylim=(0, 1.05)) -> None:
    buckets = list(summary.keys())
    vals = [summary[b].get(metric, 0.0) for b in buckets]
    idx = np.arange(len(buckets))
    plt.figure(figsize=(8, 4))
    plt.bar(idx, vals, color="#3c7dd9")
    plt.xticks(idx, buckets, rotation=20, ha="right")
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from eval/run_eval.py results JSON.")
    parser.add_argument("--results", type=Path, required=True, help="Path to JSON produced by eval/run_eval.py --out")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/plots"), help="Directory to save plots")
    args = parser.parse_args()

    summary = load_summary(args.results)
    out_dir = args.out_dir

    plot_grouped(summary, "p", "Precision@k by bucket", out_dir / "precision.png")
    plot_grouped(summary, "r", "Recall@k by bucket", out_dir / "recall.png")
    plot_grouped(summary, "ndcg", "nDCG@k by bucket", out_dir / "ndcg.png")
    plot_single(summary, "map", "MAP by bucket", out_dir / "map.png")
    plot_single(summary, "mrr", "MRR by bucket", out_dir / "mrr.png")

    # Generation metrics (if present)
    if any("bleu" in bucket for bucket in summary.values()):
        plot_single(summary, "bleu", "BLEU by bucket", out_dir / "bleu.png")
    if any("rouge_l" in bucket for bucket in summary.values()):
        plot_single(summary, "rouge_l", "ROUGE-L by bucket", out_dir / "rouge_l.png")
    if any("context_overlap" in bucket for bucket in summary.values()):
        plot_single(summary, "context_overlap", "Context overlap by bucket", out_dir / "context_overlap.png")

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
