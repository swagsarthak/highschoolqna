# eval/math_plot.py
"""
Make PPT-style bar plots for math evaluation.

Inputs:
- eval/math_retrieval_summary.json  (from math_retrieval_eval.py)
- eval/math_eval_results.jsonl      (from run_math_eval.py)

Outputs (saved under eval/figs/):
- math_retrieval_metrics.png
- math_generation_accuracy.png
- math_hallucination.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
RETR_SUMMARY = EVAL_DIR / "math_retrieval_summary.json"
GEN_RESULTS = EVAL_DIR / "math_eval_results.jsonl"
FIG_DIR = EVAL_DIR / "figs"

FIG_DIR.mkdir(exist_ok=True)


# ---------- helpers ----------

def load_retrieval_summary() -> Dict:
    if not RETR_SUMMARY.exists():
        raise FileNotFoundError(f"Missing retrieval summary: {RETR_SUMMARY}")
    return json.loads(RETR_SUMMARY.read_text(encoding="utf-8"))


def load_generation_results() -> List[Dict]:
    if not GEN_RESULTS.exists():
        raise FileNotFoundError(f"Missing generation results: {GEN_RESULTS}")
    out = []
    with GEN_RESULTS.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def token_set(s: str):
    return set(w.lower() for w in WORD_RE.findall(s))


def compute_generation_metrics(rows: List[Dict]) -> Dict[str, float]:
    total = len(rows)
    if total == 0:
        return {"accuracy": 0.0, "hallucination_rate": 0.0, "faithfulness": 0.0}

    n_correct = 0
    n_hallu = 0

    for r in rows:
        correct = bool(r.get("correct"))
        if correct:
            n_correct += 1
            continue

        gold = str(r.get("gold", ""))
        pred = str(r.get("pred", ""))
        g_set = token_set(gold)
        p_set = token_set(pred)
        inter = len(g_set & p_set)
        union = max(1, len(g_set | p_set))
        jacc = inter / union

        # 简单定义：错了且和 gold 几乎没重合 ⇒ 算 hallucination
        if jacc < 0.1:
            n_hallu += 1

    acc = n_correct / total
    hallu_rate = n_hallu / total
    faithfulness = 1.0 - hallu_rate
    return {
        "accuracy": acc,
        "hallucination_rate": hallu_rate,
        "faithfulness": faithfulness,
    }


def nice_bar(ax, labels, values, title, ylim=(0, 1.0)):
    x = range(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)


# ---------- main plotting ----------

def main() -> None:
    retr = load_retrieval_summary()
    gen_rows = load_generation_results()
    gen = compute_generation_metrics(gen_rows)

    # 1) Retrieval metrics bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["MRR", "MAP", "Recall@10", "nDCG@10"]
    values = [
        float(retr.get("MRR", 0.0)),
        float(retr.get("MAP", 0.0)),
        float(retr.get("Recall@10", 0.0)),
        float(retr.get("nDCG@10", 0.0)),
    ]
    nice_bar(ax, labels, values, "Retrieval Performance (Math)")
    fig.tight_layout()
    out1 = FIG_DIR / "math_retrieval_metrics.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out1}")

    # 2) Generation accuracy chart
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Accuracy"]
    values = [gen["accuracy"]]
    nice_bar(ax, labels, values, "Generation Accuracy (Math)")
    fig.tight_layout()
    out2 = FIG_DIR / "math_generation_accuracy.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out2}")

    # 3) Hallucination vs Faithfulness chart
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Faithful", "Hallucinated"]
    values = [gen["faithfulness"], gen["hallucination_rate"]]
    nice_bar(ax, labels, values, "Faithfulness vs Hallucination (Math)")
    fig.tight_layout()
    out3 = FIG_DIR / "math_hallucination.png"
    fig.savefig(out3, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out3}")


if __name__ == "__main__":
    main()
