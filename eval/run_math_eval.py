"""
Stable run_math_eval — using subprocess (qa.py unchanged)
Fixes path, prevents Ollama deadlocks, works on WSL2 + GPU.
"""

from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔥 Correct project root: go up *two* levels: eval/ → project root/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_PATH = PROJECT_ROOT / "eval" / "math_gold.jsonl"
OUT_PATH = PROJECT_ROOT / "eval" / "math_eval_results.jsonl"
QA_SCRIPT = PROJECT_ROOT / "retrieval" / "qa.py"


def call_qa(query, model):
    """Call qa.py via subprocess (safe, avoids GPU deadlocks)."""
    cmd = [
        sys.executable,
        str(QA_SCRIPT),
        query,
        "--subject", "math",
        "--llm-model", model,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=50
        )
        if proc.stderr.strip():
            print("STDERR:", proc.stderr)
        return proc.stdout
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


# -------- ANSWER EXTRACTION --------

ANSWER_RE = re.compile(
    r"Answer:\s*(?P<body>.*?)(?:\n\nContexts used:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_answer_text(raw: str) -> str:
    m = ANSWER_RE.search(raw)
    if m:
        return m.group("body").strip()
    return raw.strip()

def extract_key(s: str):
    nums = NUM_RE.findall(s)
    if nums:
        return nums[-1]
    return re.sub(r"\s+", " ", s.lower()).strip()

def equivalent(gold, pred):
    return extract_key(gold) == extract_key(pred)


# -------- PROCESS ONE QUESTION --------

def process_question(ex, idx, total, model):
    q = ex["question"]
    gold = ex["gold"]

    print(f"[{idx}/{total}] {ex['id']}")
    print("Q:", q)
    print("Gold:", gold)

    raw = call_qa(q, model)
    print("RAW:\n", raw)

    pred = extract_answer_text(raw)
    ok = equivalent(gold, pred)

    print(f"[{idx}/{total}] {ex['id']} - {'✓ CORRECT' if ok else '✗ WRONG'}")

    return {
        "id": ex["id"],
        "question": q,
        "gold": gold,
        "pred": pred,
        "correct": ok
    }


# -------- MAIN --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    args = ap.parse_args()

    gold_items = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            gold_items.append({
                "id": x.get("id"),
                "question": x["question"],
                "gold": str(x["answer"])
            })

    print(f"[LOAD] {len(gold_items)} questions")

    results = []
    n_correct = 0

    # Must be 1 worker (Ollama serial by design)
    with ThreadPoolExecutor(max_workers=1) as exe:
        futures = [
            exe.submit(process_question, ex, i, len(gold_items), args.model)
            for i, ex in enumerate(gold_items, 1)
        ]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r["correct"]:
                n_correct += 1

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = n_correct / len(gold_items)
    print(f"\n=== Math QA Accuracy ===")
    print(f"{n_correct}/{len(gold_items)} → {acc:.4f}")


if __name__ == "__main__":
    main()
