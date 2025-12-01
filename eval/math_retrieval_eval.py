"""
Math Retrieval Evaluation

Metrics:
- MRR
- MAP
- Recall@10
- nDCG@10

Output files:
    eval/math_retrieval_results.jsonl
    eval/math_retrieval_summary.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------
# Ensure retrieve.py can be imported
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # highschoolqna/
RETR_DIR = ROOT / "retrieval"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(RETR_DIR) not in sys.path:
    sys.path.append(str(RETR_DIR))

from retrieve import retrieve_full_ranked

GOLD_FILE = ROOT / "eval" / "math_gold_with_chunks.jsonl"
OUT_RESULTS = ROOT / "eval" / "math_retrieval_results.jsonl"
OUT_SUMMARY = ROOT / "eval" / "math_retrieval_summary.json"


# --------------------------------------------------
# nDCG computation
# --------------------------------------------------
def _dcg(scores):
    return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))


def _ndcg(ranked_ids, gold_set, k=10):
    gains = [1 if cid in gold_set else 0 for cid in ranked_ids[:k]]
    ideal = sorted(gains, reverse=True)
    ideal_dcg = _dcg(ideal)
    return _dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


# --------------------------------------------------
# Process single retrieval item
# --------------------------------------------------
def process_retrieval(ex):
    """Process a single retrieval evaluation item."""
    query = ex["query"]
    gold_chunks = ex["gold_chunks"]
    gold_set = set(gold_chunks)

    ranking = retrieve_full_ranked(query, subject="math")
    ranked_ids = [r["id"] for r in ranking]

    # MRR
    rr = 0
    for idx, cid in enumerate(ranked_ids):
        if cid in gold_set:
            rr = 1.0 / (idx + 1)
            break

    # MAP
    hits = 0
    precisions = []
    for idx, cid in enumerate(ranked_ids):
        if cid in gold_set:
            hits += 1
            precisions.append(hits / (idx + 1))
    ap = np.mean(precisions) if precisions else 0.0

    # Recall@10
    top10 = ranked_ids[:10]
    recall = len([c for c in top10 if c in gold_set]) / len(gold_set) if gold_set else 0

    # nDCG@10
    ndcg = _ndcg(ranked_ids, gold_set, k=10)

    return {
        "query": query,
        "gold_chunks": gold_chunks,
        "ranking_top50": ranked_ids[:50],
        "mrr": rr,
        "map": ap,
        "recall10": recall,
        "ndcg10": ndcg
    }


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
def main():
    items = [json.loads(l) for l in open(GOLD_FILE, "r", encoding="utf-8")]
    print(f"[LOAD] Loaded {len(items)} items from {GOLD_FILE}")
    print(f"[INFO] Processing with parallel workers (max 4)...\n")

    results = []

    # Use ThreadPoolExecutor for I/O-bound retrieval tasks
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_retrieval, ex) for ex in items]

        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(items), desc="Evaluating Retrieval"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"\n[ERROR] Retrieval task generated an exception: {exc}")

    # Write all results to file
    with open(OUT_RESULTS, "w", encoding="utf-8") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")

    # Compute summary statistics
    mrrs = [r["mrr"] for r in results]
    maps = [r["map"] for r in results]
    recalls = [r["recall10"] for r in results]
    ndcgs = [r["ndcg10"] for r in results]

    summary = {
        "MRR": float(np.mean(mrrs)),
        "MAP": float(np.mean(maps)),
        "Recall@10": float(np.mean(recalls)),
        "nDCG@10": float(np.mean(ndcgs)),
        "count": len(items)
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Retrieval Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
