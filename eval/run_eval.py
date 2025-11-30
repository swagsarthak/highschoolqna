"""Evaluation harness for Course RAG.

Supports retrieval metrics (precision/recall/MAP/MRR/nDCG) and optional
generation metrics (BLEU/ROUGE-L) against a gold QA set.

Gold QA JSONL format (one per line)
{
  "id": "chem-q1",
  "subject": "chemistry",
  "question": "What is organic chemistry?",
  "answer_ref": "Organic chemistry is the study of carbon compounds ...",
  "ref_chunks": ["chunk-00002", "chunk-00001"],
  "difficulty": "easy",
  "tags": ["intro"]
}

Usage examples
- Retrieval metrics only:
    python eval/run_eval.py --gold eval/sample_gold.jsonl --top-k 5 --skip-generation
- Retrieval + generation metrics (needs Ollama models running):
    python eval/run_eval.py --gold eval/sample_gold.jsonl --top-k 5
- Write results to JSON:
    python eval/run_eval.py --gold eval/sample_gold.jsonl --out eval/results.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = {
    "chemistry": {
        "chunks": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama_meta.jsonl",
        "title": "Chemistry",
    },
    "physics": {
        "chunks": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama_meta.jsonl",
        "title": "Physics",
    },
    "biology": {
        "chunks": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama_meta.jsonl",
        "title": "Biology",
    },
}

DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3"
EMBED_URL = "http://localhost:11434/api/embed"
CHAT_URL = "http://localhost:11434/api/chat"


# ----------------------------
# Data loading and retrieval
# ----------------------------
def load_chunks(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)["id"]: json.loads(line)["text"] for line in f}


def load_meta_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    return ids


DATA_CACHE: Dict[str, Tuple[Dict[str, str], List[str], np.ndarray]] = {}
FAISS_CACHE: Dict[str, "faiss.Index"] = {}


def get_paths(subject: str) -> Tuple[Path, Path, Path]:
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject '{subject}'. Available: {list(SUBJECTS.keys())}")
    cfg = SUBJECTS[subject]
    return cfg["chunks"], cfg["embeddings"], cfg["meta"]


def get_data(subject: str) -> Tuple[Dict[str, str], List[str], np.ndarray]:
    if subject in DATA_CACHE:
        return DATA_CACHE[subject]
    chunks_path, emb_path, meta_path = get_paths(subject)
    if not (chunks_path.exists() and emb_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"Missing data for '{subject}'. Expected {chunks_path}, {emb_path}, {meta_path}"
        )
    chunks = load_chunks(chunks_path)
    meta_ids = load_meta_ids(meta_path)
    emb_matrix = np.load(emb_path)
    DATA_CACHE[subject] = (chunks, meta_ids, emb_matrix)
    return chunks, meta_ids, emb_matrix


def get_faiss_index(subject: str, emb_matrix: np.ndarray) -> "faiss.Index":
    if subject in FAISS_CACHE:
        return FAISS_CACHE[subject]
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
    base = emb_matrix / norms
    dim = base.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(base.astype(np.float32))
    FAISS_CACHE[subject] = index
    return index


def embed_text(model: str, text: str) -> np.ndarray:
    resp = requests.post(EMBED_URL, json={"model": model, "input": [text]})
    if not resp.ok:
        raise RuntimeError(f"Embedding failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return np.array(data["embeddings"][0], dtype=np.float32)


def retrieve(
    subject: str, query_vec: np.ndarray, emb_matrix: np.ndarray, ids: List[str], k: int, use_faiss: bool
) -> List[Tuple[str, float]]:
    if use_faiss and FAISS_AVAILABLE:
        index = get_faiss_index(subject, emb_matrix)
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        scores, idxs = index.search(q[None, :].astype(np.float32), k)
        return [(ids[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]
    mat_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    q_norm = query_vec / np.linalg.norm(query_vec)
    scores = mat_norm @ q_norm
    idxs = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    sorted_idxs = idxs[np.argsort(-scores[idxs])]
    return [(ids[i], float(scores[i])) for i in sorted_idxs]


# ----------------------------
# Retrieval metrics
# ----------------------------
def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    if not retrieved:
        return 0.0
    hits = sum(1 for rid in retrieved[:k] if rid in rel)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    hits = sum(1 for rid in retrieved[:k] if rid in rel)
    return hits / len(rel)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for idx, rid in enumerate(retrieved):
        if rid in rel:
            return 1.0 / (idx + 1)
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    precisions: List[float] = []
    for idx, rid in enumerate(retrieved, start=1):
        if rid in rel:
            hits += 1
            precisions.append(hits / idx)
    return sum(precisions) / len(rel)


def dcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    dcg = 0.0
    for idx, rid in enumerate(retrieved[:k]):
        if rid in rel:
            dcg += 1.0 / math.log2(idx + 2)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(retrieved, relevant, k) / idcg


# ----------------------------
# Generation metrics
# ----------------------------
def _ngrams(tokens: List[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions: List[float] = []
    smooth = 1.0
    for n in range(1, max_n + 1):
        pred_counts = defaultdict(int)
        ref_counts = defaultdict(int)
        for ng in _ngrams(pred_tokens, n):
            pred_counts[ng] += 1
        for ng in _ngrams(ref_tokens, n):
            ref_counts[ng] += 1
        overlap = sum(min(pred_counts[ng], ref_counts.get(ng, 0)) for ng in pred_counts)
        total = sum(pred_counts.values())
        precisions.append((overlap + smooth) / (total + smooth))

    # Brevity penalty
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return float(score)


def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    beta = 1.0
    return (1 + beta**2) * prec * rec / (rec + beta**2 * prec)


def context_overlap(answer: str, contexts: Iterable[str]) -> float:
    ctx_tokens = set(" ".join(contexts).split())
    ans_tokens = answer.split()
    if not ans_tokens or not ctx_tokens:
        return 0.0
    hits = sum(1 for t in ans_tokens if t in ctx_tokens)
    return hits / len(ans_tokens)


def build_prompt(query: str, contexts: List[Tuple[str, str]]) -> str:
    ctx_block = "\n\n".join(f"[{cid}]\n{text[:1200]}" for cid, text in contexts)
    return (
        "Use only the provided context to answer. "
        "If the answer is not in the context, say you don't know. "
        "Do not mention chunk IDs unless asked.\n\n"
        f"Context:\n{ctx_block}\n\nQuestion: {query}\n\nAnswer:"
    )


def call_llm(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise tutor. Use the given context only.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(CHAT_URL, json=payload)
    if not resp.ok:
        raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return data["message"]["content"]


# ----------------------------
# Evaluation loop
# ----------------------------
@dataclass
class SampleResult:
    sample_id: str
    subject: str
    retrieval: Dict[str, float]
    retrieved: List[Dict[str, float]]
    ref_chunks: List[str]
    answer: Optional[str] = None
    answer_ref: Optional[str] = None
    generation: Optional[Dict[str, float]] = None
    meta: Dict[str, str] = field(default_factory=dict)


def eval_sample(
    sample: dict,
    top_k: int,
    embed_model: str,
    llm_model: str,
    skip_generation: bool,
    use_faiss: bool,
) -> SampleResult:
    subject = sample["subject"]
    question = sample["question"]
    ref_chunks = sample.get("ref_chunks", [])
    answer_ref = sample.get("answer_ref")

    chunks, meta_ids, emb_matrix = get_data(subject)
    q_vec = embed_text(embed_model, question)
    results = retrieve(subject, q_vec, emb_matrix, meta_ids, top_k, use_faiss)
    retrieved_ids = [cid for cid, _ in results]

    metrics: Dict[str, float] = {}
    ks = sorted({1, 3, top_k})
    for k in ks:
        metrics[f"p@{k}"] = precision_at_k(retrieved_ids, ref_chunks, k)
        metrics[f"r@{k}"] = recall_at_k(retrieved_ids, ref_chunks, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, ref_chunks, k)
    metrics["map"] = average_precision(retrieved_ids, ref_chunks)
    metrics["mrr"] = reciprocal_rank(retrieved_ids, ref_chunks)

    gen_metrics = None
    answer = None
    if answer_ref and not skip_generation:
        ctx = [(cid, chunks.get(cid, "")) for cid, _ in results]
        prompt = build_prompt(question, ctx)
        answer = call_llm(llm_model, prompt).strip()
        gen_metrics = {
            "bleu": bleu_score(answer, answer_ref),
            "rouge_l": rouge_l(answer, answer_ref),
            "context_overlap": context_overlap(answer, (c[1] for c in ctx)),
        }

    return SampleResult(
        sample_id=sample["id"],
        subject=subject,
        retrieval=metrics,
        retrieved=[{"id": cid, "score": score} for cid, score in results],
        ref_chunks=ref_chunks,
        answer=answer,
        answer_ref=answer_ref,
        generation=gen_metrics,
        meta={k: str(sample.get(k, "")) for k in ("difficulty", "tags") if k in sample},
    )


def load_gold(path: Path, max_samples: Optional[int] = None) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "id" not in rec or "question" not in rec or "subject" not in rec:
                raise ValueError("Each gold record must have id, subject, question.")
            items.append(rec)
            if max_samples and len(items) >= max_samples:
                break
    return items


def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return float(sum(vals) / len(vals)) if vals else 0.0


def aggregate(results: List[SampleResult], top_k: int) -> Dict[str, dict]:
    bucket: Dict[str, List[SampleResult]] = defaultdict(list)
    for res in results:
        bucket[res.subject].append(res)
    bucket["overall"] = results

    summary: Dict[str, dict] = {}
    ks = sorted({1, 3, top_k})
    for key, group in bucket.items():
        row = {"count": len(group)}
        for k in ks:
            row[f"p@{k}"] = mean(r.retrieval[f"p@{k}"] for r in group)
            row[f"r@{k}"] = mean(r.retrieval[f"r@{k}"] for r in group)
            row[f"ndcg@{k}"] = mean(r.retrieval[f"ndcg@{k}"] for r in group)
        row["map"] = mean(r.retrieval["map"] for r in group)
        row["mrr"] = mean(r.retrieval["mrr"] for r in group)
        if any(r.generation for r in group):
            gen = [r for r in group if r.generation]
            row["bleu"] = mean(r.generation["bleu"] for r in gen if r.generation)
            row["rouge_l"] = mean(r.generation["rouge_l"] for r in gen if r.generation)
            row["context_overlap"] = mean(
                r.generation["context_overlap"] for r in gen if r.generation
            )
        summary[key] = row
    return summary


def print_summary(summary: Dict[str, dict], ks: List[int]) -> None:
    def fmt(val: float) -> str:
        return f"{val:.3f}"

    headers = ["bucket", "count"] + [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks] + [
        f"ndcg@{k}" for k in ks
    ] + ["map", "mrr", "bleu", "rouge_l"]
    print("\n=== Summary ===")
    print(" | ".join(headers))
    for bucket, row in summary.items():
        parts = [bucket, str(row.get("count", 0))]
        for k in ks:
            parts.append(fmt(row.get(f"p@{k}", 0.0)))
        for k in ks:
            parts.append(fmt(row.get(f"r@{k}", 0.0)))
        for k in ks:
            parts.append(fmt(row.get(f"ndcg@{k}", 0.0)))
        parts.append(fmt(row.get("map", 0.0)))
        parts.append(fmt(row.get("mrr", 0.0)))
        parts.append(fmt(row.get("bleu", 0.0)))
        parts.append(fmt(row.get("rouge_l", 0.0)))
        print(" | ".join(parts))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval/generation.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold QA JSONL path")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k contexts to retrieve")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama LLM for generation metrics")
    parser.add_argument("--skip-generation", action="store_true", help="Skip answer generation/metrics")
    parser.add_argument("--max-samples", type=int, help="Limit number of gold rows for quick runs")
    parser.add_argument("--out", type=Path, help="Write detailed JSON results to this path")
    parser.add_argument("--use-faiss", action="store_true", help="Use Faiss if installed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold, args.max_samples)
    results: List[SampleResult] = []
    for sample in gold:
        res = eval_sample(
            sample=sample,
            top_k=args.top_k,
            embed_model=args.embed_model,
            llm_model=args.llm_model,
            skip_generation=args.skip_generation,
            use_faiss=args.use_faiss,
        )
        results.append(res)
        print(f"✓ {sample['id']} ({sample['subject']})")

    ks = sorted({1, 3, args.top_k})
    summary = aggregate(results, args.top_k)
    print_summary(summary, ks)

    if args.out:
        payload = {
            "summary": summary,
            "results": [res.__dict__ for res in results],
            "config": {
                "top_k": args.top_k,
                "embed_model": args.embed_model,
                "llm_model": args.llm_model,
                "skip_generation": args.skip_generation,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
