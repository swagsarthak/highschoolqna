"""Simple RAG QA: retrieve top chunks and answer with an Ollama LLM."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from requests import HTTPError

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = {
    "chemistry": {
        "chunks": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama_meta.jsonl",
    },
    "physics": {
        # Update these paths after generating physics chunks/embeddings
        "chunks": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama_meta.jsonl",
    },
    "biology": {
        "chunks": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama_meta.jsonl",
    },
}

DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3"
EMBED_URL = "http://localhost:11434/api/embed"
CHAT_URL = "http://localhost:11434/api/chat"


def get_paths(subject: str) -> Tuple[Path, Path, Path]:
    if subject not in SUBJECTS:
        sys.stderr.write(f"Unknown subject '{subject}'. Available: {list(SUBJECTS.keys())}\n")
        sys.exit(1)
    cfg = SUBJECTS[subject]
    return cfg["chunks"], cfg["embeddings"], cfg["meta"]


def load_chunks(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)["id"]: json.loads(line)["text"] for line in f}


def load_meta_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    return ids


def embed_text(model: str, text: str) -> np.ndarray:
    try:
        resp = requests.post(EMBED_URL, json={"model": model, "input": [text]})
        resp.raise_for_status()
    except HTTPError as exc:
        sys.stderr.write(f"Embedding request failed ({resp.status_code}): {resp.text}\n")
        raise SystemExit(1) from exc
    data = resp.json()
    return np.array(data["embeddings"][0], dtype=np.float32)


def cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    q_norm = query_vec / np.linalg.norm(query_vec)
    return mat_norm @ q_norm


def top_k(scores: np.ndarray, ids: List[str], k: int) -> List[Tuple[str, float]]:
    idxs = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    sorted_idxs = idxs[np.argsort(-scores[idxs])]
    return [(ids[i], float(scores[i])) for i in sorted_idxs]


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "by",
    "at",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
}


def rerank_with_overlap(
    query: str, results: List[Tuple[str, float]], chunks: Dict[str, str]
) -> List[Tuple[str, float]]:
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in STOPWORDS]
    if not tokens:
        return results
    scored: List[Tuple[str, float, float]] = []
    for cid, base in results:
        text = chunks.get(cid, "").lower()
        overlap = sum(1 for tok in tokens if tok in text)
        combined = base + 0.05 * overlap
        scored.append((cid, base, combined))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(cid, base) for cid, base, _ in scored]


def build_prompt(query: str, contexts: List[Tuple[str, str]]) -> str:
    context_block = "\n\n".join(
        f"Snippet {i+1}:\n{text[:1200]}" for i, (_cid, text) in enumerate(contexts)
    )
    return (
        "Answer using the provided context. Synthesize and explain; infer through minor typos in the question. "
        "Only say you don't know if the context has no relevant information. "
        "Do not mention chunk IDs, page IDs, or section numbers unless the user asks for them. "
        "Answer in natural prose without bracketed identifiers.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def call_llm(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise tutor. Use only the provided context. If relevant details exist, answer from them. Only say you don't know when nothing relevant is present. Do not include chunk IDs, page IDs, or section numbers unless explicitly asked by the user.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(CHAT_URL, json=payload)
        resp.raise_for_status()
    except HTTPError as exc:
        sys.stderr.write(f"LLM request failed ({resp.status_code}): {resp.text}\n")
        sys.stderr.write("Make sure the chat model is pulled and running (e.g., `ollama pull llama3`).\n")
        raise SystemExit(1) from exc
    data = resp.json()
    return data["message"]["content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve + answer via Ollama LLM.")
    parser.add_argument("query", help="User question")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama LLM for answering")
    parser.add_argument("--top-k", type=int, default=5, help="Number of contexts to include")
    parser.add_argument("--subject", default="chemistry", help="Subject key (chemistry, physics, ...)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks_path, emb_path, meta_path = get_paths(args.subject)

    if not (emb_path.exists() and meta_path.exists() and chunks_path.exists()):
        sys.stderr.write(f"Embeddings or chunk files are missing for subject '{args.subject}'.\n")
        sys.stderr.write(f"Expected: {chunks_path}, {emb_path}, {meta_path}\n")
        sys.stderr.write("Run cleaning steps and embeddings for this subject first.\n")
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    meta_ids = load_meta_ids(meta_path)
    emb_matrix = np.load(emb_path)

    q_vec = embed_text(args.embed_model, args.query)
    scores = cosine_sim_matrix(q_vec, emb_matrix)
    results = rerank_with_overlap(args.query, top_k(scores, meta_ids, args.top_k), chunks)

    contexts: List[Tuple[str, str]] = []
    for cid, _score in results:
        contexts.append((cid, chunks.get(cid, "")))

    prompt = build_prompt(args.query, contexts)
    answer = call_llm(args.llm_model, prompt)

    print("Answer:\n", answer)
    print("\nContexts used:")
    for cid, score in results:
        print(f"- {cid} (score {score:.4f})")


if __name__ == "__main__":
    main()
