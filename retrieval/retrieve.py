"""Retrieve top chunks for a query using Ollama embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

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
        # Merged biology course pack defaults
        "chunks": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "biology" / "bio_merged_embeddings_ollama_meta.jsonl",
    },
}

DEFAULT_MODEL = "mxbai-embed-large"
EMBED_URL = "http://localhost:11434/api/embed"


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
    resp = requests.post(EMBED_URL, json={"model": model, "input": [text]})
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"][0], dtype=np.float32)


def cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # Normalize embeddings to unit length for cosine similarity
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    q_norm = query_vec / np.linalg.norm(query_vec)
    return mat_norm @ q_norm


def top_k(scores: np.ndarray, ids: List[str], k: int) -> List[Tuple[str, float]]:
    idxs = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    sorted_idxs = idxs[np.argsort(-scores[idxs])]
    return [(ids[i], float(scores[i])) for i in sorted_idxs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top chunks for a query.")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama embedding model name")
    parser.add_argument("--top-k", type=int, default=15, help="Number of results to return")
    parser.add_argument("--subject", default="chemistry", help="Subject key (chemistry, physics, ...)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks_path, emb_path, meta_path = get_paths(args.subject)

    if not (emb_path.exists() and meta_path.exists() and chunks_path.exists()):
        sys.stderr.write(f"Embeddings or chunk files are missing for subject '{args.subject}'.\n")
        sys.stderr.write(f"Expected: {chunks_path}, {emb_path}, {meta_path}\n")
        sys.stderr.write("Run cleaning steps and step6 for this subject first.\n")
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    meta_ids = load_meta_ids(meta_path)
    emb_matrix = np.load(emb_path)

    q_vec = embed_text(args.model, args.query)
    scores = cosine_sim_matrix(q_vec, emb_matrix)
    results = top_k(scores, meta_ids, args.top_k)

    for rank, (cid, score) in enumerate(results, start=1):
        text = chunks.get(cid, "")[:400].replace("\n", " ")
        print(f"{rank}. {cid} | score={score:.4f}")
        print(f"   {text}...")


if __name__ == "__main__":
    main()
