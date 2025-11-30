"""Embed chunked text using a local Ollama embedding model."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests
import argparse

ROOT = Path(__file__).resolve().parent.parent
CONFIG = {
    "chemistry": {
        "chunks": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_ollama_meta.jsonl",
    },
    "physics": {
        "chunks": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_chunks.jsonl",
        "embeddings": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama.npy",
        "meta": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_embeddings_ollama_meta.jsonl",
    },
}

DEFAULT_MODEL = "mxbai-embed-large"
EMBED_URL = "http://localhost:11434/api/embed"
BATCH_SIZE = 64


def iter_chunks(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec["id"], rec["text"]


def embed_batch(model: str, texts: List[str]) -> List[List[float]]:
    resp = requests.post(EMBED_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]


def write_metadata(ids: List[str], meta_path: Path) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({"id": cid}) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunks using an Ollama embedding model.")
    parser.add_argument("--subject", choices=CONFIG.keys(), help="Use preset paths for subject")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama embedding model name")
    parser.add_argument("--chunks", type=Path, help="Chunks JSONL (overrides subject)")
    parser.add_argument("--out", type=Path, help="Output embeddings .npy (overrides subject)")
    parser.add_argument("--meta", type=Path, help="Output metadata JSONL (overrides subject)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subject:
        cfg = CONFIG[args.subject]
        chunks_path = cfg["chunks"]
        emb_path = cfg["embeddings"]
        meta_path = cfg["meta"]
    else:
        chunks_path = args.chunks
        emb_path = args.out
        meta_path = args.meta

    if chunks_path is None or emb_path is None or meta_path is None:
        raise SystemExit("Provide --subject or --chunks/--out/--meta.")
    if not chunks_path.exists():
        raise SystemExit(f"Chunks file not found: {chunks_path}")

    model = args.model

    all_ids: List[str] = []
    all_vecs: List[List[float]] = []

    batch_texts: List[str] = []
    batch_ids: List[str] = []

    for cid, text in iter_chunks(chunks_path):
        batch_ids.append(cid)
        batch_texts.append(text)
        if len(batch_texts) >= BATCH_SIZE:
            vecs = embed_batch(model, batch_texts)
            all_ids.extend(batch_ids)
            all_vecs.extend(vecs)
            batch_ids, batch_texts = [], []

    if batch_texts:
        vecs = embed_batch(model, batch_texts)
        all_ids.extend(batch_ids)
        all_vecs.extend(vecs)

    vectors = np.array(all_vecs, dtype=np.float32)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, vectors)
    write_metadata(all_ids, meta_path)

    print(f"Embedded {len(all_ids)} chunks with Ollama model '{model}'")
    print(f"Embeddings: {emb_path}")
    print(f"Metadata:   {meta_path}")


if __name__ == "__main__":
    main()
