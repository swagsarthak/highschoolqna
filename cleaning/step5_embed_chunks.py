"""Embed chunked text with a sentence transformer and write vectors + metadata.

Default model: sentence-transformers/all-MiniLM-L6-v2 (fast, strong baseline).

Usage examples:
  # chemistry default paths
  python cleaning/step5_embed_chunks.py
  # override model
  python cleaning/step5_embed_chunks.py sentence-transformers/all-mpnet-base-v2
  # custom paths (e.g., biology)
  python cleaning/step5_embed_chunks.py --chunks cleaning/chunks/biology/lebo101_chunks.jsonl --out cleaning/chunks/biology/lebo101_embeddings.npy --meta cleaning/chunks/biology/lebo101_embeddings_meta.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "chemistry": {
        "chunks": ROOT
        / "cleaning"
        / "chunks"
        / "chemistry"
        / "OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl",
        "embeddings": ROOT
        / "cleaning"
        / "chunks"
        / "chemistry"
        / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings.npy",
        "meta": ROOT
        / "cleaning"
        / "chunks"
        / "chemistry"
        / "OrganicChemistry-SAMPLE_9ADraVJ_embeddings_meta.jsonl",
    },
    "physics": {
        "chunks": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_chunks.jsonl",
        "embeddings": ROOT
        / "cleaning"
        / "chunks"
        / "physics"
        / "UniversityPhysics15e_embeddings.npy",
        "meta": ROOT
        / "cleaning"
        / "chunks"
        / "physics"
        / "UniversityPhysics15e_embeddings_meta.jsonl",
    },
}

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            "sentence-transformers is required. Install with: pip install sentence-transformers\n"
        )
        raise SystemExit(1) from exc

    return SentenceTransformer(model_name)


def iter_chunks(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec["id"], rec["text"]


def write_metadata(ids: List[str], meta_path: Path) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        for chunk_id in ids:
            f.write(json.dumps({"id": chunk_id}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunked text with a sentence transformer and write vectors + metadata."
    )
    parser.add_argument("--subject", choices=DEFAULTS.keys(), help="Use preset chunk/meta paths for a subject")
    parser.add_argument("--chunks", type=Path, help="Chunks JSONL path (overrides subject)")
    parser.add_argument("--out", type=Path, help="Output embeddings .npy (overrides subject)")
    parser.add_argument("--meta", type=Path, help="Output metadata JSONL (overrides subject)")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    args = parser.parse_args()

    if args.subject:
        cfg = DEFAULTS[args.subject]
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

    model_name = args.model
    model = load_model(model_name)

    chunk_ids: List[str] = []
    texts: List[str] = []
    for cid, text in iter_chunks(chunks_path):
        chunk_ids.append(cid)
        texts.append(text)

    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=True)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, vectors)
    write_metadata(chunk_ids, meta_path)

    print(f"Embedded {len(chunk_ids)} chunks with {model_name}")
    print(f"Embeddings: {emb_path}")
    print(f"Metadata:   {meta_path}")


if __name__ == "__main__":
    main()
