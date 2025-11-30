"""Chunk normalized text into overlapping pieces for RAG ingestion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
CONFIG = {
    "chemistry": {
        "src": ROOT / "cleaning" / "clean_text" / "OrganicChemistry-SAMPLE_9ADraVJ_clean_normalized.txt",
        "out": ROOT / "cleaning" / "chunks" / "chemistry" / "OrganicChemistry-SAMPLE_9ADraVJ_chunks.jsonl",
    },
    "physics": {
        "src": ROOT / "cleaning" / "clean_text" / "UniversityPhysics15e_clean_normalized.txt",
        "out": ROOT / "cleaning" / "chunks" / "physics" / "UniversityPhysics15e_chunks.jsonl",
    },
}

# Chunking parameters
TARGET_WORDS = 280
MAX_WORDS = 340
OVERLAP_WORDS = 40


def load_paragraphs(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def make_chunks(paragraphs: List[str]) -> Iterable[Tuple[str, int]]:
    buffer: List[str] = []
    word_count = 0

    def flush(buf: List[str]) -> Tuple[str, List[str]]:
        joined = "\n\n".join(buf).strip()
        words = joined.split()
        tail = words[-OVERLAP_WORDS:] if len(words) > OVERLAP_WORDS else words
        return joined, [" ".join(tail)] if tail else []

    def emit_long_paragraph(words: List[str]) -> Iterable[Tuple[str, int]]:
        step = max(1, TARGET_WORDS - OVERLAP_WORDS)
        for start in range(0, len(words), step):
            slice_words = words[start : start + TARGET_WORDS]
            if not slice_words:
                continue
            chunk_text = " ".join(slice_words)
            yield chunk_text, len(slice_words)

    for para in paragraphs:
        words_list = para.split()
        para_words = len(words_list)

        if not words_list:
            continue

        # Handle paragraphs longer than MAX_WORDS by splitting them directly.
        if para_words > MAX_WORDS:
            if buffer:
                chunk_text, overlap_buf = flush(buffer)
                yield chunk_text, len(chunk_text.split())
                buffer = overlap_buf.copy()
                word_count = len(" ".join(buffer).split()) if buffer else 0

            last_tail: List[str] = []
            for chunk_text, wc in emit_long_paragraph(words_list):
                yield chunk_text, wc
                last_tail = chunk_text.split()[-OVERLAP_WORDS:]

            if last_tail:
                buffer = [" ".join(last_tail)]
                word_count = len(last_tail)
            else:
                buffer = []
                word_count = 0
            continue

        while buffer and (word_count + para_words > MAX_WORDS):
            if word_count <= OVERLAP_WORDS:
                # Drop overlap to allow the new paragraph to fit comfortably.
                buffer = []
                word_count = 0
                break
            chunk_text, overlap_buf = flush(buffer)
            yield chunk_text, len(chunk_text.split())
            buffer = overlap_buf.copy()
            word_count = len(" ".join(buffer).split()) if buffer else 0

        buffer.append(para)
        word_count += para_words

    if buffer:
        chunk_text, _ = flush(buffer)
        yield chunk_text, len(chunk_text.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk normalized text into overlapping pieces.")
    parser.add_argument("--subject", choices=CONFIG.keys(), help="Use preset paths for subject")
    parser.add_argument("--src", type=Path, help="Normalized input txt (overrides subject)")
    parser.add_argument("--out", type=Path, help="Output chunks JSONL (overrides subject)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subject:
        cfg = CONFIG[args.subject]
        src = cfg["src"]
        out_path = cfg["out"]
    else:
        src = args.src
        out_path = args.out

    if src is None or out_path is None:
        raise SystemExit("Provide --subject or both --src and --out.")

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    paragraphs = load_paragraphs(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, (chunk, wc) in enumerate(make_chunks(paragraphs), start=1):
            rec = {"id": f"chunk-{idx:05d}", "word_count": wc, "text": chunk}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote chunks to {out_path}")


if __name__ == "__main__":
    main()
