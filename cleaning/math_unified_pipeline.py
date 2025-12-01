"""
math_unified_pipeline.py

Unified Math Pipeline:
- Processes ALL OpenStax Math PDFs
- Extracts text from each PDF
- Normalizes text
- Chunks text into math-friendly segments
- Merges all chunks from all textbooks
- Adds additional QA datasets (GSM8K, Grade-Math-18K, SVAMP, ASDiv, MathQA)
- Embeds with Ollama (mxbai-embed-large)

Output (compatible with full RAG system):
    cleaning/chunks/math/math_unified_chunks.jsonl
    cleaning/chunks/math/math_unified_embeddings_ollama.npy
    cleaning/chunks/math/math_unified_embeddings_ollama_meta.jsonl
"""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
from typing import Dict, Iterable

import numpy as np
import pypdf
import requests
from tqdm import tqdm
from datasets import load_dataset


# ---------------------------------------------------------------------
# Paths and PDF list
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = ROOT / "cleaning" / "raw_pdf" / "math"
TEXT_DIR = ROOT / "cleaning" / "text_workdir" / "math_texts"
NORM_DIR = ROOT / "cleaning" / "clean_text" / "math"
CHUNK_DIR = ROOT / "cleaning" / "chunks" / "math"

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)
NORM_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# These files must already exist (you run download_math_pdfs.py first)
MATH_PDFS = [
    "Prealgebra2e-WEB_0qbw93r.pdf",
    "College-Algebra-2e-WEB.pdf",
    "Algebra-and-Trigonometry-2e-WEB.pdf",
    "Precalculus_2e-WEB.pdf",
    "Calculus_Volume_1_-_WEB_68M1Z5W.pdf",
    "Calculus_Volume_2_-_WEB.pdf",
    "Calculus_Volume_3_-_WEB.pdf",
    "Statistics-WEB.pdf",
]


# Final outputs
UNIFIED_CHUNKS = CHUNK_DIR / "math_unified_chunks.jsonl"
EMBED_OUT = CHUNK_DIR / "math_unified_embeddings_ollama.npy"
META_OUT = CHUNK_DIR / "math_unified_embeddings_ollama_meta.jsonl"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def extract_pdf(pdf: Path) -> str:
    """Extract raw text with page markers."""
    reader = pypdf.PdfReader(str(pdf))
    pages = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text() or ""
        pages.append(f"[[PAGE {i+1}]]\n{text}\n")
    return "\n".join(pages)


def normalize_text(raw: str) -> str:
    """Cleaning rules tuned for OpenStax formatting."""
    lines = []
    skip_phrases = ["access for free", "OpenStax"]

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(p.lower() in line.lower() for p in skip_phrases):
            continue
        line = " ".join(line.split())
        lines.append(line)

    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30):
    """Math-friendly chunking."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------
# 1. Extract + Normalize + Chunk ALL math PDFs
# ---------------------------------------------------------------------
def process_all_textbooks() -> Iterable[Dict]:
    """Returns textbook chunks across all books as iterable records."""
    counter = 0

    for filename in MATH_PDFS:
        pdf_path = PDF_DIR / filename
        print(f"\n[PDF] Processing {filename}")

        raw_text = extract_pdf(pdf_path)
        norm_text = normalize_text(raw_text)
        chunks = chunk_text(norm_text)

        for c in chunks:
            yield {
                "id": f"math-text-{counter:07d}",
                "word_count": len(c.split()),
                "text": c,
                "source_pdf": filename,
            }
            counter += 1

        print(f"[OK] {filename}: {len(chunks)} chunks")


# ---------------------------------------------------------------------
# 2. Add HF QA datasets
# ---------------------------------------------------------------------
def qa_chunk(q: str, a: str, dataset: str, idx: int) -> Dict:
    text = f"[DATASET: {dataset}]\nQuestion: {q}\nAnswer: {a}"
    return {
        "id": f"{dataset}-qa-{idx:06d}",
        "word_count": len(text.split()),
        "text": text,
        "source_pdf": dataset,
    }


def load_all_qa() -> Iterable[Dict]:
    print("\n[HF] Loading Grade-Math-18K...")
    ds = load_dataset("prithivMLmods/Grade-Math-18K", split="train")
    for i, ex in enumerate(ds):
        if i >= 3000: break
        yield qa_chunk(ex["question"], ex["answer"], "Grade-Math-18K", i)

    print("[HF] Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for i, ex in enumerate(ds):
        if i >= 2000: break
        yield qa_chunk(ex["question"], ex["answer"], "GSM8K", i)

    print("[HF] Loading AQUA-RAT...")
    ds = load_dataset("aqua_rat", split="train")
    for i, ex in enumerate(ds):
        if i >= 1500: break
        yield qa_chunk(ex["question"], ex["rationale"], "AQUA-RAT", i)

    print("[HF] Loading SVAMP...")
    ds = load_dataset("ChilleD/SVAMP", split="train")
    for i, ex in enumerate(ds):
        if i >= 800: break
        yield qa_chunk(ex["Body"], str(ex["Answer"]), "SVAMP", i)

    print("[HF] Loading ASDiv (validation only)...")
    ds = load_dataset("EleutherAI/asdiv", split="validation")
    for i, ex in enumerate(ds):
        if i >= 800: break
        yield qa_chunk(ex["question"], ex["answer"], "ASDiv", i)

    print("[HF] Loading MathQA...")
    ds = load_dataset("math_qa", split="train")
    for i, ex in enumerate(ds):
        if i >= 2000: break
        opts = ex["options"].split(",")
        label_idx = ord(ex["correct"]) - ord("a")
        try:
            ans = opts[label_idx].strip()
        except Exception:
            ans = ex["correct"]
        yield qa_chunk(ex["Problem"], ans, "MathQA", i)



# ---------------------------------------------------------------------
# 3. Write unified chunk file
# ---------------------------------------------------------------------
def build_unified_chunks():
    if UNIFIED_CHUNKS.exists():
        print(f"[SKIP] Found existing unified chunk file, reuse it → {UNIFIED_CHUNKS}")
        print("       (Delete this file if you want to rebuild chunks.)")
        return

    print(f"\n[WRITE] Building unified chunk file → {UNIFIED_CHUNKS}")
    with UNIFIED_CHUNKS.open("w", encoding="utf-8") as f:
        # textbook chunks
        for rec in tqdm(process_all_textbooks(), desc="Textbooks"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # QA chunks
        for rec in tqdm(load_all_qa(), desc="HF QA"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] unified chunk file created: {UNIFIED_CHUNKS}")

    VISION = CHUNK_DIR / "vision_math_pdf_chunks.jsonl"
    if VISION.exists():
        print(f"[ADD] Adding vision chunks from {VISION}")
        with VISION.open("r", encoding="utf-8") as vf, \
                UNIFIED_CHUNKS.open("a", encoding="utf-8") as uf:
            for line in vf:
                try:
                    obj = json.loads(line)
                    uf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except Exception:
                    continue
        print("[ADD] Vision chunks merged into unified chunks")
    else:
        print("[WARN] No vision chunks found — skipping vision merge.")


# ---------------------------------------------------------------------
# 4. Run embedding with Ollama step6
# ---------------------------------------------------------------------
def embed_unified_math():
    print("\n[EMBED] Rebuilding embeddings for unified math corpus...")

    # Load chunks in deterministic order
    chunks = []
    ids = []
    with open(UNIFIED_CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(rec["text"])
            ids.append(rec["id"])

    total = len(chunks)
    print(f"[INFO] Loaded {total} chunks")

    # Prepare output files
    emb_out = []
    meta_f = open(META_OUT, "w", encoding="utf-8")

    # Embed each chunk
    for i, (cid, text) in enumerate(zip(ids, chunks)):
        payload = {
            "model": "mxbai-embed-large",
            "input": [text]
        }
        r = requests.post("http://localhost:11434/api/embed", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Embedding failed at index {i}: {r.text}")

        vec = r.json()["embeddings"][0]
        emb_out.append(vec)

        meta_f.write(json.dumps({"id": cid, "index": i}) + "\n")

        if i % 2000 == 0:
            print(f"[EMBED] {i}/{total} ...")

    meta_f.close()
    np.save(EMBED_OUT, np.array(emb_out, dtype=np.float32))

    print(f"[OK] Saved embeddings: {EMBED_OUT}")
    print(f"[OK] Saved meta:       {META_OUT}")


# ---------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------
def main():
    print("\n=============================================")
    print(" UNIFIED MATH PIPELINE START")
    print("=============================================")

    build_unified_chunks()
    embed_unified_math()

    print("\n=============================================")
    print(" UNIFIED MATH PIPELINE COMPLETE")
    print("=============================================")
    print(f"Chunks:     {UNIFIED_CHUNKS}")
    print(f"Embeddings: {EMBED_OUT}")
    print(f"Meta:       {META_OUT}")
    print("=============================================")


if __name__ == "__main__":
    main()
