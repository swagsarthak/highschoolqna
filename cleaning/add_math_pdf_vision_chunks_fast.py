# add_math_pdf_vision_chunks_fast.py
# Fully compatible with step1b_extract_images output

from __future__ import annotations
import json
from pathlib import Path
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "cleaning" / "images" / "math"
OUT_PATH = ROOT / "cleaning" / "chunks" / "math" / "vision_math_pdf_chunks.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen2.5vl:3b"

import base64

def describe(img_path: Path, rec: dict) -> str:
    try:
        # read image and encode base64
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": VISION_MODEL,
                "prompt": (
                    "You are a mathematical diagram describer. "
                    "Summarize only the mathematical content. "
                    "Do NOT explain steps or solve anything.\n"
                    f"Image page {rec['page']} from book {rec['book']}."
                ),
                "images": [b64],   # <-- THIS IS THE FIX
                "stream": False,
            },
            timeout=45,
        )

        r.raise_for_status()
        return r.json().get("response", "").strip()

    except Exception as e:
        return f"[VisionError: {e}]"


def load_all_metadata():
    """Load all <book>_images.jsonl files."""
    recs = []
    for meta in IMG_DIR.glob("*_images.jsonl"):
        recs.extend(json.loads(line) for line in meta.open("r", encoding="utf-8"))
    return recs

def process(rec: dict) -> dict:
    img_path = ROOT / rec["image_path"]     # key from step1b
    desc = describe(img_path, rec)

    return {
        "id": f"vision-math-pdf-{rec['book']}-{rec['page']:04d}",
        "text": desc,
        "image_path": rec["image_path"],
        "source_pdf": rec["book"],
        "page": rec["page"],
    }

def main():
    print("\n=== FAST Vision PDF → Chunk Builder ===")

    all_meta = load_all_metadata()
    print(f"[INFO] Loaded {len(all_meta)} image metadata records.")

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for out in tqdm(pool.map(process, all_meta), total=len(all_meta)):
            results.append(out)

    print(f"[OK] processed: {len(results)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[WRITE] {OUT_PATH}")

if __name__ == "__main__":
    main()
