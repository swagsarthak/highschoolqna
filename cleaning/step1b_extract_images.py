"""Extract images from a PDF into a structured folder and emit metadata.

Usage examples:
  python cleaning/step1b_extract_images.py --pdf cleaning/raw_pdf/lebo1dd/lebo101.pdf --subject biology
  python cleaning/step1b_extract_images.py --pdf cleaning/raw_pdf/lebo1dd/lebo101.pdf --subject biology --book lebo101 --out-dir cleaning/images/biology/lebo101 --ocr

Outputs:
  - Image files under: cleaning/images/<subject>/<book>/<book>_p####_i#.png
  - Metadata JSONL:     cleaning/images/<subject>/<book>_images.jsonl

Notes:
  - OCR is optional and uses pytesseract + Pillow if installed; otherwise it is skipped with a warning.
  - Page numbers are 1-based; image index resets on each page.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF (fitz) is required. Install with: pip install pymupdf") from exc


ROOT = Path(__file__).resolve().parent.parent


def file_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.md5(data).hexdigest()


def keep_record(rec: dict, min_side: int, min_pixels: int, max_ratio: float) -> bool:
    w = rec.get("width") or 0
    h = rec.get("height") or 0
    if w < min_side or h < min_side:
        return False
    if w * h < min_pixels:
        return False
    ratio = max(w, h) / max(1, min(w, h))
    if ratio > max_ratio:
        return False
    return True


def load_ocr_dependencies():
    try:
        from PIL import Image
        import pytesseract

        return Image, pytesseract
    except ImportError:
        return None, None


def run_ocr(img_bytes: bytes, Image, pytesseract) -> Optional[str]:
    if Image is None or pytesseract is None:
        return None
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            text = pytesseract.image_to_string(im)
            return text.strip()
    except Exception:
        return None


def extract_images(
    pdf_path: Path,
    out_dir: Path,
    meta_path: Path,
    book: str,
    do_ocr: bool,
    filter_opts: Optional[dict] = None,
    filtered_meta_path: Optional[Path] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    Image, pytesseract = load_ocr_dependencies() if do_ocr else (None, None)

    doc = fitz.open(pdf_path)
    records = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        images = page.get_images(full=True)
        if not images:
            continue

        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            info = doc.extract_image(xref)
            img_bytes = info["image"]
            ext = info.get("ext", "png")
            base_name = f"{book}_p{page_number + 1:04d}_i{img_index}"
            img_path = out_dir / f"{base_name}.{ext}"

            img_path.write_bytes(img_bytes)

            rec: Dict[str, Any] = {
                "book": book,
                "page": page_number + 1,
                "image_path": str(img_path.relative_to(ROOT)),
                "width": info.get("width"),
                "height": info.get("height"),
                "ext": ext,
            }

            if do_ocr:
                ocr_text = run_ocr(img_bytes, Image, pytesseract)
                if ocr_text:
                    rec["ocr_text"] = ocr_text

            records.append(rec)

    with meta_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Extracted {len(records)} images from {pdf_path.name}")
    print(f"Images written under: {out_dir}")
    print(f"Metadata: {meta_path}")

    if filter_opts:
        if filtered_meta_path is None:
            filtered_meta_path = meta_path.with_name(meta_path.stem + "_filtered" + meta_path.suffix)
        filtered_meta_path.parent.mkdir(parents=True, exist_ok=True)
        seen_hashes: set[str] = set()
        kept: List[dict] = []

        for rec in records:
            if not keep_record(
                rec,
                min_side=filter_opts["min_side"],
                min_pixels=filter_opts["min_pixels"],
                max_ratio=filter_opts["max_ratio"],
            ):
                continue
            img_path = (ROOT / rec["image_path"]).resolve()
            if not img_path.exists():
                continue
            hsh = file_hash(img_path)
            if hsh in seen_hashes:
                continue
            seen_hashes.add(hsh)
            rec = dict(rec)
            rec["hash"] = hsh
            kept.append(rec)

        with filtered_meta_path.open("w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Filtered kept {len(kept)} images -> {filtered_meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract images from a PDF and write metadata JSONL.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the PDF file")
    parser.add_argument(
        "--subject",
        default="biology",
        help="Subject folder under cleaning/images (default: biology)",
    )
    parser.add_argument(
        "--book",
        help="Book name for file naming; defaults to PDF stem (e.g., lebo101)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for images; default: cleaning/images/<subject>/<book>/",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="Output metadata JSONL; default: cleaning/images/<subject>/<book>_images.jsonl",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run OCR on each image if pytesseract + Pillow are installed",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Also write a filtered metadata JSONL (drops tiny/logos and deduplicates)",
    )
    parser.add_argument(
        "--filtered-meta",
        type=Path,
        help="Output filtered metadata JSONL; default: <meta> with _filtered suffix",
    )
    parser.add_argument("--min-side", type=int, default=150, help="Filter: minimum width/height (px)")
    parser.add_argument("--min-pixels", type=int, default=20000, help="Filter: minimum total pixels")
    parser.add_argument("--max-ratio", type=float, default=12.0, help="Filter: maximum aspect ratio")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    book = args.book or Path(args.pdf).stem
    base_images_dir = ROOT / "cleaning" / "images" / args.subject
    out_dir = args.out_dir or (base_images_dir / book)
    meta_path = args.meta or (base_images_dir / f"{book}_images.jsonl")
    filtered_meta = args.filtered_meta or meta_path.with_name(meta_path.stem + "_filtered" + meta_path.suffix)

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    filter_opts = None
    if args.filter:
        filter_opts = {
            "min_side": args.min_side,
            "min_pixels": args.min_pixels,
            "max_ratio": args.max_ratio,
        }

    extract_images(
        args.pdf,
        out_dir,
        meta_path,
        book=book,
        do_ocr=args.ocr,
        filter_opts=filter_opts,
        filtered_meta_path=filtered_meta if args.filter else None,
    )
