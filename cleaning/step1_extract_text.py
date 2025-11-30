"""Simple extractor to pull raw text from a source PDF into text_workdir.

Usage:
  # single file
  python cleaning/step1_extract_text.py --subject chemistry
  python cleaning/step1_extract_text.py --pdf cleaning/raw_pdf/YourBook.pdf --out cleaning/text_workdir/YourBook_raw.txt
  # all PDFs in a folder (writes one _raw.txt per PDF)
  python cleaning/step1_extract_text.py --pdf-dir cleaning/raw_pdf/lebo1dd --out-dir cleaning/text_workdir
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import sys

from pypdf import PdfReader, errors as pypdf_errors

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "chemistry": {
        "pdf": ROOT / "cleaning" / "raw_pdf" / "OrganicChemistry-SAMPLE_9ADraVJ.pdf",
        "out": ROOT / "cleaning" / "text_workdir" / "OrganicChemistry-SAMPLE_9ADraVJ_raw.txt",
    },
    "physics": {
        "pdf": ROOT
        / "cleaning"
        / "raw_pdf"
        / "University Physics with Modern Physics 15th Edition By Hugh D. Young_compressed.pdf",
        "out": ROOT / "cleaning" / "text_workdir" / "UniversityPhysics15e_raw.txt",
    },
}


def extract_pypdf(pdf_path: Path, out_path: Path) -> None:
    reader = PdfReader(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            out.write(f"[[PAGE {page_num}]]\n")
            out.write(text.strip() + "\n\n")


def extract_pymupdf(pdf_path: Path, out_path: Path) -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyMuPDF (fitz) is required for this backend. Install with: pip install pymupdf") from exc

    doc = fitz.open(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text() or ""
            out.write(f"[[PAGE {page_num + 1}]]\n")
            out.write(text.strip() + "\n\n")


def extract(pdf_path: Path, out_path: Path, backend: str = "auto") -> None:
    if backend == "pypdf":
        extract_pypdf(pdf_path, out_path)
        return
    if backend == "pymupdf":
        extract_pymupdf(pdf_path, out_path)
        return

    # auto: try pypdf, fall back to pymupdf on decompression or other extraction errors.
    try:
        extract_pypdf(pdf_path, out_path)
        return
    except pypdf_errors.LimitReachedError as exc:
        sys.stderr.write(f"pypdf hit decompression limit on {pdf_path.name}; falling back to PyMuPDF.\n")
        extract_pymupdf(pdf_path, out_path)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"pypdf failed on {pdf_path.name} ({exc}); falling back to PyMuPDF.\n")
        extract_pymupdf(pdf_path, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw text from a PDF with page markers.")
    parser.add_argument("--subject", choices=DEFAULTS.keys(), help="Use preset PDF/output for subject")
    parser.add_argument("--pdf", type=Path, help="Path to PDF file (overrides subject default)")
    parser.add_argument("--out", type=Path, help="Output txt path (overrides subject default)")
    parser.add_argument("--pdf-dir", type=Path, help="Process all PDFs in a directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for raw txt files when using --pdf-dir (defaults to text_workdir)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pypdf", "pymupdf"],
        default="auto",
        help="Text extraction backend (default: auto, tries pypdf then falls back to PyMuPDF).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.pdf_dir:
        if not args.pdf_dir.exists():
            raise SystemExit(f"PDF directory not found: {args.pdf_dir}")
        out_dir = args.out_dir or (ROOT / "cleaning" / "text_workdir")
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_files: Iterable[Path] = sorted(args.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise SystemExit(f"No PDFs found in {args.pdf_dir}")
        for pdf_file in pdf_files:
            out_txt = out_dir / f"{pdf_file.stem}_raw.txt"
            extract(pdf_file, out_txt, backend=args.backend)
            print(f"Wrote raw text to {out_txt}")
        raise SystemExit(0)

    if args.subject:
        pdf = DEFAULTS[args.subject]["pdf"]
        out_txt = DEFAULTS[args.subject]["out"]
    else:
        pdf = args.pdf
        out_txt = args.out

    if pdf is None or out_txt is None:
        raise SystemExit("Provide --subject or both --pdf and --out.")

    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    extract(pdf, out_txt, backend=args.backend)
    print(f"Wrote raw text to {out_txt}")
