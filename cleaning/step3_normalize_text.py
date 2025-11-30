"""Normalize extracted book text: drop headers/footers, strip page markers, and rebuild paragraphs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "chemistry": {
        "src": ROOT / "cleaning" / "clean_text" / "OrganicChemistry-SAMPLE_9ADraVJ_clean.txt",
        "out": ROOT / "cleaning" / "clean_text" / "OrganicChemistry-SAMPLE_9ADraVJ_clean_normalized.txt",
    },
    "physics": {
        "src": ROOT / "cleaning" / "clean_text" / "UniversityPhysics15e_clean.txt",
        "out": ROOT / "cleaning" / "clean_text" / "UniversityPhysics15e_clean_normalized.txt",
    },
}

PAGE_MARK_RE = re.compile(r"\[\[PAGE \d+\]\]")
ONLY_NUMBER_RE = re.compile(r"^\d+$")
PAGE_PREFIX_RE = re.compile(r"^\d{2,4}\s+[A-Za-z]")
HEADER_PATTERNS = [
    "access for free at openstax.org",
]
HEADER_PREFIX_RES = [
    re.compile(r"^\d+\s+answer key\b", re.IGNORECASE),
    re.compile(r"^\d+\s+\d+\b"),  # page number + chapter/section artifacts
]


def is_header_footer(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if PAGE_MARK_RE.fullmatch(stripped):
        return True
    if ONLY_NUMBER_RE.fullmatch(stripped):
        return True
    if PAGE_PREFIX_RE.match(stripped):
        return True
    lowered = stripped.lower()
    if any(pat in lowered for pat in HEADER_PATTERNS):
        return True
    if any(regex.match(stripped) for regex in HEADER_PREFIX_RES):
        return True
    if "\x07" in stripped:  # stray control character artifact
        return True
    return False


def normalize_whitespace(text: str) -> str:
    # Collapse internal whitespace to single spaces and strip edges.
    return re.sub(r"\s+", " ", text).strip()


def normalize_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = []
    current_parts = []

    with src.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            if is_header_footer(raw_line):
                continue

            stripped = raw_line.strip()
            if stripped == "":
                if current_parts:
                    paragraphs.append(" ".join(current_parts))
                    current_parts = []
                continue

            current_parts.append(normalize_whitespace(stripped))

    if current_parts:
        paragraphs.append(" ".join(current_parts))

    with dest.open("w", encoding="utf-8") as outfile:
        outfile.write("\n\n".join(paragraphs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize cleaned text into paragraph form.")
    parser.add_argument("--subject", choices=CONFIG.keys(), help="Use preset paths for subject")
    parser.add_argument("--src", type=Path, help="Input cleaned txt (overrides subject default)")
    parser.add_argument("--out", type=Path, help="Output normalized txt (overrides subject default)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.subject:
        cfg = CONFIG[args.subject]
        src = cfg["src"]
        out = cfg["out"]
    else:
        src = args.src
        out = args.out

    if src is None or out is None:
        raise SystemExit("Provide --subject or both --src and --out.")

    if not src.exists():
        raise SystemExit(f"Source file not found: {src}")

    normalize_file(src, out)
    print(f"Wrote normalized text to {out}")
