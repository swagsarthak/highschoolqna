"""Trim front/back matter (preface, dedication, index) by page range."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "chemistry": {
        "raw": ROOT / "cleaning" / "text_workdir" / "OrganicChemistry-SAMPLE_9ADraVJ_raw.txt",
        "clean": ROOT / "cleaning" / "clean_text" / "OrganicChemistry-SAMPLE_9ADraVJ_clean.txt",
        "start": 19,
        "end": 464,
    },
    "physics": {
        "raw": ROOT / "cleaning" / "text_workdir" / "UniversityPhysics15e_raw.txt",
        "clean": ROOT / "cleaning" / "clean_text" / "UniversityPhysics15e_clean.txt",
        "start": 30,
        "end": 1544,  # stop before appendices/answers/index
    },
}

PAGE_RE = re.compile(r"\[\[PAGE (\d+)\]\]")


def should_keep(page_number: int, start: int, end: int) -> bool:
    return start <= page_number <= end


def trim_pages(src: Path, dest: Path, start: int, end: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    write = False

    with src.open("r", encoding="utf-8") as infile, dest.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            match = PAGE_RE.match(line.strip())
            if match:
                current_page = int(match.group(1))
                write = should_keep(current_page, start, end)
                if not write:
                    continue
            if write:
                outfile.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim front/back matter by page range.")
    parser.add_argument("--subject", choices=CONFIG.keys(), help="Use preset start/end and paths for subject")
    parser.add_argument("--raw", type=Path, help="Raw txt with [[PAGE N]] markers (overrides subject default)")
    parser.add_argument("--out", type=Path, help="Output cleaned txt path (overrides subject default)")
    parser.add_argument("--start", type=int, help="Start page (inclusive, overrides subject default)")
    parser.add_argument("--end", type=int, help="End page (inclusive, overrides subject default)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.subject:
        cfg = CONFIG[args.subject]
        raw = cfg["raw"]
        out_txt = cfg["clean"]
        start = cfg["start"]
        end = cfg["end"]
    else:
        raw = args.raw
        out_txt = args.out
        start = args.start
        end = args.end

    if raw is None or out_txt is None or start is None or end is None:
        raise SystemExit("Provide --subject or --raw/--out/--start/--end.")

    if not raw.exists():
        raise SystemExit(f"Raw file not found: {raw}")

    trim_pages(raw, out_txt, start, end)
    print(f"Wrote trimmed text to {out_txt}")
