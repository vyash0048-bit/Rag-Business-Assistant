"""
src/processing/clean.py

Cleaning / normalization layer for the RAG pipeline.

parse_pdf.py (and the SEC/NHTSA ingestion clients) produce raw, per-page
or per-record text in data/processed/*.jsonl. This module takes that
output and:

  1. Normalizes unicode and whitespace.
  2. Strips repeated headers/footers/boilerplate (detected by how often
     a line repeats across the pages of the same source document).
  3. Removes near-empty / junk records (e.g. pages that are just a
     table of contents entry or a page number).
  4. De-duplicates exact and near-duplicate records (the same
     boilerplate paragraph, e.g. "Cautionary Statement", often repeats
     verbatim across multiple filings/pages).
  5. Fills in normalized metadata (fiscal_year inferred from the source
     filename when not already present).

It is deliberately independent of any single source's schema — it
works on any record that has at least a "text" field, and preserves
whatever other metadata fields (company, source, page, doc_type, ...)
are already present.

Usage (clean every file in data/processed/, write *_clean.jsonl next
to each one, plus a merged data/processed/all_docs_clean.jsonl):

    python -m src.processing.clean
"""

import re
import json
import hashlib
import logging
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# A page/record shorter than this after cleaning is almost never useful
# context for RAG (title pages, blank pages, stray page numbers, etc.)
MIN_RECORD_LENGTH = 80

# A line that repeats on more than this fraction of a document's pages
# is treated as a running header/footer and stripped.
BOILERPLATE_LINE_THRESHOLD = 0.4

# Lines longer than this are never treated as boilerplate even if they
# repeat (avoids accidentally stripping a real repeated disclaimer
# paragraph that's part of the substantive content).
BOILERPLATE_MAX_LINE_LENGTH = 100


# --------------------------------------------------------------------------
# Text-level cleaning
# --------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize unicode (curly quotes, full-width chars, ligatures, etc.)."""
    text = unicodedata.normalize("NFKC", text)
    # Common PDF-extraction artifacts
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u00a0": " ",                   # non-breaking space
        "\uf0b7": "-",                   # PDF bullet glyph artifact
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace, strip stray control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def strip_common_pdf_noise(text: str) -> str:
    """Remove patterns that are near-universal PDF extraction noise."""
    patterns = [
        r"Page \d+ of \d+",
        r"^\s*\d+\s*$",                 # a line that's just a page number
        r"Table of Contents\s*$",
        r"^-{3,}$",                     # stray horizontal-rule artifacts
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def clean_text(text: str) -> str:
    """Full single-record cleaning pipeline (order matters)."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = strip_common_pdf_noise(text)
    text = collapse_whitespace(text)
    return text


# --------------------------------------------------------------------------
# Boilerplate (running header/footer) removal — document-level, since it
# requires seeing all pages of a document to know what "repeats" means.
# --------------------------------------------------------------------------

def _line_key(line: str) -> str:
    """Normalize a line for repetition comparison (case/space-insensitive)."""
    return re.sub(r"\s+", " ", line.strip().lower())


def find_boilerplate_lines(records: list[dict], group_key: str = "source") -> dict[str, set[str]]:
    """
    For each source document, find lines that repeat across a large
    fraction of its pages/records — these are running headers/footers
    (company name banners, "CONFIDENTIAL", filing titles, etc.).

    Returns: {source_name: {normalized_line, ...}}
    """
    grouped = defaultdict(list)
    for r in records:
        grouped[r.get(group_key, "unknown")].append(r["text"])

    boilerplate_by_source = {}
    for source, texts in grouped.items():
        n_pages = len(texts)
        if n_pages < 3:
            boilerplate_by_source[source] = set()
            continue
        line_counts = Counter()
        for text in texts:
            seen_this_page = set()
            for line in text.split("\n"):
                key = _line_key(line)
                if key and len(key) <= BOILERPLATE_MAX_LINE_LENGTH:
                    seen_this_page.add(key)
            line_counts.update(seen_this_page)
        threshold = max(2, int(n_pages * BOILERPLATE_LINE_THRESHOLD))
        boilerplate_by_source[source] = {
            line for line, count in line_counts.items() if count >= threshold
        }
    return boilerplate_by_source


def remove_boilerplate(text: str, boilerplate_lines: set[str]) -> str:
    """Strip any line matching a known boilerplate line for this document."""
    if not boilerplate_lines:
        return text
    kept = [line for line in text.split("\n") if _line_key(line) not in boilerplate_lines]
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """Hash of normalized text for exact-duplicate detection."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def near_dup_key(text: str, prefix_chars: int = 200) -> str:
    """
    Cheap near-duplicate signal: hash of the first N characters.
    Catches repeated disclaimers/cover pages that are byte-identical
    at the start but may have trailing extraction noise differences.
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())[:prefix_chars]
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Drop exact and near-duplicate records, keeping the first occurrence."""
    seen_exact, seen_near = set(), set()
    deduped = []
    for r in records:
        h = text_hash(r["text"])
        nd = near_dup_key(r["text"])
        if h in seen_exact or nd in seen_near:
            continue
        seen_exact.add(h)
        seen_near.add(nd)
        deduped.append(r)
    return deduped


# --------------------------------------------------------------------------
# Metadata normalization
# --------------------------------------------------------------------------

def infer_fiscal_year(source_filename: str) -> int | None:
    """Best-effort fiscal year extraction from a source filename, e.g.
    'HMC_2023_10k.pdf' or 'Honda_Report_2024-en-all.pdf' -> 2023 / 2024."""
    match = re.search(r"(20\d{2})", source_filename)
    return int(match.group(1)) if match else None


def normalize_metadata(record: dict) -> dict:
    record.setdefault("company", "unknown")
    record.setdefault("doc_type", "unknown")
    record.setdefault("page", None)
    if not record.get("fiscal_year"):
        record["fiscal_year"] = infer_fiscal_year(record.get("source", ""))
    return record


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

def clean_records(records: list[dict]) -> list[dict]:
    """
    Run the full cleaning pipeline over a list of raw records.
    Each record must have at least a "text" field.
    """
    # Step 1: per-record text cleaning
    for r in records:
        r["text"] = clean_text(r.get("text", ""))

    # Step 2: document-level boilerplate removal
    boilerplate_by_source = find_boilerplate_lines(records, group_key="source")
    for r in records:
        boilerplate = boilerplate_by_source.get(r.get("source", "unknown"), set())
        r["text"] = remove_boilerplate(r["text"], boilerplate)

    # Step 3: drop junk/near-empty records
    records = [r for r in records if len(r["text"]) >= MIN_RECORD_LENGTH]

    # Step 4: metadata normalization
    records = [normalize_metadata(r) for r in records]

    # Step 5: deduplication
    before = len(records)
    records = deduplicate_records(records)
    logger.info(f"Deduplication: {before} -> {len(records)} records "
                f"({before - len(records)} duplicates removed)")

    return records


def clean_file(path: Path) -> Path:
    """Clean a single data/processed/*.jsonl file, writing a *_clean.jsonl sibling."""
    records = [json.loads(line) for line in path.open() if line.strip()]
    logger.info(f"Loaded {len(records)} raw records from {path}")

    cleaned = clean_records(records)

    out_path = path.with_name(path.stem + "_clean.jsonl")
    with out_path.open("w") as f:
        for r in cleaned:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Wrote {len(cleaned)} cleaned records to {out_path}")
    return out_path


def run(processed_dir: Path = PROCESSED_DIR) -> None:
    """
    Clean every raw *.jsonl file in data/processed/ (skipping files that
    are already cleaning output) and also write a single merged file
    across all sources for convenience in the chunking step.
    """
    input_files = [
        p for p in processed_dir.glob("*.jsonl")
        if not p.stem.endswith("_clean") and p.name != "all_docs_clean.jsonl"
    ]
    if not input_files:
        logger.warning(f"No input .jsonl files found in {processed_dir}/")
        return

    all_cleaned = []
    for path in input_files:
        out_path = clean_file(path)
        all_cleaned.extend(json.loads(l) for l in out_path.open() if l.strip())

    merged_path = processed_dir / "all_docs_clean.jsonl"
    with merged_path.open("w") as f:
        for r in all_cleaned:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Merged {len(all_cleaned)} total cleaned records into {merged_path}")


if __name__ == "__main__":
    run()
