"""
src/processing/parse_pdf.py

Raw PDF -> per-page text extraction.

This module owns ONLY extraction: pulling text (and page numbers) out
of PDFs and writing one JSON record per page to data/processed/. It
deliberately does NOT clean, normalize, deduplicate, or strip
boilerplate — that is entirely the job of src/processing/clean.py,
which runs as the next pipeline stage. Keeping the two separate means:

  - parse_pdf.py can be re-run cheaply whenever new PDFs are added,
    without re-deciding cleaning rules.
  - clean.py can see ALL pages of a document at once (needed to detect
    repeated headers/footers), which wouldn't be possible if cleaning
    happened one page at a time during extraction.

Handles every PDF-based source currently in the pipeline:
  - data/raw/sec/*.pdf        -> data/processed/sec_docs.jsonl        (doc_type="10-K")
  - data/raw/honda_ir/*.pdf   -> data/processed/honda_ir_docs.jsonl   (doc_type="integrated_report")

Usage:
    python -m src.processing.parse_pdf
"""

import json
import logging
from pathlib import Path

import pdfplumber

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Each source folder maps to (doc_type, output filename, company_from_filename)
SOURCES = {
    "sec": {"doc_type": "10-K", "output_file": "sec_docs.jsonl"},
    "honda_ir": {"doc_type": "integrated_report", "output_file": "honda_ir_docs.jsonl"},
}


def infer_company(filename: str) -> str:
    """
    Best-effort company ticker from filename conventions used by the
    ingestion clients, e.g. 'HMC_0000...pdf' -> 'HMC'. Honda IR reports
    don't carry a ticker prefix, so default to Honda's ticker for that
    source since global.honda only publishes Honda's own reports.
    """
    stem = filename.split("_")[0].upper()
    if stem in {"HMC", "TM", "F"}:
        return stem
    if "honda" in filename.lower():
        return "HMC"
    return "unknown"


def extract_pdf_pages(path: Path) -> list[dict]:
    """
    Extract raw text per page from a single PDF. No cleaning is applied
    here — text is passed through exactly as pdfplumber returns it,
    aside from dropping pages that produced no extractable text at all
    (e.g. pure-image pages), which is a parsing concern, not a
    cleaning concern.
    """
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text is None:
                    logger.debug(f"{path.name} page {i + 1}: no extractable text (skipped)")
                    continue
                pages.append({"page": i + 1, "text": text})
    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")
    return pages


def parse_source(source_dir: Path, doc_type: str) -> list[dict]:
    """Parse every PDF in a source directory into raw page records."""
    records = []
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {source_dir}/")
        return records

    for pdf_path in pdf_paths:
        logger.info(f"Parsing {pdf_path.name} ...")
        pages = extract_pdf_pages(pdf_path)
        company = infer_company(pdf_path.name)
        for p in pages:
            records.append({
                "source": pdf_path.name,
                "company": company,
                "doc_type": doc_type,
                "page": p["page"],
                "text": p["text"],
            })
        logger.info(f"  -> {len(pages)} pages extracted")

    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {len(records)} raw page records to {out_path}")


def run() -> None:
    for source_name, cfg in SOURCES.items():
        source_dir = RAW_DIR / source_name
        records = parse_source(source_dir, doc_type=cfg["doc_type"])
        if records:
            write_jsonl(records, PROCESSED_DIR / cfg["output_file"])
        else:
            logger.warning(f"Skipping output for '{source_name}': nothing extracted")


if __name__ == "__main__":
    run()
