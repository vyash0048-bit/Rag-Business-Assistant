"""
src/ingestion/honda_ir_client.py

Downloads Honda's Integrated Report ("Honda Report") PDFs from Honda's
public Global Investor Relations / Sustainability site. These are
supplementary, unstructured narrative documents (strategy, governance,
value-creation story, financial & digital strategy, etc.) that add
useful RAG content beyond the numeric-heavy SEC 10-K filings.

Source page (human-browsable):
    https://global.honda/en/sustainability/integratedreport/

There is no JSON API for this content, so the script:
  1. Fetches the integrated-report index page.
  2. Parses out every linked PDF href (regex on the raw HTML — no heavy
     HTML-parsing dependency required).
  3. Falls back to a small set of manually-verified URLs if the page
     structure changes or parsing finds nothing, so the pipeline never
     silently returns zero documents.
  4. Downloads each PDF into data/raw/honda_ir/, skipping files that
     already exist (so re-runs are cheap and idempotent).

Usage:
    python -m src.ingestion.honda_ir_client
"""

import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")
logger = logging.getLogger(__name__)

INDEX_URL = "https://global.honda/en/sustainability/integratedreport/"
OUTPUT_DIR = Path("data/raw/honda_ir")

# Be a good citizen: identify the client and don't hammer the server.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; portfolio-research-bot/1.0; "
                  "contact: your-email@example.com)"
}
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 30

# Manually verified fallback list (checked against the live page).
# Used only if live scraping of INDEX_URL turns up no PDF links.
FALLBACK_PDF_URLS = [
    "https://global.honda/en/sustainability/integratedreport/pdf/Honda_Report_2025-en-all.pdf",
    "https://global.honda/en/sustainability/integratedreport/pdf/Honda_Report_2024-en-all.pdf",
    "https://global.honda/en/sustainability/integratedreport/pdf/Honda_Report_2023-en-all.pdf",
    "https://global.honda/en/sustainability/integratedreport/pdf/Honda_Report_2022-en-all-k.pdf",
]


def fetch_index_html(url: str = INDEX_URL) -> str:
    """Fetch the raw HTML of the integrated-report index page."""
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.text


def extract_pdf_urls(html: str, base_url: str = INDEX_URL) -> list[str]:
    """
    Pull every href ending in .pdf out of the page HTML and resolve it
    to an absolute URL. Deduplicates while preserving order.
    """
    hrefs = re.findall(r'href="([^"]+\.pdf[^"]*)"', html, flags=re.IGNORECASE)
    seen, urls = set(), []
    for href in hrefs:
        absolute = urljoin(base_url, href)
        # Strip tracking query params (utm_source etc.) for a clean, stable filename.
        clean = absolute.split("?")[0]
        if clean not in seen:
            seen.add(clean)
            urls.append(clean)
    return urls


def download_pdf(url: str, out_dir: Path = OUTPUT_DIR) -> Path | None:
    """Download a single PDF to out_dir, skipping it if already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest = out_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Already downloaded, skipping: {filename}")
        return dest

    logger.info(f"Downloading: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None

    dest.write_bytes(resp.content)
    logger.info(f"Saved: {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)")
    return dest


def run() -> list[Path]:
    """Full pipeline: discover PDF URLs, then download each one."""
    try:
        html = fetch_index_html()
        pdf_urls = extract_pdf_urls(html)
    except requests.RequestException as e:
        logger.warning(f"Could not fetch index page ({e}); using fallback URL list.")
        pdf_urls = []

    if not pdf_urls:
        logger.info("No PDF links found on the index page; using verified fallback list.")
        pdf_urls = FALLBACK_PDF_URLS

    logger.info(f"Found {len(pdf_urls)} PDF(s) to fetch.")

    saved_paths = []
    for url in pdf_urls:
        path = download_pdf(url)
        if path:
            saved_paths.append(path)
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(f"Done. {len(saved_paths)}/{len(pdf_urls)} PDFs saved to {OUTPUT_DIR}/")
    return saved_paths


if __name__ == "__main__":
    run()
