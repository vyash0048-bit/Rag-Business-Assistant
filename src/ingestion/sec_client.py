# src/ingestion/sec_client.py
import requests, json, time, os
from pathlib import Path

HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}
COMPANIES = {"HMC": "0000715153", "TM": "0001094517", "F": "0000037996"}

def fetch_company_facts(ticker, cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    out = Path(f"data/raw/sec/{ticker}_facts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r.text)
    return out

def fetch_recent_10k_urls(ticker, cik):
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=HEADERS).json()
    forms = r["filings"]["recent"]
    urls = []
    for i, form in enumerate(forms["form"]):
        if form == "10-K":
            accn = forms["accessionNumber"][i].replace("-", "")
            doc = forms["primaryDocument"][i]
            urls.append(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{doc}")
    return urls[:3]  # last 3 annual filings

if __name__ == "__main__":
    for ticker, cik in COMPANIES.items():
        fetch_company_facts(ticker, cik)
        for url in fetch_recent_10k_urls(ticker, cik):
            fname = f"data/raw/sec/{ticker}_{url.split('/')[-1]}"
            r = requests.get(url, headers=HEADERS)
            Path(fname).write_bytes(r.content)
            time.sleep(0.3)  # SEC rate limit courtesy