# src/ingestion/nhtsa_client.py
import requests, json
from pathlib import Path

MAKES_MODELS = [("honda", "civic"), ("honda", "accord"), ("honda", "cr-v")]

def fetch_recalls(make, model, years=range(2020, 2026)):
    Path("data/raw/nhtsa").mkdir(parents=True, exist_ok=True)
    for y in years:
        r = requests.get("https://api.nhtsa.gov/recalls/recallsByVehicle",
                          params={"make": make, "model": model, "modelYear": y})
        Path(f"data/raw/nhtsa/{make}_{model}_{y}_recalls.json").write_text(r.text)

if __name__ == "__main__":
    for make, model in MAKES_MODELS:
        fetch_recalls(make, model)