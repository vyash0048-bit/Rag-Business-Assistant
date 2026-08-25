import json, sqlite3
from pathlib import Path

conn = sqlite3.connect("data/structured/business.db")
conn.executescript(open("src/structured/schema.sql").read())

def load_financials():
    for f in Path("data/raw/sec").glob("*_facts.json"):
        company = f.name.split("_")[0]
        facts = json.loads(f.read_text())
        for taxonomy in ["us-gaap", "ifrs-full"]:
            tax_facts = facts.get("facts", {}).get(taxonomy, {})
            for metric in ["Revenues", "Revenue", "ResearchAndDevelopmentExpense", "NetIncomeLoss", "ProfitLoss"]:
                db_metric = metric
                if metric == "Revenue": db_metric = "Revenues"
                if metric == "ProfitLoss": db_metric = "NetIncomeLoss"
                units = tax_facts.get(metric, {}).get("units", {})
                for unit, entries in units.items():
                    for e in entries:
                        if e.get("form") in ["10-K", "20-F"]:
                            conn.execute("INSERT INTO financials VALUES (?,?,?,?,?)",
                                         (company, e.get("fy"), db_metric, e.get("val"), unit))
    conn.commit()

def load_recalls():
    for f in Path("data/raw/nhtsa").glob("*_recalls.json"):
        parts = f.stem.split("_")
        company, model, year = parts[0], parts[1], parts[2]
        data = json.loads(f.read_text())
        for r in data.get("results", []):
            conn.execute("INSERT INTO recalls VALUES (?,?,?,?,?,?)",
                         (company, model, year, r.get("ReportReceivedDate"),
                          r.get("Component"), r.get("Consequence")))
    conn.commit()

if __name__ == "__main__":
    load_financials(); load_recalls()
