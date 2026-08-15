import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Using token-aware splitter as requested for better sizing (~180 tokens)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=180, chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " "],
)

def run():
    Path("data/chunks").mkdir(parents=True, exist_ok=True)
    out = open("data/chunks/chunks.jsonl", "w")
    cid = 0
    try:
        with open("data/processed/all_docs_clean.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                for chunk in splitter.split_text(rec["text"]):
                    out.write(json.dumps({
                        "chunk_id": f"chunk_{cid}",
                        "text": chunk,
                        "company": rec["company"],
                        "source": rec["source"],
                        "page": rec["page"],
                        "doc_type": rec["doc_type"],
                    }) + "\n")
                    cid += 1
    except FileNotFoundError:
        pass
    out.close()

if __name__ == "__main__":
    run()
