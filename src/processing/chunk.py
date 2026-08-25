import json
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token-aware splitter sized for financial documents (~512 tokens per chunk)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=64,
    separators=["\n\n", "\n", ". ", "; ", " "],
)


def run():
    input_path = Path("data/processed/all_docs_clean.jsonl")
    output_path = Path("data/chunks/chunks.jsonl")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w") as out, open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            source_base = Path(rec["source"]).stem
            page = rec["page"]
            page_chunk_idx = 0

            for chunk in splitter.split_text(rec["text"]):
                chunk_id = f"{source_base}_p{page}_c{page_chunk_idx}"
                out.write(json.dumps({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "company": rec["company"],
                    "source": rec["source"],
                    "page": page,
                    "doc_type": rec["doc_type"],
                    "fiscal_year": rec.get("fiscal_year", ""),
                }) + "\n")
                page_chunk_idx += 1
                total += 1

    print(f"Chunked {total} passages into {output_path}")


if __name__ == "__main__":
    run()
