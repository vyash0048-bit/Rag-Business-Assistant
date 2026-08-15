import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

import os
import numpy as np

def embed_chunks(path="data/chunks/chunks.jsonl"):
    vec_path = "data/embeddings/vectors.npy"
    rec_path = "data/embeddings/records.json"
    
    if os.path.exists(vec_path) and os.path.exists(rec_path):
        print("Loading cached embeddings from disk...")
        records = json.load(open(rec_path))
        vectors = np.load(vec_path)
        return records, vectors
        
    records = [json.loads(l) for l in open(path)]
    texts = ["Represent this document for retrieval: " + r["text"] for r in records]
    print(f"Encoding {len(texts)} chunks...")
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    os.makedirs("data/embeddings", exist_ok=True)
    np.save(vec_path, vectors)
    with open(rec_path, "w") as f:
        json.dump(records, f)
        
    return records, vectors

if __name__ == "__main__":
    records, vectors = embed_chunks()
    print(vectors.shape)  # (num_chunks, 1024)
