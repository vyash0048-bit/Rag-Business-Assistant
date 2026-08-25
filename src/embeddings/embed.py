import os
import json
import numpy as np

def embed_chunks(path="data/chunks/chunks.jsonl"):
    vec_path = "data/embeddings/vectors.npy"
    rec_path = "data/embeddings/records.json"
    
    if os.path.exists(vec_path) and os.path.exists(rec_path) and os.path.exists(path):
        if os.path.getmtime(vec_path) > os.path.getmtime(path):
            print("Loading cached embeddings from disk...")
            with open(rec_path, "r") as f:
                records = json.load(f)
            vectors = np.load(vec_path)
            return records, vectors
        
    with open(path, "r") as f:
        records = [json.loads(l) for l in f]
    
    texts = ["Represent this document for retrieval: " + r["text"] for r in records]
    print(f"Encoding {len(texts)} chunks...")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    os.makedirs("data/embeddings", exist_ok=True)
    np.save(vec_path, vectors)
    with open(rec_path, "w") as f:
        json.dump(records, f)
        
    return records, vectors

if __name__ == "__main__":
    records, vectors = embed_chunks()
    print(vectors.shape)  # (num_chunks, 1024)
