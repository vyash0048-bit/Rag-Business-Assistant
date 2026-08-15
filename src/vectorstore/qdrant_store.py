from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os, json
from src.embeddings.embed import embed_chunks

# We are using local storage instead of Docker because the Docker daemon is not running.
client = QdrantClient(path="qdrant_storage")

COLLECTION = "business_docs"

def build_index():
    records, vectors = embed_chunks()
    
    # Check if collection exists before attempting to create
    if not client.collection_exists(collection_name=COLLECTION):
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    else:
        # We optionally recreate to start fresh on index build
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload=records[i])
        for i in range(len(records))
    ]
    client.upload_points(collection_name=COLLECTION, points=points, batch_size=64)
    print(f"Indexed {len(points)} chunks")
    
    # Verify count
    count = client.count(collection_name=COLLECTION).count
    print(f"Collection {COLLECTION} has {count} items.")

if __name__ == "__main__":
    build_index()
