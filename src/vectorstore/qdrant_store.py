# src/vectorstore/qdrant_store.py
import os
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from src.embeddings.embed import embed_chunks

from dotenv import load_dotenv
load_dotenv()

url = os.environ.get("QDRANT_URL")
if url:
    client = QdrantClient(url=url)
else:
    client = QdrantClient(path="qdrant_storage")

COLLECTION = "business_docs"

def build_index():
    records, vectors = embed_chunks()
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    
    client.create_payload_index(collection_name=COLLECTION, field_name="company", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name=COLLECTION, field_name="doc_type", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name=COLLECTION, field_name="fiscal_year", field_schema=PayloadSchemaType.KEYWORD)

    namespace = uuid.UUID('12345678-1234-5678-1234-567812345678')
    points = [
        PointStruct(
            id=str(uuid.uuid5(namespace, str(records[i].get("chunk_id", i)))), 
            vector=vectors[i].tolist(), 
            payload=records[i]
        )
        for i in range(len(records))
    ]
    client.upload_points(collection_name=COLLECTION, points=points, batch_size=64)
    print(f"Indexed {len(points)} chunks")
    
    # Verify: client.count(COLLECTION) returns chunk count matching chunks.jsonl line count.
    print(f"Count from DB: {client.count(COLLECTION).count}")

if __name__ == "__main__":
    build_index()
