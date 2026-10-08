from src.vectorstore.qdrant_store import client, COLLECTION
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _embed_model

def retrieve(query, top_k=30, score_threshold=0.3, company_filter=None, doc_type=None, fiscal_year=None):
    model = get_embed_model()
    qvec = model.encode("Represent this query for retrieval: " + query, normalize_embeddings=True).tolist()
    
    conditions = []
    if company_filter:
        conditions.append(FieldCondition(key="company", match=MatchValue(value=company_filter)))
    if doc_type:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if fiscal_year:
        conditions.append(FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year)))
        
    flt = Filter(must=conditions) if conditions else None
    
    response = client.query_points(
        collection_name=COLLECTION, 
        query=qvec, 
        limit=top_k, 
        query_filter=flt,
        score_threshold=score_threshold
    )
    
    return [{"text": h.payload["text"], "source": h.payload["source"],
              "page": h.payload["page"], "company": h.payload["company"], 
              "doc_type": h.payload.get("doc_type", ""), "fiscal_year": h.payload.get("fiscal_year", ""),
              "score": h.score} for h in response.points]
