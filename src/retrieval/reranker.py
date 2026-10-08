import copy
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker

def rerank(query, candidates, top_k=7, min_score=-2.0):
    if not candidates:
        return []
        
    model = get_reranker()
    pairs = [[query, doc["text"]] for doc in candidates]
    scores = model.predict(pairs)
    
    scored_candidates = []
    for doc, score in zip(candidates, scores):
        doc_copy = copy.deepcopy(doc)
        doc_copy["rerank_score"] = float(score)
        if score >= min_score:
            scored_candidates.append(doc_copy)
            
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_candidates[:top_k]
