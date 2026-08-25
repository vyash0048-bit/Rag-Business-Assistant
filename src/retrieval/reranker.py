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
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    
    scored_candidates = []
    for c, s in zip(candidates, scores):
        score = float(s)
        if score >= min_score:
            c_copy = copy.deepcopy(c)
            c_copy["rerank_score"] = score
            scored_candidates.append(c_copy)
            
    return sorted(scored_candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
