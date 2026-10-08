import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """You are an expert financial and business analyst. 
Answer the user's question using ONLY the context provided below.

Guidelines:
1. Cite EVERY claim using [Source N] references.
2. Structure your answer with clear sections when appropriate.
3. If the context does not contain enough information to answer the question, state "I don't have enough information" rather than hallucinating.
4. Cross-reference and synthesize information across multiple sources when possible.
5. Handle numerical data and financial figures with high precision.

Context:
{context}

Question: {query}"""

def answer(query, company_filter=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    candidates = retrieve(query, company_filter=company_filter, top_k=5)
    if not candidates:
        return {"answer": "I don't have enough information. No context was retrieved.", "sources": []}
        
    top = rerank(query, candidates, top_k=3)
    if not top:
        return {"answer": "I don't have enough information. No relevant context was found after reranking.", "sources": []}
        
    context = "\n\n".join(
        f"[Source {i+1}: {c['source']} p.{c['page']}] {c['text']}" for i, c in enumerate(top)
    )
    prompt = system_prompt.format(context=context, query=query)
    
    from langchain_groq import ChatGroq
    
    from tenacity import retry, wait_exponential, stop_after_attempt
    @retry(wait=wait_exponential(min=5, max=30), stop=stop_after_attempt(5))
    def _do_call():
        llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=os.environ["GROQ_API_KEY"],
            max_tokens=250
        )
        return llm.invoke(prompt)
    
    interaction = _do_call()
    
    return {"answer": interaction.content, "sources": top}

if __name__ == "__main__":
    import json
    res = answer("What did Honda say about supply chain risk?", company_filter="HMC")
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(f"- {s['source']} p.{s['page']} [Type: {s.get('doc_type', 'N/A')}, FY: {s.get('fiscal_year', 'N/A')}] (Score: {s['rerank_score']:.4f})")
