import os
import re
from dotenv import load_dotenv
from google import genai
from typing import List, Dict

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(wait=wait_exponential(min=15, max=120), stop=stop_after_attempt(10))
def _call_llm(prompt: str) -> str:
    interaction = client.interactions.create(
        model='gemini-3.7-flash',
        input=prompt
    )
    return interaction.output_text

def _extract_score(text: str) -> float:
    match = re.search(r'SCORE:\s*([0-1](?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    return 0.0

def evaluate_context_precision(question: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
    
    prompt = f"""Given the question, assess whether the provided context chunks are relevant.
For each chunk, determine if it contains useful information to answer the question.
Output a final score between 0.0 and 1.0 representing the proportion of relevant chunks.

Question: {question}

Contexts:
"""
    for i, ctx in enumerate(contexts):
        prompt += f"Chunk {i+1}:\n{ctx}\n\n"
        
    prompt += "End your response with 'SCORE: <value>' (e.g. SCORE: 0.75)"
    
    result = _call_llm(prompt)
    return _extract_score(result)

def evaluate_context_recall(question: str, ground_truth: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
        
    combined_context = "\n".join(contexts)
    prompt = f"""Given the expected answer (ground truth) and the retrieved context, evaluate if the context contains enough information to fully cover the ground truth.
Output a score between 0.0 and 1.0 representing the extent to which the context covers the ground truth (1.0 = fully covered, 0.5 = partially covered, 0.0 = not covered).

Ground Truth: {ground_truth}

Retrieved Context:
{combined_context}

End your response with 'SCORE: <value>' (e.g. SCORE: 0.8)"""

    result = _call_llm(prompt)
    return _extract_score(result)

def evaluate_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""Evaluate how relevant the provided answer is to the given question. Does the answer directly address the question without providing unnecessary or unrelated information?
Output a score between 0.0 and 1.0 (1.0 = highly relevant and direct, 0.0 = completely irrelevant).

Question: {question}
Answer: {answer}

End your response with 'SCORE: <value>' (e.g. SCORE: 1.0)"""

    result = _call_llm(prompt)
    return _extract_score(result)

def evaluate_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
        
    combined_context = "\n".join(contexts)
    prompt = f"""Evaluate if the provided answer is faithful to the retrieved context. Are all the claims made in the answer supported by the context, or does it hallucinate information not present in the context?
Output a score between 0.0 and 1.0 (1.0 = fully faithful, 0.0 = completely hallucinates or ignores context).

Context:
{combined_context}

Answer: {answer}

End your response with 'SCORE: <value>' (e.g. SCORE: 1.0)"""

    result = _call_llm(prompt)
    return _extract_score(result)

def evaluate_single(question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    return {
        "context_precision": evaluate_context_precision(question, contexts),
        "context_recall": evaluate_context_recall(question, ground_truth, contexts),
        "answer_relevancy": evaluate_answer_relevancy(question, answer),
        "faithfulness": evaluate_faithfulness(question, answer, contexts)
    }

def evaluate_batch(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {"context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}
        
    agg = {"context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}
    for res in results:
        scores = res.get("metrics", {})
        agg["context_precision"] += scores.get("context_precision", 0.0)
        agg["context_recall"] += scores.get("context_recall", 0.0)
        agg["answer_relevancy"] += scores.get("answer_relevancy", 0.0)
        agg["faithfulness"] += scores.get("faithfulness", 0.0)
        
    n = len(results)
    return {k: v / n for k, v in agg.items()}
