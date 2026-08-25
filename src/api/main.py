from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import ChatRequest, ChatResponse
from src.agents.graph import run_agent
import logging

logging.basicConfig(filename="logs/app.log", level=logging.INFO)
app = FastAPI(title="Business Assistant API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        answer = run_agent(req.query)
        return ChatResponse(answer=answer)
    except Exception as e:
        logging.exception("chat failed")
        return ChatResponse(answer=f"Error: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}
