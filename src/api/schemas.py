from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    company: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    tool_trace: list[str] = []
