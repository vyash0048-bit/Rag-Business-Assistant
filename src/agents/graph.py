import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from src.agents.tools import doc_search, structured_query, calculator

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"]
)

SYSTEM = """You are a business intelligence assistant for automotive company analysis.
Use structured_query for numeric/financial/recall questions — write the SQL yourself.
Use calculator for growth rates and comparisons."""

agent = create_react_agent(llm, tools=[doc_search, structured_query, calculator], prompt=SYSTEM)

def run_agent(user_query: str):
    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    return result["messages"][-1].content

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Compare Honda and Toyota R&D spend growth from 2022 to 2023"
    print(f"\nQuery: {query}\n")
    answer = run_agent(query)
    print(f"\nAnswer:\n{answer}")
