from langchain_core.tools import tool
from src.structured.sql_tool import sql_query
from src.retrieval.rag_answer import answer as rag_answer

@tool
def doc_search(query: str, company: str = None) -> str:
    """Search company filings (10-Ks) for narrative info like risks, strategy, business description."""
    import time
    print(f"[{time.strftime('%X')}] Tool doc_search called with query={query}, company={company}")
    r = rag_answer(query, company_filter=company)
    print(f"[{time.strftime('%X')}] Tool doc_search completed")
    return r["answer"]

@tool
def structured_query(sql: str) -> str:
    """Run a SELECT query against tables: financials(company, fiscal_year, metric, value, unit),
    recalls(company, model, model_year, recall_date, component, consequence)."""
    return sql_query(sql)

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. growth rate calculations."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"
