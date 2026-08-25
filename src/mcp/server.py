from mcp.server import MCPServer
from src.structured.sql_tool import sql_query
from src.retrieval.rag_answer import answer as rag_answer

mcp = MCPServer("business-intelligence")

@mcp.tool()
def sql_financials(query: str) -> str:
    """Run a read-only SELECT against financials/recalls tables."""
    return sql_query(query)

@mcp.tool()
def document_search(query: str, company: str = "") -> str:
    """Search 10-K filings for narrative/qualitative answers with citations."""
    r = rag_answer(query, company_filter=company or None)
    return r["answer"]

@mcp.tool()
def generate_report(company: str, topic: str) -> str:
    """Generate a short structured business report combining financial data and filing text."""
    fin = sql_query(f"SELECT * FROM financials WHERE company='{company}' ORDER BY fiscal_year")
    narrative = rag_answer(f"{topic} for {company}", company_filter=company)["answer"]
    return f"## {company} Report: {topic}\n\n### Financial Data\n{fin}\n\n### Analysis\n{narrative}"

@mcp.resource("business://companies")
def list_companies() -> str:
    return "HMC (Honda), TM (Toyota), F (Ford)"

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or transport="sse" for network clients
