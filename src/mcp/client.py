import asyncio
import sys
import os
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Use the venv python path
PYTHON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "venv", "Scripts", "python.exe")

async def call_mcp_tool(session: ClientSession, tool_name: str, args: dict) -> str:
    """Call an MCP tool on an active session."""
    result = await session.call_tool(tool_name, args)
    return result.content[0].text

async def main():
    server_params = StdioServerParameters(
        command=PYTHON_PATH,
        args=["-m", "src.mcp.server"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List registered tools
            tools = await session.list_tools()
            print(f"Registered tools: {[t.name for t in tools.tools]}\n")

            # List resources
            resources = await session.list_resources()
            print(f"Registered resources: {[r.name for r in resources.resources]}\n")

            # Call sql_financials
            print("--- Calling sql_financials ---")
            result = await call_mcp_tool(session, "sql_financials", {
                "query": "SELECT company, fiscal_year, metric, value FROM financials WHERE company='HMC' AND metric='Revenues' AND fiscal_year >= 2022 ORDER BY fiscal_year LIMIT 5"
            })
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
