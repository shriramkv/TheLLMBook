"""Section 9.3: start the purchase-order MCP server and call its tool, as a host would.
    python sec_9_3_mcp_client_test.py
"""
import asyncio, sys
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    server = StdioServerParameters(command=sys.executable, args=["sec_9_3_mcp_po_server.py"])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for t in tools.tools:
                print("Tool:", t.name, "-", t.description.splitlines()[0])
                print("Schema:", getattr(t, "input_schema", None) or getattr(t, "inputSchema", None))   # v2 / v1 name
            for po in ("PO-7003", "PO-7099"):
                result = await session.call_tool("get_purchase_order", {"po_number": po})
                print(f"\nget_purchase_order({po}) ->", result.content[0].text)

asyncio.run(main())
