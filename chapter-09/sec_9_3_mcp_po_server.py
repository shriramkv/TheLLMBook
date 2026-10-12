"""Section 9.3: a small MCP server that exposes the purchase-order lookup as a tool.

    pip install "mcp>=2"
    python sec_9_3_mcp_po_server.py            # waits silently on stdin: an MCP host starts it
    python sec_9_3_mcp_client_test.py          # starts this server and calls the tool

To use it from an MCP host such as Claude Desktop or an IDE, add a server entry whose
command is `python` and whose argument is the full path to this file.

This uses version 2 of the Python SDK, where the class is MCPServer. In version 1 it was
FastMCP, imported with `from mcp.server.fastmcp import FastMCP`; the rest is the same.
"""
from mcp.server.mcpserver import MCPServer      # Python SDK v2; v1 used FastMCP
from invoice_workflow import POS                # stands in for your ERP, read-only

class ERP:
    @staticmethod
    def lookup(po_number):
        po = POS.get(po_number)
        if not po:
            return None
        return {"po_number": po_number, "vendor": po["vendor"],
                "lines": [{"item": k, "quantity": q, "unit_price": p} for k, (q, p) in po["lines"].items()]}

mcp = MCPServer("purchase-orders")

@mcp.tool()
def get_purchase_order(po_number: str) -> dict:
    """Return the vendor and ordered lines (quantity, unit price) for a purchase order.
    PO numbers look like PO-7001. Returns an error message if the PO does not exist."""
    po = ERP.lookup(po_number)                  # your existing system, read-only credentials
    return po if po else {"error": f"{po_number} not found"}

if __name__ == "__main__":
    mcp.run(transport="stdio")                  # local; use streamable HTTP for remote clients
