"""Section 9.2: a minimal agent loop with tool calling.

The model only asks for tools; this code runs them. Two read-only tools from the lab's
purchasing system are offered: get_po and get_grn.

    python sec_9_2_agent_loop.py                       # offline: a scripted stand-in model
    export OPENAI_API_KEY=...                          # or any OpenAI-compatible server (Ch 8)
    python sec_9_2_agent_loop.py gpt-4o-mini
    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=x python sec_9_2_agent_loop.py llama3.1:8b

Use a model that supports tool calling.
"""
import json, sys
from types import SimpleNamespace as NS
from invoice_workflow import POS, GRNS

def run_agent(client, model, messages, tools, registry, max_steps=8):
    """Minimal agent loop: call the model, run requested tools, repeat until it answers."""
    for step in range(max_steps):
        reply = client.chat.completions.create(model=model, messages=messages, tools=tools)
        msg = reply.choices[0].message
        if not msg.tool_calls:                               # no tool requested: this is the answer
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            args = call.function.arguments
            try:
                args = json.loads(call.function.arguments)
                result = registry[call.function.name](**args)  # your code runs the tool
            except Exception as e:                           # errors go back to the model, not up the stack
                result = {"error": f"{type(e).__name__}: {e}"}
            print(f"  step {step + 1}: {call.function.name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})
    raise RuntimeError("step budget exhausted without an answer")

# ---- the tools: one clear job each, constrained arguments, helpful errors
def get_po(po_number: str):
    po = POS.get(po_number)
    if not po:
        return {"error": f"{po_number} not found. PO numbers look like PO-7001."}
    return {"vendor": po["vendor"],
            "lines": [{"item": k, "quantity": q, "unit_price": p} for k, (q, p) in po["lines"].items()]}

def get_grn(po_number: str):
    grn = GRNS.get(po_number)
    return {"received": grn} if grn else {"error": f"no goods received note for {po_number}"}

registry = {"get_po": get_po, "get_grn": get_grn}
po_param = {"type": "object", "required": ["po_number"],
            "properties": {"po_number": {"type": "string", "pattern": "^PO-\\d{4}$",
                                         "description": "Purchase order number, e.g. PO-7003"}}}
tools = [
    {"type": "function", "function": {"name": "get_po", "parameters": po_param,
     "description": "Look up a purchase order: vendor id and ordered lines with quantity and unit price. Read-only."}},
    {"type": "function", "function": {"name": "get_grn", "parameters": po_param,
     "description": "Look up the goods received note for a purchase order: quantity actually received per item. Read-only."}},
]

class ScriptedClient:
    """Offline stand-in for an OpenAI client. It plays back a fixed plan (look up the PO,
    then the GRN, then answer) so that the loop can be followed without a model."""
    def __init__(self):
        self.chat = NS(completions=NS(create=self.create))
    def create(self, model, messages, tools):
        n_tool_results = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        def ask(name):
            call = NS(id=f"call_{n_tool_results}", type="function",
                      function=NS(name=name, arguments=json.dumps({"po_number": "PO-7003"})))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call], role="assistant"))])
        if n_tool_results == 0:
            return ask("get_po")
        if n_tool_results == 1:
            return ask("get_grn")
        po, grn = (json.loads(m["content"]) for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        ordered = po["lines"][0]["quantity"]; received = list(grn["received"].values())[0]
        text = (f"PO-7003 ordered {ordered} door seals; {received} have been received, "
                f"so an invoice for {ordered} should be held until the remaining {ordered - received} arrive.")
        return NS(choices=[NS(message=NS(content=text, tool_calls=None, role="assistant"))])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from openai import OpenAI
        client, model = OpenAI(), sys.argv[1]
    else:
        client, model = ScriptedClient(), "scripted"
        print("Offline: a scripted stand-in plays the model.\n")
    messages = [{"role": "system", "content": "You are an accounts-payable assistant. Use the tools; never guess figures."},
                {"role": "user", "content": "Marina Rubber Works has invoiced 1,500 front door seals against PO-7003. "
                                            "Has everything been received?"}]
    print("Answer:", run_agent(client, model, messages, tools, registry))
