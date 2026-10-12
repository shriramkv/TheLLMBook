"""Section 9.4 and Lab 9.1, exercise 4: the accounts-payable workflow as a LangGraph graph,
with a human approval step that pauses the graph and resumes it later.

    pip install langgraph
    python sec_9_4_langgraph_workflow.py

The nodes reuse the agents from invoice_workflow.py unchanged. Escalated invoices stop at
human_approval; the script then plays the approver, approving KC-2291 and rejecting the
rest, and resumes each paused graph from its checkpoint.
"""
from dataclasses import asdict
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
import invoice_workflow as wf

class APState(TypedDict, total=False):
    raw: str
    invoice: dict
    findings: list
    decision: str
    trace: list

def to_case(state):
    """Rebuild the lab's Case object from the graph state, so the agents run unchanged."""
    case = wf.Case(state["raw"], findings=list(state.get("findings", [])), trace=list(state.get("trace", [])),
                   decision=state.get("decision", ""))
    if state.get("invoice"):
        case.invoice = wf.Invoice(**state["invoice"])
    return case

def node(agent):
    def run(state: APState):
        case = to_case(state)
        agent(case)
        return {"invoice": asdict(case.invoice), "findings": case.findings,
                "decision": case.decision, "trace": case.trace}
    return run

intake, match, comply, resolve = (node(a) for a in (wf.intake_agent, wf.matching_agent,
                                                     wf.compliance_agent, wf.resolution_agent))

def human_approval(state: APState):
    answer = interrupt({"invoice": state["invoice"]["number"], "findings": state["findings"]})
    return {"decision": "APPROVE" if answer == "approve" else "REJECT",      # resumes here later
            "trace": state["trace"] + [f"{'human':11s} answered '{answer}'"]}

graph = StateGraph(APState)
for name, fn in [("intake", intake), ("match", match), ("comply", comply),
                 ("resolve", resolve), ("human_approval", human_approval)]:
    graph.add_node(name, fn)
graph.add_edge(START, "intake")
graph.add_edge("intake", "match")
graph.add_conditional_edges("match",                     # a blocking finding skips compliance
    lambda s: "resolve" if any(f[0] == "block" for f in s["findings"]) else "comply")
graph.add_edge("comply", "resolve")
graph.add_conditional_edges("resolve",
    lambda s: "human_approval" if s["decision"] == "ESCALATE TO HUMAN" else END)
graph.add_edge("human_approval", END)
app = graph.compile(checkpointer=InMemorySaver())       # use a database checkpointer in production

def pay(state):
    """The payment guard from the lab, applied to the graph's final state."""
    case = to_case(state)
    return wf.schedule_payment(case)

if __name__ == "__main__":
    paused = []
    for i, raw in enumerate(wf.INVOICES):
        config = {"configurable": {"thread_id": f"invoice-{i}"}}
        state = app.invoke({"raw": raw}, config)
        number = state["invoice"]["number"]
        if "__interrupt__" in state:
            paused.append((number, config))
            print(f"{number:13s} PAUSED for a person: {state['__interrupt__'][0].value['findings']}")
        else:
            print(f"{number:13s} {state['decision']:22s} paid: {pay(state)}")

    print(f"\n{len(paused)} invoices are waiting. Their state is checkpointed, so a person can")
    print("answer hours later and each graph resumes exactly where it stopped.\n")
    for number, config in paused:
        answer = "approve" if number == "KC-2291" else "reject"
        state = app.invoke(Command(resume=answer), config)
        print(f"{number:13s} person said '{answer}' -> {state['decision']:8s} paid: {pay(state)}")

    print("\nFull trace for KC-2291:")
    print("\n".join("   " + t for t in app.get_state(paused[0][1]).values["trace"]))
