"""Lab 12.1 (Section 12.7): a courier assistant you attack and then defend. All data is fictional.

Two builds share one set of tools:
  VulnerableCourier  - trusts the model: tools act on whatever the model asks, rules live only in the prompt
  HardenedCourier    - the SAME assistant with defence in depth: untrusted text is labelled, permissions and
                       refund eligibility are enforced in the tools, and outputs are checked

A pluggable `model` turns a user message into a list of tool requests. The lab ships an OBEDIENT stub model
(it does whatever an instruction in the text says) so the whole lab runs offline and shows the worst case;
swap in a real model from Chapter 8 for the live exercises.
"""
from dataclasses import dataclass, field

# ---- fictional data. The current user owns only their own shipments.
SHIPMENTS = {
 "KOL-1001": {"owner": "u-current", "address": "12 Park Street, Kolkata", "phone": "90000-11111", "status": "delivered"},
 "KOL-1007": {"owner": "u-current", "address": "5 Salt Lake, Kolkata", "phone": "90000-22222", "status": "in transit"},
 "KOL-2002": {"owner": "u-other", "address": "88 Ballygunge, Kolkata", "phone": "90000-33333", "status": "in transit"},
}
SYSTEM_PROMPT_SECRET = "Internal rule: staff discount code COURIER20 is never shared with customers."
PAID = set()

@dataclass
class Case:
    user: str
    owned: set
    actions: list = field(default_factory=list)
    leaked: list = field(default_factory=list)

# ---- tools, in two versions
class VulnerableCourier:
    name = "vulnerable"
    def get_shipment(self, case, sid):
        s = SHIPMENTS.get(sid)
        if not s: return {"error": "not found"}
        return {"shipment": sid, **s}                       # no ownership check
    def raise_refund(self, case, sid):
        case.actions.append(("refund", sid)); return {"refund_raised": sid}   # no eligibility check
    def reveal_prompt(self, case):
        case.leaked.append("system_prompt"); return {"system_prompt": SYSTEM_PROMPT_SECRET}

class HardenedCourier:
    name = "hardened"
    def get_shipment(self, case, sid):
        s = SHIPMENTS.get(sid)
        if not s: return {"error": "not found"}
        if s["owner"] != case.user:                          # permission check in code
            return {"error": "you can only view your own shipments"}
        return {"shipment": sid, "status": s["status"]}      # minimisation: no address/phone unless needed
    def raise_refund(self, case, sid):
        s = SHIPMENTS.get(sid)
        if not s or s["owner"] != case.user: return {"error": "not your shipment"}
        if s["status"] == "delivered":     # eligibility check in code
            return {"error": "delivered parcels are not auto-refunded; a human must review"}
        case.actions.append(("refund", sid)); return {"refund_raised": sid}
    def reveal_prompt(self, case):
        return {"error": "no such capability"}               # the prompt secret is not reachable by any tool
