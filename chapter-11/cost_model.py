"""Chapter 11, Section 11.4: an illustrative cost model for one support conversation.
Prices are illustrative (US$ per million tokens), not quotes."""
BIG_IN, BIG_OUT, SMALL_IN, SMALL_OUT, CACHE_READ = 1.00, 5.00, 0.10, 0.40, 0.10   # cache read = 10% of input price
# four model calls per conversation: (name, stable prefix tokens, retrieved tokens, other input tokens, output tokens, uses retrieval)
calls = [("route the request", 1200, 0, 600, 20), ("answer with retrieval", 1200, 1500, 600, 250),
         ("decide on a tool call", 1200, 1500, 600, 60), ("summarise for the ticket", 1200, 0, 1200, 150)]
def cost(cache_hit=0.0, small_for=(), retrieved=1500, out_scale=1.0):
    total = 0.0
    for name, prefix, ret, other, out in calls:
        pin, pout = (SMALL_IN, SMALL_OUT) if name in small_for else (BIG_IN, BIG_OUT)
        r = retrieved if ret else 0
        o = out * (out_scale if name == "answer with retrieval" else 1)
        cached = prefix * cache_hit
        total += ((prefix - cached) + r + other) * pin / 1e6 + cached * pin * CACHE_READ / 1e6 + o * pout / 1e6
    return total * 1000   # per 1,000 conversations
steps = [("Baseline: four calls on a mid-sized model", cost())]
steps.append(("Cache the 1,200-token stable prefix (90% hits)", cost(0.9)))
steps.append(("Route and summarise with a small model", cost(0.9, ("route the request", "summarise for the ticket"))))
steps.append(("Rerank and send 800 instead of 1,500 retrieved tokens", cost(0.9, ("route the request", "summarise for the ticket"), 800)))
steps.append(("Ask for shorter answers (250 to 180 tokens)", cost(0.9, ("route the request", "summarise for the ticket"), 800, 0.72)))
prev = None
for s, c in steps:
    print(f"{s:58s} ${c:6.2f} per 1,000" + ("" if prev is None else f"   change ${c - prev:+.2f}")); prev = c
print("reduction %.0f%%" % (100 * (1 - steps[-1][1] / steps[0][1])))
print("weekly cost at 30,000 conversations: $%.0f -> $%.0f" % (steps[0][1] * 30, steps[-1][1] * 30))
