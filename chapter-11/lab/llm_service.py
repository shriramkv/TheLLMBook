"""Lab 11.1 (Section 11.8): an LLM service you can deploy and monitor.

An OpenAI-compatible endpoint in front of any model backend, with the production basics of Chapter 11:
pinned configuration, a timeout, a per-user rate limit, an exact-match response cache, one structured
log record per call (Table 11.4) and Prometheus metrics (Section 11.5).

    pip install fastapi uvicorn prometheus_client pyyaml openai
    uvicorn llm_service:app --port 8080          # backend: stub runs fully offline
    curl localhost:8080/metrics                  # Prometheus metrics

With `backend: openai`, set OPENAI_BASE_URL and OPENAI_API_KEY to use a commercial API, Ollama or vLLM.
"""
import asyncio, hashlib, json, logging, os, time, uuid
from collections import OrderedDict, defaultdict, deque
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

CFG = yaml.safe_load(open(os.getenv("LLM_SERVICE_CONFIG", "config.yaml")))
log = logging.getLogger("llm"); logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG_FILE = open(os.getenv("LLM_LOG_FILE", "calls.jsonl"), "a")

# ---- metrics (Section 11.5)
REQS = Counter("llm_requests_total", "Requests by outcome", ["outcome"])
LAT = Histogram("llm_request_seconds", "End-to-end latency", buckets=[.25, .5, 1, 2, 4, 8, 16, 32])
TTFT = Histogram("llm_ttft_seconds", "Time to first token", buckets=[.1, .25, .5, 1, 2, 4, 8, 16])
TOKENS = Counter("llm_tokens_total", "Tokens", ["direction"])
COST = Counter("llm_cost_usd_total", "Model cost in US dollars (illustrative prices)")
CACHE = Counter("llm_cache_total", "Response cache lookups", ["result"])
INFLIGHT = Gauge("llm_inflight_requests", "Requests being served")

def n_tokens(text):               # rough count; use the model's tokeniser in production (Section 4.1)
    return max(1, len(text) // 4)

# ---- backends
SLOTS = asyncio.Semaphore(CFG.get("upstream_slots", 8))

async def stub_backend(messages, max_tokens):
    """A simulated GPU: requests wait for one of N decode slots, then pay a prefill cost that grows with
    the prompt and a decode cost per output token. No model is called; the answer text is a placeholder."""
    t0 = time.perf_counter()
    prompt = " ".join(m["content"] for m in messages); n_in = n_tokens(prompt)
    async with SLOTS:
        await asyncio.sleep(0.25 + 0.0002 * n_in)                 # prefill
        ttft = time.perf_counter() - t0
        n_out = min(max_tokens, 120)
        await asyncio.sleep(0.02 * n_out)                         # decode at 50 tokens per second
    text = "Simulated answer to: " + messages[-1]["content"][:80]
    return {"text": text, "input_tokens": n_in, "output_tokens": n_out, "ttft_s": ttft}

async def openai_backend(messages, max_tokens):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    t0, first, parts, usage = time.perf_counter(), None, [], None
    stream = await client.chat.completions.create(model=CFG["model"], messages=messages, max_tokens=max_tokens,
                                                  stream=True, stream_options={"include_usage": True})
    async for ev in stream:
        if ev.choices and ev.choices[0].delta.content:
            first = first or time.perf_counter(); parts.append(ev.choices[0].delta.content)
        if ev.usage: usage = ev.usage
    end = time.perf_counter()
    return {"text": "".join(parts), "ttft_s": (first or end) - t0,
            "input_tokens": usage.prompt_tokens if usage else None, "output_tokens": usage.completion_tokens if usage else None}

BACKEND = stub_backend if CFG["backend"] == "stub" else openai_backend

# ---- exact-match cache and rate limiter
class TTLCache:
    def __init__(self, ttl, size=10_000): self.ttl, self.size, self.d = ttl, size, OrderedDict()
    def get(self, k):
        v = self.d.get(k)
        if not v or time.time() - v[0] > self.ttl: self.d.pop(k, None); return None
        self.d.move_to_end(k); return v[1]
    def put(self, k, val):
        self.d[k] = (time.time(), val); self.d.move_to_end(k)
        while len(self.d) > self.size: self.d.popitem(last=False)
CACHE_STORE = TTLCache(CFG.get("cache_ttl_s", 0))
HITS = defaultdict(deque)
def rate_limited(user):
    q, now = HITS[user], time.time()
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= CFG.get("rate_limit_per_minute", 120): return True
    q.append(now); return False

app = FastAPI(title="LLM service, Lab 11.1")

@app.get("/healthz")
def healthz(): return {"status": "ok", "model": CFG["model"], "prompt_version": CFG["prompt_version"]}

@app.get("/metrics")
def metrics(): return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages, max_tokens = body["messages"], int(body.get("max_tokens", 300))
    user = hashlib.sha256(str(body.get("user", "anonymous")).encode()).hexdigest()[:16]   # never log raw user IDs
    req_id, t0 = str(uuid.uuid4()), time.perf_counter()
    if rate_limited(user):
        REQS.labels("rate_limited").inc(); raise HTTPException(429, "Too many requests, please slow down.")
    key = hashlib.sha256(json.dumps([CFG["model"], CFG["prompt_version"], messages], sort_keys=True).encode()).hexdigest()
    cached = CACHE_STORE.get(key) if CFG.get("cache_ttl_s") else None
    CACHE.labels("hit" if cached else "miss").inc()
    INFLIGHT.inc()
    try:
        if cached:
            r, outcome = dict(cached, ttft_s=0.0), "cache_hit"
        else:
            r = await asyncio.wait_for(BACKEND(messages, max_tokens), timeout=CFG.get("timeout_s", 30))
            outcome = "ok"
            if CFG.get("cache_ttl_s"): CACHE_STORE.put(key, r)
    except asyncio.TimeoutError:
        REQS.labels("timeout").inc(); raise HTTPException(504, "The model took too long to answer.")
    finally:
        INFLIGHT.dec()
    total = time.perf_counter() - t0
    cost = 0.0 if outcome == "cache_hit" else ((r["input_tokens"] or 0) * CFG["price_in_per_million"] +
                                               (r["output_tokens"] or 0) * CFG["price_out_per_million"]) / 1e6
    REQS.labels(outcome).inc(); LAT.observe(total); TTFT.observe(r["ttft_s"])
    if outcome == "ok":
        TOKENS.labels("input").inc(r["input_tokens"] or 0); TOKENS.labels("output").inc(r["output_tokens"] or 0); COST.inc(cost)
    record = {"ts": time.time(), "request_id": req_id, "user": user, "feature": body.get("metadata", {}).get("feature", "chat"),
              "model": CFG["model"], "prompt_version": CFG["prompt_version"], "outcome": outcome,
              "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "cost_usd": round(cost, 6),
              "ttft_s": round(r["ttft_s"], 3), "latency_s": round(total, 3)}
    LOG_FILE.write(json.dumps(record) + "\n"); LOG_FILE.flush()
    return JSONResponse({"id": req_id, "object": "chat.completion", "model": CFG["model"],
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": r["input_tokens"], "completion_tokens": r["output_tokens"]}})
