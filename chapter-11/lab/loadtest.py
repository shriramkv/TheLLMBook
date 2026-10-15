"""Lab 11.1: load-test the service at rising concurrency and report percentiles.
    python loadtest.py --url http://localhost:8080 --requests 96
    python loadtest.py --repeat 0.5      # half the requests repeat an earlier question (cache hits)
"""
import argparse, asyncio, json, random, statistics, time
import httpx
p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8080"); p.add_argument("--requests", type=int, default=96)
p.add_argument("--levels", default="1,4,8,16,32"); p.add_argument("--repeat", type=float, default=0.0)
p.add_argument("--out", default="loadtest_results.json"); p.add_argument("--offset", type=int, default=0)
a = p.parse_args()
SYSTEM = "You are a support assistant for a travel company. " * 30          # a long, stable system prompt
def question(i): return f"Question {i}: can I change the date on booking TRV{100000 + i} and what will it cost?"
def pct(xs, q): xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]
async def run(level, rng):
    lat, sem, seen = [], asyncio.Semaphore(level), []
    async with httpx.AsyncClient(base_url=a.url, timeout=60) as c:
        async def one(i):
            n = rng.choice(seen) if seen and rng.random() < a.repeat else i; seen.append(n)
            body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question(n)}],
                    "max_tokens": 120, "user": f"user-{n % 50}"}
            async with sem:
                t = time.perf_counter(); r = await c.post("/v1/chat/completions", json=body); r.raise_for_status()
                lat.append(time.perf_counter() - t)
        t0 = time.perf_counter(); await asyncio.gather(*(one(a.offset + level * 10_000 + i) for i in range(a.requests)))
        wall = time.perf_counter() - t0
    return {"concurrency": level, "p50_s": round(statistics.median(lat), 2), "p95_s": round(pct(lat, 0.95), 2),
            "throughput_rps": round(a.requests / wall, 2)}
async def main():
    rng, results = random.Random(0), []
    for lv in [int(x) for x in a.levels.split(",")]:
        results.append(await run(lv, rng)); print(results[-1], flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
asyncio.run(main())
