# Chapter 11: LLMOps, Deploying GenAI to Production

Five experiments on what it takes to run a model in production: caching, quantisation, serving, cost, and a release gate.

## Files

- `semantic_cache_experiment.py` + `cache_pairs.json` — shows why a naive semantic cache is dangerous: no similarity threshold cleanly separates genuine paraphrase hits from wrong near-misses. Needs `wordllama`.
- `quantise_minigpt.py` — quantises the bundled MiniGPT to 8, 4, 3 and 2 bits and measures the loss. Needs `torch`; uses `mini_gpt.py` and `minigpt.pt` (and `input.txt` as the corpus).
- `cost_model.py` — an illustrative per-conversation cost waterfall (caching, routing, batching). Standard library only.
- `lab/llm_service.py` — a FastAPI, OpenAI-compatible LLM service with a stub backend, a cache, a rate limiter, Prometheus metrics and structured logging. `lab/loadtest.py` drives it; `lab/config.yaml` configures it.
- `eval_gate.py` — a CI-style release gate: task quality, regression and honest-refusal thresholds that must pass before shipping. Exits non-zero to block a release. Standard library only.

## Run

```bash
python3 semantic_cache_experiment.py
python3 quantise_minigpt.py
python3 cost_model.py
python3 eval_gate.py            # exits 0 to ship, non-zero to block

# the lab service
pip install fastapi uvicorn pyyaml httpx
uvicorn lab.llm_service:app --port 8000   # in one terminal
python3 lab/loadtest.py                    # in another
```

`quant_results.json`, `cache_results.json` and `lab/loadtest_results.json` hold the book's runs for comparison.
