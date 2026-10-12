"""One interface for the four deployment modes of Lab 8.1 (Section 8.6), plus the
retry wrapper from Section 8.3.

Every backend is a function call(messages, max_tokens=300, temperature=0) that returns
{"text", "ttft_s", "total_s", "input_tokens", "output_tokens"}.

Fill in the model names below, or set them as environment variables, before running:
    API_MODEL       a current mid-sized model from your provider (uses OPENAI_API_KEY,
                    and OPENAI_BASE_URL if your provider offers an OpenAI-compatible endpoint)
    OLLAMA_MODEL    for example a 4-bit 8B model you have pulled with `ollama pull`
    VLLM_HOST       the address of your GPU server, for example 203.0.113.10
    VLLM_MODEL      the Hugging Face id of the same model as Ollama, in 16-bit
    LOCAL_HF_MODEL  a small instruction-tuned model for Mode 2, about 1 to 3B parameters
"""
import os, random, time

ENDPOINTS = {
    "api":    dict(base_url=os.getenv("OPENAI_BASE_URL"),
                   model=os.getenv("API_MODEL", "<provider-model-id>")),
    "ollama": dict(base_url="http://localhost:11434/v1",
                   model=os.getenv("OLLAMA_MODEL", "<ollama-model-name>")),
    "vllm":   dict(base_url=f"http://{os.getenv('VLLM_HOST', 'localhost')}:8000/v1",
                   model=os.getenv("VLLM_MODEL", "<hf-model-id>")),
}

def make_openai_backend(name):
    """Modes 1, 3 and 4: anything that speaks the OpenAI-compatible protocol."""
    from openai import OpenAI                         # pip install openai
    cfg = ENDPOINTS[name]
    if cfg["model"].startswith("<"):
        raise SystemExit(f"Set the model name for '{name}' in backends.py or the environment.")
    if cfg["base_url"]:                               # local servers ignore the key, but the
        client = OpenAI(base_url=cfg["base_url"],      # client insists on having one
                        api_key=os.getenv("OPENAI_API_KEY", "unused"))
    else:
        client = OpenAI()                             # reads OPENAI_API_KEY

    def call(messages, max_tokens=300, temperature=0):
        t0, first, parts = time.perf_counter(), None, []
        stream = client.chat.completions.create(model=cfg["model"], messages=messages,
                                                max_tokens=max_tokens, temperature=temperature,
                                                stream=True, stream_options={"include_usage": True})
        usage = None
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                first = first or time.perf_counter()  # time to first token
                parts.append(event.choices[0].delta.content)
            if event.usage:
                usage = event.usage
        end = time.perf_counter()
        return {"text": "".join(parts), "ttft_s": (first or end) - t0, "total_s": end - t0,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None}
    return call

def make_transformers_backend(model_id=None):
    """Mode 2: the model runs inside this Python process (Section 8.1). No streaming,
    so only total time is recorded."""
    from transformers import pipeline                 # pip install transformers accelerate
    model_id = model_id or os.getenv("LOCAL_HF_MODEL")
    if not model_id:
        raise SystemExit("Set LOCAL_HF_MODEL to a small instruction-tuned model id.")
    generator = pipeline("text-generation", model=model_id, device_map="auto")
    tok = generator.tokenizer

    def call(messages, max_tokens=300, temperature=0):
        t0 = time.perf_counter()
        out = generator(messages, max_new_tokens=max_tokens, do_sample=False)
        text = out[0]["generated_text"][-1]["content"]
        total = time.perf_counter() - t0
        n_in = len(tok.apply_chat_template(messages, add_generation_prompt=True))
        return {"text": text, "ttft_s": None, "total_s": total,
                "input_tokens": n_in, "output_tokens": len(tok(text)["input_ids"])}
    return call

def make_stub_backend():
    """Offline stand-in using the Chapter 7 stub, so the harness can be tested without
    a model. Its timings measure nothing but this laptop."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "chapter-07"))
    from rag_pipeline import offline_stub

    def call(messages, max_tokens=300, temperature=0):
        t0 = time.perf_counter()
        r = offline_stub(messages)
        t = time.perf_counter() - t0
        return {**r, "ttft_s": t, "total_s": t}
    return call

def get_backend(mode):
    if mode == "stub":
        return make_stub_backend()
    if mode == "inprocess":
        return make_transformers_backend()
    return make_openai_backend(mode)

# Section 8.3: retries with exponential backoff and jitter
RETRYABLE = (TimeoutError, ConnectionError)   # extend with each SDK's rate-limit and server errors
try:
    import openai
    RETRYABLE += (openai.RateLimitError, openai.APITimeoutError,
                  openai.APIConnectionError, openai.InternalServerError)
except ImportError:
    pass

def call_with_retries(call_fn, messages, max_attempts=5, base_delay=1.0, **kwargs):
    """Call any adapter, retrying transient failures with backoff and jitter."""
    for attempt in range(1, max_attempts + 1):
        try:
            return call_fn(messages, **kwargs)
        except RETRYABLE:
            if attempt == max_attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1) * (0.5 + random.random())  # 1, 2, 4 s ... with jitter
            time.sleep(delay)
