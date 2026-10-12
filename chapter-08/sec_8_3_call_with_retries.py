"""Section 8.3: retries with exponential backoff and jitter.

Runs offline: a deliberately unreliable stand-in for an API fails twice and then succeeds,
so you can watch the delays grow. The same call_with_retries wraps any adapter.
    python sec_8_3_call_with_retries.py
"""
import time
from backends import call_with_retries

attempts = {"n": 0}
t0 = time.perf_counter()

def flaky_api(messages, **kwargs):
    attempts["n"] += 1
    print(f"  attempt {attempts['n']} at {time.perf_counter() - t0:4.1f} s")
    if attempts["n"] < 3:
        raise ConnectionError("simulated network failure")
    return {"text": "OK", "input_tokens": 5, "output_tokens": 1}

print("Calling an API that fails twice:")
print("Result:", call_with_retries(flaky_api, [{"role": "user", "content": "hi"}], base_delay=0.5))

def bad_request(messages, **kwargs):
    raise ValueError("invalid request: this should not be retried")

print("\nAn invalid request is not retried:")
try:
    call_with_retries(bad_request, [])
except ValueError as e:
    print("  raised immediately:", e)
