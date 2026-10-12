"""Section 8.4: one OpenAI-compatible client for every local server.

Start Ollama (`ollama serve`, then `ollama pull <model>`) or vLLM (`vllm serve <hf-model-id>`)
and pass the backend and model name:
    python sec_8_4_one_client.py ollama llama3.1:8b
    python sec_8_4_one_client.py vllm Qwen/Qwen2.5-7B-Instruct
"""
import sys
from openai import OpenAI

backends = {
    "ollama": OpenAI(base_url="http://localhost:11434/v1", api_key="unused"),
    "vllm":   OpenAI(base_url="http://localhost:8000/v1", api_key="unused"),
}
if len(sys.argv) < 3:
    raise SystemExit(__doc__)
reply = backends[sys.argv[1]].chat.completions.create(
    model=sys.argv[2], messages=[{"role": "user", "content": "Hello"}], max_tokens=100)
print(reply.choices[0].message.content)
