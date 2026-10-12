"""Section 8.1: a chat model in a few lines with the Transformers pipeline.

    pip install transformers accelerate torch
    python sec_8_1_transformers_pipeline.py Qwen/Qwen2.5-0.5B-Instruct

Pass any instruction-tuned model whose licence suits your purpose. The first run downloads
the weights from the Hugging Face Hub.
"""
import sys
from transformers import pipeline

model_id = sys.argv[1] if len(sys.argv) > 1 else "your-org/small-instruct-model"
if model_id.startswith("your-org"):
    raise SystemExit(__doc__)
generator = pipeline("text-generation", model=model_id, device_map="auto")
messages = [
    {"role": "system", "content": "You explain tax notices in plain English for accountants."},
    {"role": "user", "content": "What does a notice under section 143(1) usually mean?"},
]
result = generator(messages, max_new_tokens=200)
print(result[0]["generated_text"][-1]["content"])      # the assistant's reply
