# Lab 1.1: Meet the Models  (Chapter 1, Section "Hands-on Lab 1.1")
# pip install transformers torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: load a small open-weight model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

# Step 2: look inside one prediction
prompt = "The capital of India is"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]   # scores for the next token
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, k=5)
for p, idx in zip(top.values, top.indices):
    print(f"{tokenizer.decode(idx)!r:>14}  {p.item():.3f}")

# Step 3: watch sampling change the output
prompt = "Write one sentence about the future of manufacturing:"
inputs = tokenizer(prompt, return_tensors="pt")
for temperature in (0.2, 0.8, 1.5):
    output = model.generate(**inputs, max_new_tokens=40,
                            do_sample=True, temperature=temperature,
                            pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    print(f"\nT = {temperature}: {text}")
