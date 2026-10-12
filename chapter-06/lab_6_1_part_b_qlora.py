"""Lab 6.1 Part B (Section 6.7): QLoRA fine-tuning of an open-weight model.

Needs a GPU with at least 16 GB of memory (a free cloud-notebook T4 works for 1 to 3 billion
parameter models; use 24 GB for 7 to 8 billion). Install:
    pip install transformers peft trl datasets bitsandbytes accelerate
Set MODEL_ID to an instruction-tuned model whose licence permits fine-tuning for your purpose.
Data: chat-format JSONL as in Section 6.4 (python make_sample_data.py writes a small sample).

Library interfaces in this area change quickly; check the TRL documentation for your version.
"""
import json, sys, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "your-org/your-chosen-instruct-model"         # set this to the model you have chosen
if "your-org" in MODEL_ID:
    sys.exit("Edit MODEL_ID first (see the docstring).")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

data = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/valid.jsonl"})

# Learn the replies, not the questions. assistant_only_loss needs a chat template with
# {% generation %} markers; if yours has none, use the prompt-completion format instead.
template = tokenizer.chat_template or ""
USE_ASSISTANT_ONLY = "{% generation %}" in template
if not USE_ASSISTANT_ONLY:
    def to_prompt_completion(row):
        return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}
    data = data.map(to_prompt_completion, remove_columns=["messages"])
    print("Chat template has no generation markers: using prompt-completion format with completion-only loss.")

print("One formatted example:\n", tokenizer.apply_chat_template(
    load_dataset("json", data_files="data/train.jsonl")["train"][0]["messages"], tokenize=False)[:800])

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

args = SFTConfig(output_dir="adapter-out", num_train_epochs=2, learning_rate=2e-4,
                 per_device_train_batch_size=4, gradient_accumulation_steps=4,
                 gradient_checkpointing=True, logging_steps=10, eval_strategy="epoch",
                 **({"assistant_only_loss": True} if USE_ASSISTANT_ONLY else {"completion_only_loss": True}))

trainer = SFTTrainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["test"],
                     peft_config=lora, processing_class=tokenizer)
trainer.train()
trainer.save_model("adapter-out")                     # saves only the small LoRA adapter
print("Adapter saved to adapter-out/. Evaluate it with the Chapter 5 workbench (Section 6.6).")
