"""Lab 6.1 Part A (Section 6.7): LoRA from scratch.

Adapts the Shakespeare MiniGPT from Chapter 3 to software-licence English and compares
full fine-tuning with LoRA at ranks 2, 8 and 32 (Table 6.7, Figure 6.4):
trainable parameters, loss on an unseen licence, and how much Shakespeare is forgotten.

Before running, copy mini_gpt.py, minigpt.pt and input.txt from chapter-03 into this folder.
Runs on a CPU: about 3 minutes per run, roughly 12 minutes in total.
The licence texts in data/ are verbatim copies, which their licences permit."""
import math, time, copy, json, re
import torch, torch.nn as nn, torch.nn.functional as F

exec(open("mini_gpt.py").read().split("# ---------------- 6. Training")[0])   # model, encode, data...
base_state = torch.load("minigpt.pt")
shakes_val = val_data

def clean(t):            # keep only characters the model knows; collapse spaces
    t = "".join(c if c in stoi else " " for c in t)
    return re.sub(r"[ ]{2,}", " ", t)
lic_train = torch.tensor(encode(clean(open("data/licences_train.txt", encoding="utf-8").read())))
lic_val   = torch.tensor(encode(clean(open("data/licences_val.txt", encoding="utf-8").read())))

def batch(src, bs=32):
    ix = torch.randint(len(src) - block_size - 1, (bs,))
    return (torch.stack([src[i:i+block_size] for i in ix]),
            torch.stack([src[i+1:i+block_size+1] for i in ix]))

@torch.no_grad()
def loss_on(m, src, n=60, seed=0):
    g = torch.Generator().manual_seed(seed); m.eval(); tot = 0
    for _ in range(n):
        ix = torch.randint(len(src) - block_size - 1, (32,), generator=g)
        x = torch.stack([src[i:i+block_size] for i in ix]); y = torch.stack([src[i+1:i+block_size+1] for i in ix])
        tot += m(x, y)[1].item()
    return tot / n

class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and only A, B trained."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero: starts as the base model
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

def merge_lora(model):
    """Exercise 3 helper: fold each LoRA update into its base weight, W + (alpha / r) B A."""
    for blk in model.blocks:
        for name in ("qkv", "proj"):
            layer = getattr(blk.attn, name)
            if isinstance(layer, LoRALinear):
                with torch.no_grad():
                    layer.base.weight += layer.scale * (layer.B @ layer.A)
                setattr(blk.attn, name, layer.base)
    return model

def add_lora(model, r):
    """Freeze everything, then add LoRA to the attention projections (as printed in Section 6.7)."""
    for p in model.parameters():
        p.requires_grad = False
    for blk in model.blocks:
        blk.attn.qkv  = LoRALinear(blk.attn.qkv,  r, alpha=2 * r)
        blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha=2 * r)
    return model

def make(method, r=None):
    m = MiniGPT(); m.load_state_dict(base_state)
    return add_lora(m, r) if method == "lora" else m

def train(m, steps=1500, lr=1e-3):
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    m.train(); t0 = time.time()
    for s in range(steps):
        x, y = batch(lic_train)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return time.time() - t0

torch.set_num_threads(2)
base = make("full")
results = [{"method": "Base model (no tuning)", "trainable": 0,
            "licence_val": loss_on(base, lic_val), "shakespeare_val": loss_on(base, shakes_val), "secs": 0}]
total = sum(p.numel() for p in base.parameters())
for method, r, lr in [("full", None, 3e-4), ("lora", 2, 1e-3), ("lora", 8, 1e-3), ("lora", 32, 1e-3)]:
    torch.manual_seed(0)
    m = make(method, r)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    secs = train(m, lr=lr)
    name = "Full fine-tuning" if method == "full" else f"LoRA, rank {r}"
    results.append({"method": name, "trainable": trainable, "licence_val": loss_on(m, lic_val),
                    "shakespeare_val": loss_on(m, shakes_val), "secs": round(secs, 1)})
    print(results[-1], flush=True)
    if method == "lora" and r == 8:
        m.eval(); torch.manual_seed(1)
        start = torch.tensor([encode("You may ")])
        print("SAMPLE:", decode(m.generate(start, 250, temperature=0.8)[0].tolist()))
torch.manual_seed(1); base.eval()
print("BASE SAMPLE:", decode(base.generate(torch.tensor([encode("You may ")]), 250, temperature=0.8)[0].tolist()))
print("total params", total, "licence chars", len(lic_train), len(lic_val))
json.dump({"total": total, "results": results}, open("lora_results.json", "w"), indent=1)
print("\n%-24s %10s %8s %16s %16s" % ("method", "trainable", "share", "licence loss", "Shakespeare loss"))
for r in results:
    print("%-24s %10d %7.2f%% %16.2f %16.2f" % (r["method"], r["trainable"], 100 * r["trainable"] / total,
                                               r["licence_val"], r["shakespeare_val"]))
