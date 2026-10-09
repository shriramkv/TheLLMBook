import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------- 1. Data: a character-level tokeniser ----------------
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

# ---------------- 2. Hyperparameters ----------------
block_size = 64      # context length in characters
batch_size = 32
d_model = 128
n_heads = 4
n_layers = 4
dropout = 0.1
max_steps = 3000
lr = 3e-4

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i:i + block_size] for i in ix])          # (B, T)
    y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])  # (B, T) shifted by one
    return x, y

# ---------------- 3. Causal multi-head self-attention ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_Q, W_K, W_V in one matrix
        self.proj = nn.Linear(d_model, d_model)      # W_O
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)
        self.last_weights = None                     # kept for visualisation (Section 3.7)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=2)                    # each (B, T, C)
        # split into heads: (B, n_heads, T, head_dim)
        q, k, v = [t.view(B, T, n_heads, C // n_heads).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))      # (B, h, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        out = self.drop(weights) @ v                                   # (B, h, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)           # concatenate heads
        return self.proj(out)

# ---------------- 4. The transformer block (pre-norm) ----------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # residual connection around attention
        x = x + self.ffn(self.ln2(x))      # residual connection around the FFN
        return x

# ---------------- 5. The full model ----------------
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)   # learned positions
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight              # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Small initial weights (std 0.02), as in GPT-2, keep the first logits near zero
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B, T, C)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))                   # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = MiniGPT()
print(f"vocabulary: {vocab_size} characters, parameters: "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# ---------------- 6. Training ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

@torch.no_grad()
def estimate_loss(n_batches=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(n_batches)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

for step in range(max_steps + 1):
    if step % 500 == 0:
        l = estimate_loss()
        print(f"step {step:5d}  train loss {l['train']:.3f}  val loss {l['val']:.3f}")
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# ---------------- 7. Generate ----------------
model.eval()
start = torch.tensor([encode("ROMEO:")], dtype=torch.long)
print(decode(model.generate(start, max_new_tokens=300, temperature=0.8)[0].tolist()))
torch.save(model.state_dict(), "minigpt.pt")
