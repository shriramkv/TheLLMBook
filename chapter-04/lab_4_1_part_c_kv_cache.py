# Lab 4.1 Part C: KV cache for MiniGPT. Copy mini_gpt.py and minigpt.pt from chapter-03 first.
import time, math
import torch
import torch.nn.functional as F

# Reuse the MiniGPT definition and trained weights from Chapter 3
exec(open("mini_gpt.py").read().split("# ---------------- 6. Training")[0])
model.load_state_dict(torch.load("minigpt.pt"))
model.eval()

@torch.no_grad()
def step_with_cache(model, idx_new, cache, start_pos):
    """Run only the NEW tokens through the model, reusing cached keys and values."""
    B, T = idx_new.shape
    pos = torch.arange(start_pos, start_pos + T)
    x = model.tok_emb(idx_new) + model.pos_emb(pos)
    for layer, blk in enumerate(model.blocks):
        a = blk.attn
        h = blk.ln1(x)
        q, k, v = a.qkv(h).split(d_model, dim=2)
        q, k, v = [t.view(B, T, n_heads, -1).transpose(1, 2) for t in (q, k, v)]
        if cache[layer] is not None:                       # append to the stored keys and values
            k = torch.cat([cache[layer][0], k], dim=2)
            v = torch.cat([cache[layer][1], v], dim=2)
        cache[layer] = (k, v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        if T > 1:                                          # causal mask only needed in prefill
            mask = torch.tril(torch.ones(T, k.size(2), dtype=torch.bool), diagonal=k.size(2) - T)
            scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        x = x + a.proj(out)
        x = x + blk.ffn(blk.ln2(x))
    return model.head(model.ln_f(x))

@torch.no_grad()
def generate_no_cache(model, idx, n):
    for _ in range(n):
        logits, _ = model(idx)                             # recompute the whole sequence every step
        idx = torch.cat([idx, logits[:, -1:].argmax(-1)], dim=1)
    return idx

@torch.no_grad()
def generate_with_cache(model, idx, n):
    cache = [None] * n_layers
    logits = step_with_cache(model, idx, cache, 0)         # prefill: the whole prompt at once
    for _ in range(n):
        nxt = logits[:, -1:].argmax(-1)
        idx = torch.cat([idx, nxt], dim=1)
        logits = step_with_cache(model, nxt, cache, idx.size(1) - 1)   # decode: one token
    return idx

prompt = torch.tensor([encode("ROMEO:\n")])
n_new = block_size - prompt.size(1)                        # stay inside the 64-token context
a = generate_no_cache(model, prompt, n_new)
b = generate_with_cache(model, prompt, n_new)
print("identical output:", torch.equal(a, b))

torch.set_num_threads(1)
for name, fn in [("without cache", generate_no_cache), ("with cache", generate_with_cache)]:
    t0 = time.perf_counter()
    for _ in range(20):
        fn(model, prompt, n_new)
    print(f"{name:14s} {(time.perf_counter() - t0) / 20 * 1000:.1f} ms per {n_new}-token generation")
