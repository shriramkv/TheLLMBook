"""Chapter 11, Section 11.3: weight-only quantisation of the Chapter 3 MiniGPT.
Round-to-nearest, symmetric; 8-bit per output channel, 4/3/2-bit in groups of 32 weights.
Measures validation loss on held-out Shakespeare and the size of the stored weights."""
import copy, json, time, torch, torch.nn as nn
torch.set_num_threads(2)
exec(open("mini_gpt.py").read().split("# ---------------- 6. Training")[0])
base = MiniGPT(); base.load_state_dict(torch.load("minigpt.pt")); base.eval()

@torch.no_grad()
def val_loss(m, n=200, seed=0):
    g = torch.Generator().manual_seed(seed); tot = 0.0
    for _ in range(n):
        ix = torch.randint(len(val_data) - block_size - 1, (32,), generator=g)
        x = torch.stack([val_data[i:i + block_size] for i in ix]); y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
        tot += m(x, y)[1].item()
    return tot / n

def quantise(w, bits, group):
    """Fake-quantise (quantise then dequantise) a 2-D weight, returning the new weight and its stored size in bytes."""
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    g = in_f if group is None else group
    wg = w.reshape(out_f, in_f // g, g)
    scale = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax)
    size = w.numel() * bits / 8 + scale.numel() * 2          # integer codes + one 16-bit scale per group
    return (q * scale).reshape(out_f, in_f), size

results = []
for name, bits, group in [("32-bit float", 32, None), ("16-bit float", 16, None), ("8-bit, per channel", 8, None),
                          ("4-bit, groups of 32", 4, 32), ("3-bit, groups of 32", 3, 32), ("2-bit, groups of 32", 2, 32)]:
    m = copy.deepcopy(base); size = 0
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)) and mod.weight.dim() == 2:
                w = mod.weight
                if bits == 32: size += w.numel() * 4; continue
                if bits == 16:
                    mod.weight.copy_(w.half().float()); size += w.numel() * 2; continue
                new, s = quantise(w, bits, group); mod.weight.copy_(new); size += s
        # count each tied weight once
    seen = {id(p) for p in m.parameters()}
    loss = val_loss(m)
    results.append(dict(format=name, bits=bits, size_kb=round(size / 1024, 1), val_loss=round(loss, 3)))
    print(results[-1], flush=True)
base_loss = results[0]["val_loss"]
for r in results: r["loss_change_pct"] = round(100 * (r["val_loss"] - base_loss) / base_loss, 1)
json.dump(results, open("quant_results.json", "w"), indent=1)
for r in results: print(r)
