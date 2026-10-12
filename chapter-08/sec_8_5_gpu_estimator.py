"""Section 8.5: does it fit, how fast will it be, and what will each token cost?

Reproduces Figure 8.2 (the single-user decoding ceiling for an 8 billion parameter model)
and the cost-per-million-tokens example. Change the numbers to match your own model,
hardware and prices.
    python sec_8_5_gpu_estimator.py            # prints the tables
    python sec_8_5_gpu_estimator.py --plot     # also saves figure_8_2.png (needs matplotlib)
"""
import sys

PARAMS_B = 8                                  # model size in billions of parameters
WEIGHTS_GB = {"4-bit": 4.9, "16-bit": 16.0}   # typical file sizes for an 8B model
GPUS = {                                      # memory (GB), bandwidth (GB/s): Table 8.4
    "NVIDIA T4":        (16, 320),
    "NVIDIA L4":        (24, 300),
    "GeForce RTX 4090": (24, 1008),
    "NVIDIA A100 80GB": (80, 2039),
    "NVIDIA H100 SXM":  (80, 3350),
}

def ceiling(bandwidth_gbs, weights_gb, memory_gb, headroom=0.9):
    """Tokens per second for one user if every weight is read once per token.
    Returns None when the weights do not fit (leaving some memory for the KV cache)."""
    if weights_gb > memory_gb * headroom:
        return None
    return bandwidth_gbs / weights_gb

def cost_per_million(server_usd_per_hour, tokens_per_second, utilisation=1.0):
    return server_usd_per_hour / (tokens_per_second * 3600 * utilisation) * 1e6

print(f"Single-user decoding ceiling, {PARAMS_B}B model (tokens per second)")
print(f"{'GPU':18s}" + "".join(f"{k:>14s}" for k in WEIGHTS_GB))
rows = {}
for gpu, (mem, bw) in GPUS.items():
    rows[gpu] = [ceiling(bw, w, mem) for w in WEIGHTS_GB.values()]
    print(f"{gpu:18s}" + "".join(f"{'does not fit' if c is None else round(c):>14}" for c in rows[gpu]))

print("\nCost per million tokens for a server at 2 USD an hour sustaining 1,000 tokens/s")
for u in (1.0, 0.5, 0.2):
    print(f"  utilisation {u:4.0%}: {cost_per_million(2.0, 1000, u):5.2f} USD")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt, numpy as np
    x, w = np.arange(len(GPUS)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, (label, colour) in enumerate((("4-bit", "#2A9D8F"), ("16-bit", "#1F3864"))):
        vals = [rows[g][j] or 0 for g in GPUS]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, label=f"{label} ({WEIGHTS_GB[label]} GB)", color=colour)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 8, "does not fit" if v == 0 else f"{v:.0f}",
                    ha="center", fontsize=8, rotation=90 if v == 0 else 0)
    ax.set_xticks(x, list(GPUS), fontsize=9)
    ax.set_ylabel("tokens per second, one user (ceiling)")
    ax.set_title(f"Decoding ceiling for an {PARAMS_B}B model")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("figure_8_2.png", dpi=150)
    print("\nSaved figure_8_2.png")
