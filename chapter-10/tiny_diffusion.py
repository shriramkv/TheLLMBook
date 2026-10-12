"""Lab 10.1, Part A (Section 10.6): a denoising diffusion model (DDPM) on 2D points.

The data is a spiral; the model learns to turn Gaussian noise into points on the spiral.
Reproduces Table 10.5 and the data for Figure 10.2. Trains in about 15 seconds on a CPU.
    python tiny_diffusion.py            # prints the table, writes snapshots.json
    python tiny_diffusion.py --plot     # also saves figure_10_2.png (needs matplotlib)
For exercise 1, change T below to 50 or 20; for exercise 2, replace spiral() with two circles.
"""
import math, sys, time, json, torch, torch.nn as nn
torch.manual_seed(0)

def spiral(n):
    t = torch.rand(n) * 3 * math.pi + math.pi / 2
    x = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 5.0
    return x + 0.03 * torch.randn_like(x)

T = 200
betas = torch.linspace(1e-4, 0.02 * 1000 / T, T)       # linear schedule, scaled for 200 steps
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)                          # cumulative product: how much signal survives

class Denoiser(nn.Module):
    """Predicts the noise that was added to x_t, given x_t and the step t."""
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t):
        freqs = torch.exp(torch.arange(8) * -math.log(1000) / 8)
        ang = t[:, None].float() * freqs[None]
        return self.net(torch.cat([x, torch.sin(ang), torch.cos(ang)], 1))   # sinusoidal step embedding

model = Denoiser()
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
data = spiral(20000)
t0 = time.time()
for step in range(6000):
    x0 = data[torch.randint(len(data), (512,))]
    t = torch.randint(T, (512,))
    eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps   # forward process in one jump
    loss = ((model(xt, t) - eps) ** 2).mean()                                # learn to predict the noise
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1500 == 0: print(f"step {step:5d}  loss {loss.item():.3f}", flush=True)
train_s = time.time() - t0

@torch.no_grad()
def sample(n, keep=(200, 40, 15, 0)):
    x, snaps = torch.randn(n, 2), {}
    for t in reversed(range(T)):
        if t + 1 in keep: snaps[t + 1] = x.clone()
        eps = model(x, torch.full((n,), t))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
    snaps[0] = x
    return snaps

torch.manual_seed(1)
snaps = sample(600)
ref = spiral(5000)
def mean_nn(p): return torch.cdist(p, ref).min(1).values.mean().item()
noise = torch.randn(600, 2)
print(f"parameters {sum(p.numel() for p in model.parameters()):,}, training {train_s:.0f} s")
for k in sorted(snaps, reverse=True): print(f"step {k:3d}: mean distance to nearest spiral point {mean_nn(snaps[k]):.3f}")
print(f"pure noise baseline {mean_nn(noise):.3f}; real data {mean_nn(spiral(600)):.3f}")
json.dump({str(k): [[round(a, 3), round(b, 3)] for a, b in v[:300].tolist()] for k, v in snaps.items()},
          open("snapshots.json", "w"))
print("signal kept sqrt(abar) at t=50,100,200:", [round(abar[i - 1].sqrt().item(), 3) for i in (50, 100, 200)])

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharex=True, sharey=True)
    for ax, k in zip(axes, sorted(snaps, reverse=True)):
        p = snaps[k][:600]
        ax.scatter(ref[:, 0], ref[:, 1], s=1, color="#d9d9d9")
        ax.scatter(p[:, 0], p[:, 1], s=4, color="#2A9D8F" if k == 0 else "#1F3864")
        ax.set_title("pure noise (step 200)" if k == 200 else f"step {k}", fontsize=10)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig("figure_10_2.png", dpi=150)
    print("Saved figure_10_2.png")
