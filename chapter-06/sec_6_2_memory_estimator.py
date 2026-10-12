"""Section 6.2 and Table 6.4: rough GPU memory needed to fine-tune a model.

Rules of thumb, before activations:
  full fine-tuning, mixed-precision AdamW: 2 (weights) + 2 (gradients) + 4 (fp32 master) + 8 (Adam) = 16 bytes/param
  LoRA on a 16-bit base: 2 bytes/param for the frozen base, plus a small trainable part
  QLoRA on a 4-bit base: about 0.5 bytes/param for the frozen base (plus quantisation constants)
Activations grow with batch size and sequence length; gradient checkpointing reduces them.
"""
import sys

def lora_params(d_model, n_layers, rank, matrices_per_layer=7):
    # all-linear LoRA on a typical block: q, k, v, o and three MLP projections, approximated as d x d each
    return n_layers * matrices_per_layer * rank * (d_model + d_model)

def estimate(params_b, d_model=4096, n_layers=32, rank=16):
    n = params_b * 1e9
    lp = lora_params(d_model, n_layers, rank)
    lora_train = lp * 16 / 1e9                     # LoRA weights, grads and Adam states
    rows = [
        ("Full fine-tuning (mixed precision)", n * 2 / 1e9, n * 14 / 1e9),
        ("LoRA on a 16-bit base", n * 2 / 1e9, lora_train),
        ("QLoRA on a 4-bit base", n * 0.55 / 1e9, lora_train),
    ]
    print(f"\nModel: {params_b:g} billion parameters; LoRA rank {rank} on all linear layers "
          f"= {lp / 1e6:.1f} million trainable values ({100 * lp / n:.2f}%)\n")
    print(f"{'Approach':38s} {'base weights':>13s} {'grads+optimiser':>16s} {'total':>9s}")
    for name, base, extra in rows:
        print(f"{name:38s} {base:10.1f} GB {extra:13.1f} GB {base + extra:6.1f} GB")
    print("\nAdd activations and framework overhead; measure before you buy or rent hardware.")

if __name__ == "__main__":
    estimate(float(sys.argv[1]) if len(sys.argv) > 1 else 7)
