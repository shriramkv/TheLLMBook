# Section 3.7: plot attention heatmaps from a trained MiniGPT.
# Run mini_gpt.py first; it saves minigpt.pt in this folder.
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the model definition from mini_gpt.py without retraining
exec(open("mini_gpt.py").read().split("# ---------------- 6. Training")[0])
model.load_state_dict(torch.load("minigpt.pt"))
model.eval()

def show_attention(model, text, layer, head):
    idx = torch.tensor([encode(text)])
    model(idx)                                              # forward pass stores weights
    w = model.blocks[layer].attn.last_weights[0, head]     # (T, T)
    labels = [c if c != " " else "␣" for c in text]   # make spaces visible
    plt.figure(figsize=(5, 5))
    plt.imshow(w, cmap="Blues")
    plt.xticks(range(len(text)), labels)
    plt.yticks(range(len(text)), labels)
    plt.xlabel("attended-to character (key)")
    plt.ylabel("current character (query)")
    plt.title(f"Layer {layer + 1}, head {head + 1}")
    plt.savefig(f"attention_L{layer + 1}_H{head + 1}.png", dpi=150, bbox_inches="tight")
    plt.close()

for layer in range(n_layers):
    for head in range(n_heads):
        show_attention(model, "hear me speak.", layer=layer, head=head)
print("Saved one heatmap per head.")
