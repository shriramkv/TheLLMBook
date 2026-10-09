import numpy as np

# Toy 4-dimensional embeddings. Dimensions, loosely:
# [royalty, maleness, femaleness, is_a_fruit]
emb = {
    "king":   np.array([0.9, 0.8, 0.1, 0.0]),
    "queen":  np.array([0.9, 0.1, 0.8, 0.0]),
    "man":    np.array([0.1, 0.9, 0.1, 0.0]),
    "woman":  np.array([0.1, 0.1, 0.9, 0.0]),
    "mango":  np.array([0.0, 0.0, 0.0, 1.0]),
}

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(round(cosine(emb["king"], emb["queen"]), 2))   # related words
print(round(cosine(emb["king"], emb["mango"]), 2))   # unrelated words

# The analogy: king - man + woman should land near queen
target = emb["king"] - emb["man"] + emb["woman"]
best = max((w for w in emb if w not in ("king", "man", "woman")),
           key=lambda w: cosine(target, emb[w]))
print(best)                                          # queen
