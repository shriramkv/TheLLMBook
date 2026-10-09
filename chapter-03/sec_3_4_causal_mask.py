import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

n, d_k = 5, 4
rng = np.random.default_rng(1)
Q5, K5, V5 = (rng.standard_normal((n, d_k)) for _ in range(3))

causal_mask = np.tril(np.ones((n, n), dtype=bool))   # lower triangle = allowed
out, w = attention(Q5, K5, V5, mask=causal_mask)
print(np.round(w, 2))
