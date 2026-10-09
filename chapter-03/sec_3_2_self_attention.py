import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """Scaled dot-product attention. Q, K: (n, d_k); V: (n, d_v)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)            # (n, n) similarity of every pair
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # block disallowed positions
    weights = softmax(scores, axis=-1)         # each row sums to 1
    return weights @ V, weights                # (n, d_v), (n, n)

# Three tokens: "the", "river", "bank", with d_k = d_v = 2
Q = np.array([[1.0, 0.0],    # the
              [0.0, 1.0],    # river
              [1.0, 1.0]])   # bank
K = np.array([[1.0, 0.0],
              [0.0, 2.0],
              [0.5, 0.5]])
V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [0.5, 0.5]])

out, w = attention(Q, K, V)
print(np.round(w, 2))
print(np.round(out, 2))
