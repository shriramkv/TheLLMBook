import numpy as np

def softmax(z, temperature=1.0):
    z = np.asarray(z, dtype=float) / temperature
    z = z - z.max()                   # subtract the max for numerical stability
    e = np.exp(z)
    return e / e.sum()

logits = [4.0, 2.0, 1.0, 0.5]         # raw scores for four candidate tokens
p = softmax(logits)
print(np.round(p, 3))                 # [0.811 0.11  0.04  0.024]

entropy = -(p * np.log2(p)).sum()
print(round(entropy, 3))              # uncertainty in bits

target = 0                            # the true next token is candidate 0
cross_entropy = -np.log(p[target])
print(round(cross_entropy, 3))        # loss for this one prediction
print(round(np.exp(cross_entropy), 3))  # perplexity for this one prediction
