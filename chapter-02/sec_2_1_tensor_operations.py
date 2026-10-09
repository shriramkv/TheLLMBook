import numpy as np

# Vectors and the dot product
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print(a @ b)                      # 1*4 + 2*5 + 3*6 = 32.0

# Matrix multiplication: (2, 3) @ (3, 2) -> (2, 2)
W = np.array([[1, 0, 2],
              [0, 1, 1]])
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print((W @ X).shape)              # (2, 2)

# A batch of token embeddings: (batch, sequence, d_model)
batch, seq_len, d_model = 2, 5, 8
H = np.random.randn(batch, seq_len, d_model)
W_q = np.random.randn(d_model, d_model)
Q = H @ W_q                       # the same weights applied to every token
print(Q.shape)                    # (2, 5, 8)

# Broadcasting: add one bias vector to every token in every sequence
bias = np.zeros(d_model)
print((Q + bias).shape)           # (2, 5, 8)
