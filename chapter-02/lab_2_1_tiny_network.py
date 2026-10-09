import numpy as np

rng = np.random.default_rng(0)

# ---------- Step 1: create a dataset that a straight line cannot separate ----------
def make_circles(n=400, noise=0.15):
    """Inner circle = class 1, outer ring = class 0."""
    angles = rng.uniform(0, 2 * np.pi, n)
    radius = np.where(np.arange(n) < n // 2, 0.5, 1.0)       # half inner, half outer
    X = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    X += rng.normal(0, noise, X.shape)
    y = (radius == 0.5).astype(float).reshape(-1, 1)
    return X, y

X, y = make_circles()
idx = rng.permutation(len(X))
X_train, y_train = X[idx[:300]], y[idx[:300]]
X_val, y_val = X[idx[300:]], y[idx[300:]]
print("train:", X_train.shape, "validation:", X_val.shape)  # (300, 2) (100, 2)

# ---------- Step 2: initialise a 2 -> 16 -> 1 network ----------
n_in, n_hidden, n_out = 2, 16, 1
W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))     # He initialisation
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out))
b2 = np.zeros((1, n_out))

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---------- Step 3: the forward pass ----------
def forward(X):
    z1 = X @ W1 + b1          # (N, 16)
    h = relu(z1)              # (N, 16)
    z2 = h @ W2 + b2          # (N, 1)
    p = sigmoid(z2)           # (N, 1) probability of class 1
    return z1, h, p

def bce_loss(p, y, eps=1e-9):
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

def accuracy(p, y):
    return np.mean((p > 0.5) == y)

# ---------- Step 4 and 5: backpropagation and gradient descent ----------
lr, epochs = 0.5, 2000
N = len(X_train)
for epoch in range(epochs + 1):
    z1, h, p = forward(X_train)

    # Backward pass: the chain rule, from the loss back to each parameter
    dz2 = (p - y_train) / N             # (N, 1)  sigmoid + cross-entropy
    dW2 = h.T @ dz2                     # (16, 1)
    db2 = dz2.sum(axis=0, keepdims=True)
    dh = dz2 @ W2.T                     # (N, 16)
    dz1 = dh * (z1 > 0)                 # (N, 16) ReLU passes gradient only where z1 > 0
    dW1 = X_train.T @ dz1               # (2, 16)
    db1 = dz1.sum(axis=0, keepdims=True)

    # Gradient descent update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoch % 400 == 0:
        _, _, p_val = forward(X_val)
        print(f"epoch {epoch:4d}  train loss {bce_loss(p, y_train):.3f}  "
              f"val loss {bce_loss(p_val, y_val):.3f}  "
              f"val accuracy {accuracy(p_val, y_val):.2f}")

# ---------- Step 6: check the gradients numerically ----------
def numerical_grad_W1(i, j, h=1e-5):
    old = W1[i, j]
    W1[i, j] = old + h; loss_plus = bce_loss(forward(X_train)[2], y_train)
    W1[i, j] = old - h; loss_minus = bce_loss(forward(X_train)[2], y_train)
    W1[i, j] = old
    return (loss_plus - loss_minus) / (2 * h)

z1, h, p = forward(X_train)
dz2 = (p - y_train) / N
dz1 = (dz2 @ W2.T) * (z1 > 0)
analytic = (X_train.T @ dz1)[0, 0]
print(f"gradient check: analytic {analytic:.6f}  numerical {numerical_grad_W1(0, 0):.6f}")
