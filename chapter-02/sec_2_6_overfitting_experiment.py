# Figure 2.5: polynomial overfitting experiment (Section 2.6)
import numpy as np
rng = np.random.default_rng(42)
f = lambda x: np.sin(2 * np.pi * x)
x_tr = np.sort(rng.uniform(0, 1, 15)); y_tr = f(x_tr) + rng.normal(0, 0.2, 15)
x_va = np.sort(rng.uniform(0, 1, 200)); y_va = f(x_va) + rng.normal(0, 0.2, 200)
for d in range(1, 9):          # degrees shown in Figure 2.5
    c = np.polyfit(x_tr, y_tr, d)
    tr = np.mean((np.polyval(c, x_tr) - y_tr) ** 2)
    va = np.mean((np.polyval(c, x_va) - y_va) ** 2)
    print(d, round(tr, 4), round(va, 4))
