import numpy as np

def sinusoidal_positions(n_positions, d_model):
    pos = np.arange(n_positions)[:, None]                  # (n, 1)
    i = np.arange(0, d_model, 2)[None, :]                  # (1, d/2)
    angles = pos / (10000 ** (i / d_model))                # (n, d/2)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angles)                           # even dimensions
    pe[:, 1::2] = np.cos(angles)                           # odd dimensions
    return pe

pe = sinusoidal_positions(n_positions=128, d_model=64)
print(pe.shape)                       # (128, 64)
print(np.round(pe[1, :4], 3))         # first values for position 1
