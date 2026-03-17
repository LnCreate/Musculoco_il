import numpy as np

def _hinge_sq(x):
    z = np.maximum(0.0, x)
    return z * z

bw = 75.0 * 9.81
fz = np.linspace(0, 1000, 100)

low_tr = 0.02 * bw
high_tr = 0.08 * bw
p_stance = np.clip((fz - low_tr) / (high_tr - low_tr), 0.0, 1.0)
print(f"low_tr={low_tr}, high_tr={high_tr}")
for f, p in zip(fz[::10], p_stance[::10]):
    print(f"{f:.1f} N -> {p:.2f}")
