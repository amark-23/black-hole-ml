"""Generate a photon capture/escape dataset using the C++ integrator.

Samples random photons (impact parameter b, start radius r0), labels each by
whether it is captured, and saves the result to an .npz for training.

Run from the repo root (with the venv active and the _bhsim extension built):
    python -m bhml.data [n_samples] [out.npz]
"""

import os
import sys

import numpy as np

from bhml.sim import trace_photon

B_RANGE = (3.0, 8.0)      # brackets b_crit = 3*sqrt(3) ~ 5.196
R0_RANGE = (20.0, 50.0)   # decoy: capture is independent of r0
SEED = 0


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "capture.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    rng = np.random.default_rng(SEED)
    b = rng.uniform(*B_RANGE, size=n)
    r0 = rng.uniform(*R0_RANGE, size=n)

    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        _, outcome = trace_photon(float(b[i]), float(r0[i]))
        y[i] = 1 if outcome == "captured" else 0

    X = np.column_stack([b, r0]).astype(np.float64)
    np.savez(out, X=X, y=y, feature_names=np.array(["b", "r0"]))

    n_cap = int(y.sum())
    b_crit = 3.0 * np.sqrt(3.0)
    print(f"saved {out}: {n} samples, {n_cap} captured "
          f"({100 * n_cap / n:.1f}%), {n - n_cap} escaped")
    if 0 < n_cap < n:
        print(f"empirical boundary: max captured b = {b[y == 1].max():.4f}, "
              f"min escaped b = {b[y == 0].min():.4f}")
    print(f"analytic b_crit = 3*sqrt(3) = {b_crit:.4f}")


if __name__ == "__main__":
    main()
