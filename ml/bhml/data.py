"""Generate datasets for the black-hole ML tasks, using the C++ integrator.

Run from the repo root (venv active, _bhsim built):
    python -m bhml.data capture    [n] [out.npz]   # capture/escape labels
    python -m bhml.data deflection [n] [out.npz]   # escaping-photon deflection angle
"""

import os
import sys

import numpy as np

from bhml.sim import trace_photon

SEED = 0

# capture task
CAP_B_RANGE = (3.0, 8.0)      # brackets b_crit = 3*sqrt(3) ~ 5.196
CAP_R0_RANGE = (20.0, 50.0)   # decoy: capture is independent of r0

# deflection task
DEF_B_RANGE = (5.25, 30.0)    # escaping photons, just above b_crit outward
DEF_R0 = 1.0e4                # start far out so the measured deflection is ~asymptotic


def gen_capture(n: int, out: str) -> None:
    rng = np.random.default_rng(SEED)
    b = rng.uniform(*CAP_B_RANGE, size=n)
    r0 = rng.uniform(*CAP_R0_RANGE, size=n)

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


def gen_deflection(n: int, out: str) -> None:
    rng = np.random.default_rng(SEED)
    b = rng.uniform(*DEF_B_RANGE, size=n)

    delta = np.empty(n, dtype=np.float64)
    n_bad = 0
    for i in range(n):
        traj, outcome = trace_photon(float(b[i]), DEF_R0)
        if outcome != "escaped":
            delta[i] = np.nan
            n_bad += 1
            continue
        phi_end = float(traj["phi"][-1])
        delta[i] = phi_end - np.pi + 2.0 * np.arcsin(b[i] / DEF_R0)

    keep = ~np.isnan(delta)
    b, delta = b[keep], delta[keep]
    X = b.reshape(-1, 1).astype(np.float64)
    y = delta.astype(np.float64)
    np.savez(out, X=X, y=y, feature_names=np.array(["b"]))

    j = int(np.argmax(b))  # largest b -> weakest field, check against 4M/b
    print(f"saved {out}: {len(b)} samples ({n_bad} non-escaping dropped)")
    print(f"deflection range: {delta.min():.4f} to {delta.max():.4f} rad")
    print(f"weak-field check at b = {b[j]:.2f}: delta = {delta[j]:.5f}, "
          f"4M/b = {4.0 / b[j]:.5f}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    if mode not in ("capture", "deflection"):
        sys.exit(f"unknown mode {mode!r}; use 'capture' or 'deflection'")

    default_n = 4000 if mode == "capture" else 3000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else default_n
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join("data", f"{mode}.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if mode == "capture":
        gen_capture(n, out)
    else:
        gen_deflection(n, out)


if __name__ == "__main__":
    main()
