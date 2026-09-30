"""Kerr photon fan: the asymmetric shadow from frame dragging.

Fires an equatorial photon beam past a spinning black hole for a = 0 and
a = 0.9 side by side. Captured rays are red; frame dragging shifts the capture
region (the shadow) off-center. The hole spins counter-clockwise, so prograde
rays (co-rotating) are captured at smaller impact parameter than retrograde ones.

Usage:
    python bhml/plot_kerr_fan.py [path/to/bhsim(.exe)] [out.png]
"""

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

R0 = 60.0
B_MAX = 10.0
N_RAYS = 161
WINDOW = 15.0
SPINS = [0.0, 0.9]


def find_exe() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    for c in [os.path.join("sim", "build", "Release", "bhsim.exe"),
              os.path.join("sim", "build", "bhsim"),
              os.path.join("sim", "build", "Debug", "bhsim.exe")]:
        if os.path.exists(c):
            return c
    return os.path.join("sim", "build", "Release", "bhsim.exe")


def trace(exe: str, a: float, b: float):
    """Run one Kerr equatorial photon; return rotated (x, y) and capture flag."""
    res = subprocess.run([exe, "kerr", f"{a}", f"{b}", f"{R0}"],
                         capture_output=True, text=True)
    if not res.stdout.strip():
        return None, None, False
    d = np.genfromtxt(res.stdout.splitlines(), delimiter=",", names=True)
    x = np.atleast_1d(d["x"])
    y = np.atleast_1d(d["y"])
    if x.size < 2:
        return None, None, False
    # Kerr is axisymmetric, so rotating the whole path in phi is exact.
    # Rotate so the entry velocity points along -x (a parallel beam from the right).
    delta = np.pi - np.arctan2(y[1] - y[0], x[1] - x[0])
    cs, sn = np.cos(delta), np.sin(delta)
    return x * cs - y * sn, x * sn + y * cs, "captured" in res.stderr


def main() -> None:
    exe = find_exe()
    if not os.path.exists(exe):
        sys.exit(f"bhsim not found at {exe!r} — build it first, or pass the path as arg 1.")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs", "figures", "kerr_fan.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax, a in zip(axes, SPINS):
        r_plus = 1.0 + np.sqrt(1.0 - a * a)  # outer horizon (M = 1)
        ax.add_patch(plt.Circle((0, 0), r_plus, color="black", zorder=5))

        for b in np.linspace(-B_MAX, B_MAX, N_RAYS):
            if abs(b) < 1e-6:
                continue
            xr, yr, captured = trace(exe, a, b)
            if xr is None:
                continue
            ax.plot(xr, yr, lw=0.6, alpha=0.75, zorder=3,
                    color="#d62728" if captured else "#1f77b4")

        ax.set_aspect("equal")
        ax.set_xlim(-WINDOW, WINDOW)
        ax.set_ylim(-WINDOW, WINDOW)
        ax.set_xlabel("x / M")
        ax.set_ylabel("y / M")
        ax.set_title(f"a = {a}" + ("  (Schwarzschild)" if a == 0 else "  (frame dragging)"))

    fig.suptitle("Photon beam past a black hole: the shadow shifts with spin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
