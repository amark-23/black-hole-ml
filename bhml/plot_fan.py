"""Fan of photons past a Schwarzschild black hole -> the shadow.

Calls the `bhsim` CLI over a range of impact parameters, rotates each
trajectory so it enters as a horizontal (parallel) ray, and plots them all.
Captured rays are drawn in red; the dark wedge they fill is the shadow.

Usage:
    python bhml/plot_fan.py [path/to/bhsim(.exe)] [out.png]
"""

import os
import subprocess
import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

M = 1.0
R0 = 100.0          # start radius (far enough that rays enter nearly parallel)
B_MAX = 14.0        # widest impact parameter
N_RAYS = 141        # number of rays across the beam
WINDOW = 22.0       # half-width of the plotted region


def find_exe() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = [
        os.path.join("sim", "build", "Release", "bhsim.exe"),  # MSVC
        os.path.join("sim", "build", "bhsim"),                 # make/ninja
        os.path.join("sim", "build", "Debug", "bhsim.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # report the expected path if none found


def trace(exe: str, b: float):
    """Run one photon; return rotated (x, y) and whether it was captured."""
    res = subprocess.run([exe, f"{b}", f"{R0}"], capture_output=True, text=True)
    if not res.stdout.strip():
        return None, None, False
    d = np.genfromtxt(StringIO(res.stdout), delimiter=",", names=True)
    x = np.atleast_1d(d["x"])
    y = np.atleast_1d(d["y"])
    if x.size < 2:
        return None, None, False
    # rotate so the entry velocity points along -x (a parallel beam from the right)
    delta = np.pi - np.arctan2(y[1] - y[0], x[1] - x[0])
    cs, sn = np.cos(delta), np.sin(delta)
    xr = x * cs - y * sn
    yr = x * sn + y * cs
    captured = "captured" in res.stderr
    return xr, yr, captured


def main() -> None:
    exe = find_exe()
    if not os.path.exists(exe):
        sys.exit(f"bhsim not found at {exe!r} — build it first, or pass the path as arg 1.")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs", "figures", "fan.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Circle((0, 0), 2 * M, color="black", zorder=5))
    ax.add_patch(plt.Circle((0, 0), 3 * M, color="orange", fill=False,
                            ls="--", lw=1.0, zorder=4))

    for b in np.linspace(-B_MAX, B_MAX, N_RAYS):
        if abs(b) < 1e-6:
            continue
        xr, yr, captured = trace(exe, b)
        if xr is None:
            continue
        ax.plot(xr, yr, lw=0.6, alpha=0.75, zorder=3,
                color="#d62728" if captured else "#1f77b4")

    ax.set_aspect("equal")
    ax.set_xlim(-WINDOW, WINDOW)
    ax.set_ylim(-WINDOW, WINDOW)
    ax.set_xlabel("x / M")
    ax.set_ylabel("y / M")
    ax.set_title("Photon beam and the black-hole shadow")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
