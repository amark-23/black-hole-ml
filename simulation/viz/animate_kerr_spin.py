"""Animate the Kerr shadow as spin ramps up.

Sweeps the spin a from 0 to 0.99 and, at each value, fires an equatorial photon
beam past the hole. As a grows, frame dragging pushes the captured region (red)
off-center and lopsided -- the shadow morphs from Schwarzschild-symmetric to the
classic asymmetric Kerr shape. Frames are stitched into a GIF.

Usage:
    python bhml/animate_kerr_spin.py [path/to/bhsim(.exe)] [out.gif]
"""

import os
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # render off-screen

R0 = 40.0
B_MAX = 9.0
N_RAYS = 81
WINDOW = 12.0
N_FRAMES = 25
FPS = 12


def find_exe() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    for c in [os.path.join("simulation", "build", "Release", "bhsim.exe"),
              os.path.join("simulation", "build", "bhsim"),
              os.path.join("simulation", "build", "Debug", "bhsim.exe")]:
        if os.path.exists(c):
            return c
    return os.path.join("simulation", "build", "Release", "bhsim.exe")


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
    delta = np.pi - np.arctan2(y[1] - y[0], x[1] - x[0])
    cs, sn = np.cos(delta), np.sin(delta)
    return x * cs - y * sn, x * sn + y * cs, "captured" in res.stderr


def main() -> None:
    exe = find_exe()
    if not os.path.exists(exe):
        sys.exit(f"bhsim not found at {exe!r} — build it first, or pass the path as arg 1.")
    default_out = os.path.join("simulation", "figures", "kerr_spin.gif")
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    frames = []
    for a in np.linspace(0.0, 0.99, N_FRAMES):
        ax.clear()
        r_plus = 1.0 + np.sqrt(1.0 - a * a)
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
        ax.set_title(f"Kerr shadow,  spin a = {a:.2f}")
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())
        print(f"  a = {a:.2f} done")

    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
