"""Animated Schwarzschild photon beam: the shadow carving out ray by ray.

Fires a parallel beam of photons past the black hole, then reveals the rays one
group at a time so the captured region (red) fills in as the shadow.

Usage:
    python simulation/viz/animate_fan.py [path/to/bhsim(.exe)] [out.gif]
"""

import os
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # render off-screen

R0 = 100.0
B_MAX = 14.0
N_RAYS = 121
WINDOW = 22.0
REVEAL = 3   # rays added per frame
FPS = 15


def find_exe() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    for c in [os.path.join("simulation", "build", "Release", "bhsim.exe"),
              os.path.join("simulation", "build", "bhsim"),
              os.path.join("simulation", "build", "Debug", "bhsim.exe")]:
        if os.path.exists(c):
            return c
    return os.path.join("simulation", "build", "Release", "bhsim.exe")


def trace(exe: str, b: float):
    """Run one photon; return rotated (x, y) and capture flag."""
    res = subprocess.run([exe, "photon", f"{b}", f"{R0}"], capture_output=True, text=True)
    if not res.stdout.strip():
        return None
    d = np.genfromtxt(res.stdout.splitlines(), delimiter=",", names=True)
    x = np.atleast_1d(d["x"])
    y = np.atleast_1d(d["y"])
    if x.size < 2:
        return None
    delta = np.pi - np.arctan2(y[1] - y[0], x[1] - x[0])
    cs, sn = np.cos(delta), np.sin(delta)
    return x * cs - y * sn, x * sn + y * cs, "captured" in res.stderr


def main() -> None:
    exe = find_exe()
    if not os.path.exists(exe):
        sys.exit(f"bhsim not found at {exe!r}; build it first or pass the path as arg 1.")
    default_out = os.path.join("simulation", "figures", "fan.gif")
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # fire every ray once, keep them all
    rays = []
    for b in np.linspace(-B_MAX, B_MAX, N_RAYS):
        if abs(b) < 1e-6:
            continue
        r = trace(exe, b)
        if r is not None:
            rays.append(r)

    fig, ax = plt.subplots(figsize=(7, 7))
    frames = []
    steps = list(range(REVEAL, len(rays) + REVEAL, REVEAL))
    for k in steps:
        ax.clear()
        ax.add_patch(plt.Circle((0, 0), 2.0, color="black", zorder=5))
        ax.add_patch(plt.Circle((0, 0), 3.0, color="orange", fill=False,
                                ls="--", lw=1.0, zorder=4))
        for xr, yr, captured in rays[:k]:
            ax.plot(xr, yr, lw=0.6, alpha=0.75, zorder=3,
                    color="#d62728" if captured else "#1f77b4")
        ax.set_aspect("equal")
        ax.set_xlim(-WINDOW, WINDOW)
        ax.set_ylim(-WINDOW, WINDOW)
        ax.set_xlabel("x / M")
        ax.set_ylabel("y / M")
        ax.set_title("Photon beam and the black-hole shadow")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    frames += [frames[-1]] * FPS  # hold the finished shadow for ~1 second
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
