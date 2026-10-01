"""Animated Kerr photon beam: the shadow carving out, side by side for two spins.

Fires a photon beam past a non-spinning (a=0) and a fast-spinning (a=0.9) black
hole, then reveals the rays group by group so the two shadows build up together.
The spinning one comes out lopsided: frame dragging shifts its capture region.

Usage:
    python simulation/viz/animate_kerr_fan.py [path/to/bhsim(.exe)] [out.gif]
"""

import os
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")  # render off-screen

R0 = 60.0
B_MAX = 10.0
N_RAYS = 101
WINDOW = 15.0
REVEAL = 3
FPS = 15
SPINS = [0.0, 0.9]


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
    res = subprocess.run([exe, "kerr", f"{a}", f"{b}", f"{R0}"], capture_output=True, text=True)
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
    default_out = os.path.join("simulation", "figures", "kerr_fan.gif")
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # fire every ray once for each spin
    rays_by_spin = []
    for a in SPINS:
        rays = []
        for b in np.linspace(-B_MAX, B_MAX, N_RAYS):
            if abs(b) < 1e-6:
                continue
            r = trace(exe, a, b)
            if r is not None:
                rays.append(r)
        rays_by_spin.append(rays)

    n = min(len(r) for r in rays_by_spin)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    frames = []
    for k in range(REVEAL, n + REVEAL, REVEAL):
        for ax, a, rays in zip(axes, SPINS, rays_by_spin):
            ax.clear()
            r_plus = 1.0 + np.sqrt(1.0 - a * a)
            ax.add_patch(plt.Circle((0, 0), r_plus, color="black", zorder=5))
            for xr, yr, captured in rays[:k]:
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
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    frames += [frames[-1]] * FPS
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
