"""Animated Schwarzschild photon beam, propagating in real time.

A parallel beam of photons enters from the right; every photon then flies its own
orbit at once, advancing by the shared affine parameter. Captured photons (red)
peel in and vanish at the horizon; the rest bend around. The shadow is the gap
they leave behind.

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

R0 = 25.0        # start radius: photons begin lined up near the right edge
B_MAX = 14.0
N_RAYS = 121
WINDOW = 22.0
N_FRAMES = 80
FPS = 20


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
    """Run one photon; return (lambda, x, y, captured) with x,y rotated so the
    photon enters moving along -x (a parallel beam from the right)."""
    res = subprocess.run([exe, "photon", f"{b}", f"{R0}"], capture_output=True, text=True)
    if not res.stdout.strip():
        return None
    d = np.genfromtxt(res.stdout.splitlines(), delimiter=",", names=True)
    lam = np.atleast_1d(d["lambda"])
    x = np.atleast_1d(d["x"])
    y = np.atleast_1d(d["y"])
    if x.size < 2:
        return None
    delta = np.pi - np.arctan2(y[1] - y[0], x[1] - x[0])
    cs, sn = np.cos(delta), np.sin(delta)
    return lam, x * cs - y * sn, x * sn + y * cs, "captured" in res.stderr


def main() -> None:
    exe = find_exe()
    if not os.path.exists(exe):
        sys.exit(f"bhsim not found at {exe!r}; build it first or pass the path as arg 1.")
    default_out = os.path.join("simulation", "figures", "fan.gif")
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rays = []
    for b in np.linspace(-B_MAX, B_MAX, N_RAYS):
        if abs(b) < 1e-6:
            continue
        r = trace(exe, b)
        if r is not None:
            rays.append(r)

    # shared clock: advance lambda from 0 to where the beam has cleared the view
    lam_max = min(max(lam[-1] for lam, *_ in rays), 2.0 * R0 + 10.0)
    clock = np.linspace(0.0, lam_max, N_FRAMES)

    fig, ax = plt.subplots(figsize=(7, 7))
    frames = []
    for cur in clock:
        ax.clear()
        ax.add_patch(plt.Circle((0, 0), 2.0, color="black", zorder=5))
        ax.add_patch(plt.Circle((0, 0), 3.0, color="orange", fill=False,
                                ls="--", lw=1.0, zorder=4))
        for lam, xr, yr, captured in rays:
            idx = int(np.searchsorted(lam, cur, side="right"))
            if idx < 1:
                continue
            color = "#d62728" if captured else "#1f77b4"
            ax.plot(xr[:idx], yr[:idx], lw=0.6, alpha=0.6, color=color, zorder=3)
            if idx < len(lam):  # still in flight: draw the moving photon
                ax.plot(xr[idx - 1], yr[idx - 1], "o", ms=2.0, color=color, zorder=6)
        ax.set_aspect("equal")
        ax.set_xlim(-WINDOW, WINDOW)
        ax.set_ylim(-WINDOW, WINDOW)
        ax.set_xlabel("x / M")
        ax.set_ylabel("y / M")
        ax.set_title("Photon beam and the black-hole shadow")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    frames += [frames[-1]] * FPS  # hold the final shadow
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
