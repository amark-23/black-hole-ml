"""Animate a single photon orbit into a GIF.

Reads a CSV from the `bhsim` CLI and draws the path growing frame by frame,
with a moving dot for the photon. Frames are rendered off-screen and stitched
into a GIF with imageio.

Usage:
    python bhml/animate_orbit.py [orbit.csv] [out.gif]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # off-screen rendering; no window needed

import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

M = 1.0
N_FRAMES = 120
FPS = 25


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "orbit.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs", "figures", "whirl.gif")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x, y = np.atleast_1d(data["x"]), np.atleast_1d(data["y"])

    # static scene
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(plt.Circle((0, 0), 2 * M, color="black", zorder=3))
    ax.add_patch(plt.Circle((0, 0), 3 * M, color="orange", fill=False,
                            ls="--", lw=1.0, zorder=2))
    ax.set_aspect("equal")
    lim = 1.05 * max(np.max(np.abs(x)), np.max(np.abs(y)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / M")
    ax.set_ylabel("y / M")
    ax.set_title("Schwarzschild photon orbit")

    (path_line,) = ax.plot([], [], lw=1.5, color="#1f77b4", zorder=4)
    (photon,) = ax.plot([], [], "o", color="gold", ms=7, zorder=5)

    # pick ~N_FRAMES indices spread along the path
    idx = np.linspace(0, len(x) - 1, min(N_FRAMES, len(x))).astype(int)

    frames = []
    for j in idx:
        path_line.set_data(x[: j + 1], y[: j + 1])
        photon.set_data([x[j]], [y[j]])
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())  # drop alpha for GIF

    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
