"""Plot a Schwarzschild photon orbit from a CSV written by the `bhsim` CLI.

Usage:
    python bhml/plot_orbit.py [orbit.csv] [out.png]

The CSV has columns: lambda, r, phi, x, y  (x, y in units of M).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

M = 1.0  # black-hole mass; matches the simulator's default


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "orbit.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs", "figures", "orbit.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x, y = data["x"], data["y"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Landmarks: event horizon (2M, filled) and photon sphere (3M, dashed).
    ax.add_patch(plt.Circle((0, 0), 2 * M, color="black", zorder=3))
    ax.add_patch(plt.Circle((0, 0), 3 * M, color="orange", fill=False,
                            ls="--", lw=1.0, zorder=2, label="photon sphere (3M)"))

    ax.plot(x, y, lw=1.5, color="#1f77b4", zorder=4, label="photon path")
    ax.plot(x[0], y[0], "o", color="green", ms=6, zorder=5, label="start")

    ax.set_aspect("equal")  # so the horizon is a real circle, not an ellipse
    lim = 1.05 * max(np.max(np.abs(x)), np.max(np.abs(y)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / M")
    ax.set_ylabel("y / M")
    ax.set_title("Schwarzschild photon orbit")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
