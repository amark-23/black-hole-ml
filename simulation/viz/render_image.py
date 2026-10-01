"""Render a black-hole image with the C++ ray tracer and colour it.

`bhsim image` traces one light ray per pixel and prints a grid of brightness
values. This script runs it, applies a colormap, and saves a PNG. With --sweep it
renders a sequence of increasing spins and stitches them into a gif, so you can
watch the shadow go lopsided as frame dragging sets in.

Usage:
    python simulation/viz/render_image.py [path/to/bhsim(.exe)]
    python simulation/viz/render_image.py --sweep [path/to/bhsim(.exe)]
"""

import os
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np

INCL = 80.0        # viewing inclination from the spin axis, degrees (near edge-on)
HALF_WIDTH = 12.0  # half the image width on the sky, in units of M
RES = 300          # pixels per side for a single still
CMAP = "inferno"


def find_exe() -> str:
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            return arg
    for c in [os.path.join("simulation", "build", "Release", "bhsim.exe"),
              os.path.join("simulation", "build", "bhsim"),
              os.path.join("simulation", "build", "Debug", "bhsim.exe")]:
        if os.path.exists(c):
            return c
    return os.path.join("simulation", "build", "Release", "bhsim.exe")


def render(exe: str, a: float, res: int) -> np.ndarray:
    """Trace an image and return it as a (res, res) array of brightness."""
    res_arg = f"{res}"
    out = subprocess.run([exe, "image", f"{a}", f"{INCL}", f"{HALF_WIDTH}", res_arg],
                         capture_output=True, text=True)
    lines = out.stdout.splitlines()
    if not lines:
        sys.exit(f"no output from {exe!r}; build it first or pass the path as arg 1.")
    return np.array([[float(v) for v in row.split()] for row in lines[1:]])


def colorize(img: np.ndarray) -> np.ndarray:
    """Map brightness to 8-bit RGB with the chosen colormap.

    Doppler beaming gives the approaching side a huge brightness spike, so a few
    pixels would otherwise wash out everything else. Normalising to a high
    percentile (not the raw maximum) keeps the dim receding side visible.
    """
    lit = img[img > 0]
    hi = np.percentile(lit, 99.5) if lit.size else 1.0
    rgba = matplotlib.colormaps[CMAP](np.clip(img / hi, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def main() -> None:
    exe = find_exe()
    if not os.path.exists(exe):
        sys.exit(f"bhsim not found at {exe!r}; build it first or pass the path as arg 1.")
    fig_dir = os.path.join("simulation", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if "--sweep" in sys.argv:
        spins = np.linspace(0.0, 0.99, 24)
        frames = [colorize(render(exe, a, res=200)) for a in spins]
        frames += [frames[-1]] * 12  # hold the final frame
        out_path = os.path.join(fig_dir, "raytrace_spin.gif")
        imageio.mimsave(out_path, frames, fps=12, loop=0)
        print(f"wrote {out_path} ({len(frames)} frames)")
        return

    for a, name in [(0.0, "raytrace_schwarzschild.png"), (0.9, "raytrace_kerr.png")]:
        img = colorize(render(exe, a, res=RES))
        out_path = os.path.join(fig_dir, name)
        imageio.imwrite(out_path, img)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
