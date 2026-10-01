"""Camera flythrough: render a looping orbit around the hole and encode a clip.

Imported by the Kaggle notebook; also runnable standalone for a quick test.
"""
import math

import gpu_render as G
import imageio.v2 as imageio
import numpy as np
import torch


def camera_at(t, cfg):
    """Camera dict at loop phase t in [0, 1). All motion is periodic so the clip
    loops seamlessly: a full azimuth orbit, with a gentle inclination sway and a
    slow dolly in and out."""
    az = 2 * math.pi * t
    incl = math.radians(cfg["incl0"] + cfg["incl_sway"] * math.sin(2 * math.pi * t))
    dist = cfg["dist0"] + cfg["dolly"] * math.cos(2 * math.pi * t)
    return dict(a=cfg["a"], incl=incl, az=az, dist=dist,
                fov=math.radians(cfg["fov"]), res=cfg["res"],
                r_in=cfg["r_in"], r_out=cfg["r_out"])


def render_flythrough(cfg, device, out_mp4="flythrough.mp4", out_gif=None,
                      progress=print):
    sky = G.make_starfield(device=device)
    lut = G.inferno_lut(device)
    a = cfg["a"]

    # Calibrate disk brightness once, from a frame where the beamed side is in
    # view, so brightness is stable across the whole clip (no flicker).
    cam0 = camera_at(0.0, cfg)
    _, bright, otype = G.render_frame(cam0, a, sky, lut, 1.0, device,
                                      n_steps=cfg["n_steps"], return_raw=True)
    disk_b = bright[otype == 2]
    scale = torch.quantile(disk_b, 0.995).item() if disk_b.numel() else 1.0
    scale = max(scale, 1e-3)
    progress(f"disk_scale = {scale:.3f}")

    frames = []
    n = cfg["n_frames"]
    for i in range(n):
        cam = camera_at(i / n, cfg)
        f = G.render_frame(cam, a, sky, lut, scale, device, n_steps=cfg["n_steps"])
        frames.append((f.cpu().numpy() * 255).astype(np.uint8))
        if (i + 1) % max(1, n // 20) == 0 or i == n - 1:
            progress(f"frame {i + 1}/{n}")

    imageio.mimsave(out_mp4, frames, fps=cfg["fps"], quality=8, macro_block_size=None)
    progress(f"wrote {out_mp4}")
    if out_gif:
        small = [f[::2, ::2] for f in frames]  # half-size for a lighter gif
        imageio.mimsave(out_gif, small, fps=cfg["fps"], loop=0)
        progress(f"wrote {out_gif}")
    return frames


DEFAULTS = dict(
    a=0.9, incl0=78.0, incl_sway=10.0, dist0=55.0, dolly=12.0, fov=22.0,
    r_in=6.0, r_out=20.0, res=(576, 1024), n_frames=144, n_steps=1400, fps=30,
)
