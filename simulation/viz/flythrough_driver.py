"""Spinning black hole: render a looping clip of a Kerr hole and its disk.

By default the camera holds still and the accretion disk turns: the gas orbits
at its Keplerian rate (inner edge fastest), so the clip shows the hole's disk
spinning, with the Doppler-bright side, the lensed far side arching over the
shadow, and the photon ring. Because nothing about the light paths changes when
only the gas moves, the rays are traced ONCE and every frame is just re-shaded,
so the whole clip costs about one frame of ray tracing.

Set orbit_turns to a non-zero value to also swing the camera slowly around the
hole (no zoom); then every frame is traced.

Imported by the Kaggle notebook; also runnable standalone for a quick test.
"""
import math
import time

import gpu_render as G
import imageio.v2 as imageio
import numpy as np
import torch


def camera_at(t, cfg):
    """Camera dict at loop phase t in [0, 1). Fixed distance and inclination; the
    azimuth advances by orbit_turns full turns per loop (0 = a still camera)."""
    a = cfg["a"]
    r_in = cfg["r_in"] if cfg.get("r_in") else G.isco_radius(a)
    return dict(a=a, incl=math.radians(cfg["incl"]),
                az=math.radians(cfg.get("az", 0.0)) + 2 * math.pi * cfg.get("orbit_turns", 0) * t,
                dist=cfg["dist"], fov=math.radians(cfg["fov"]), res=cfg["res"],
                r_in=r_in, r_out=cfg["r_out"])


def _to_uint8(img):
    return (img.cpu().numpy() * 255).astype(np.uint8)


def render_clip(cfg, device, out_mp4="black_hole.mp4", out_gif="black_hole.gif",
                progress=print):
    """Render the looping clip and write an mp4 and/or a gif. Returns the frames."""
    sky = G.make_starfield(device=device)
    lut = G.fire_lut(device)
    tex = G.DiskTexture(device=device)
    look = dict(G.DEFAULT_LOOK, **cfg.get("look", {}))
    a = cfg["a"]
    opts = dict(n_steps=cfg["n_steps"], C0=cfg["C0"], use_compile=cfg.get("use_compile", True))
    still = not cfg.get("orbit_turns", 0)
    n = cfg["n_frames"]

    # The first trace also pays the one-off torch.compile cost (tens of seconds).
    t0 = time.time()
    tr = G.trace_frame(camera_at(0.0, cfg), a, device, **opts)
    scale = G.disk_scale(tr, look)        # fixed for the whole clip: no flicker
    mode = "compiled" if (opts["use_compile"] and G._STEP["compiled"]) else "eager"
    progress(f"traced {tr['res'][1]}x{tr['res'][0]} in {time.time() - t0:.1f}s "
             f"({mode}; the first trace of a session includes compiling)")

    frames = []
    t_start = time.time()
    for i in range(n):
        t = i / n
        if not still and i > 0:
            tr = G.trace_frame(camera_at(t, cfg), a, device, **opts)
        frames.append(_to_uint8(G.shade(tr, sky, lut, scale, look, tex, t)))
        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n - i - 1)
            progress(f"frame {i + 1}/{n}  {elapsed / (i + 1):.2f}s/frame  "
                     f"elapsed {elapsed / 60:.1f} min  eta {eta / 60:.1f} min")

    fps = cfg["fps"]
    if out_mp4:
        imageio.mimsave(out_mp4, frames, fps=fps, quality=8, macro_block_size=None)
        progress(f"wrote {out_mp4}")
    if out_gif:
        step = max(1, round(frames[0].shape[1] / cfg.get("gif_width", 640)))
        small = [f[::step, ::step] for f in frames]   # gifs get big fast; downsample
        imageio.mimsave(out_gif, small, duration=1000 / fps, loop=0)
        progress(f"wrote {out_gif} ({small[0].shape[1]}x{small[0].shape[0]})")
    return frames


# a: spin (0..1). incl: viewing angle from the spin axis (90 = edge-on).
# dist / fov: how close and wide the shot is. r_in: inner disk edge (None = ISCO).
# spin_turns (in look): how many turns the inner edge makes per loop.
DEFAULTS = dict(
    a=0.9, incl=84.0, az=0.0, dist=40.0, fov=30.0, r_in=None, r_out=18.0,
    res=(720, 1280), n_frames=96, fps=24, orbit_turns=0, n_steps=1500, C0=0.013,
    use_compile=True, gif_width=640, look={},
)
