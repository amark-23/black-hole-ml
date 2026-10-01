"""Generate the Kerr lensing dataset: per-pixel light-bending maps for an FNO and
for the inverse problem.

The GPU ray tracer is run over a grid of black holes (spin a, viewing
inclination) on a FIXED image grid, and for each one we store the fields that say
how that hole bends light: where each pixel's ray lands on the disk, the redshift
there, and where escaping rays point on the sky. Those fields ARE the lensing
operator.

Each black hole is traced once. Downstream you push any disk-emission field
through the stored (hit_r, hit_phi) to get an observed image (emission_to_image
below), so a single trace yields unlimited (emission -> image) pairs: emission
becomes free data augmentation for the FNO, and the inverse problem reads the
fields (or images made from them) back to the parameters.

Runs on a GPU (Kaggle T4); falls back to CPU for a small smoke test.
"""
import json
import math
import os
import time

import gpu_render as G
import numpy as np
import torch

# float16 per-pixel fields stored per sample; otype is stored separately as int8.
CHANNELS = ["hit_r", "hit_ph", "hit_g", "sky_th", "sky_ph"]

# Fixed framing so every sample shares one image domain (an FNO operates on a
# fixed grid); only the physics (a, inclination) varies.
DIST = 40.0
FOV = 30.0
AZ = 0.0


def make_cam(a, incl_deg, res, r_out):
    """Camera dict for one black hole; inner disk edge follows the ISCO."""
    return dict(a=a, incl=math.radians(incl_deg), az=math.radians(AZ),
                dist=DIST, fov=math.radians(FOV), res=res,
                r_in=G.isco_radius(a), r_out=r_out)


def sample_params(n, seed, a_range=(0.0, 0.99), incl_range=(15.0, 85.0)):
    """Uniformly sample (spin, inclination) for n black holes."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(*a_range, n)
    incl = rng.uniform(*incl_range, n)
    return np.stack([a, incl], axis=1).astype(np.float32)


def assign_splits(n, seed):
    """Deterministic 80/10/10 train/val/test labels."""
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    split = np.empty(n, dtype="<U5")
    split[perm[: int(0.8 * n)]] = "train"
    split[perm[int(0.8 * n): int(0.9 * n)]] = "val"
    split[perm[int(0.9 * n):]] = "test"
    return split


def generate(out_dir, n=2000, res=128, r_out=18.0, seed=0, shard=250,
             device=None, n_steps=1500, C0=0.013, use_compile=True, progress=print):
    """Trace n black holes and write npz shards plus a manifest to out_dir."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    params = sample_params(n, seed)
    split = assign_splits(n, seed)
    H = W = res

    t0 = time.time()
    shards = []
    for s0 in range(0, n, shard):
        s1 = min(s0 + shard, n)
        otype = np.zeros((s1 - s0, H, W), np.int8)
        ch = {c: np.zeros((s1 - s0, H, W), np.float16) for c in CHANNELS}
        for i in range(s0, s1):
            a, incl = float(params[i, 0]), float(params[i, 1])
            tr = G.trace_frame(make_cam(a, incl, (H, W), r_out), a, device,
                               n_steps=n_steps, C0=C0, use_compile=use_compile)
            otype[i - s0] = tr["otype"].reshape(H, W).cpu().numpy().astype(np.int8)
            for c in CHANNELS:
                ch[c][i - s0] = tr[c].reshape(H, W).cpu().numpy().astype(np.float16)
            if (i + 1) % max(1, n // 20) == 0 or i == n - 1:
                progress(f"  {i + 1}/{n}   {(time.time() - t0) / (i + 1):.2f}s/sample")
        fn = os.path.join(out_dir, f"shard_{s0:05d}.npz")
        np.savez_compressed(fn, otype=otype, params=params[s0:s1], split=split[s0:s1], **ch)
        shards.append(os.path.basename(fn))
        progress(f"wrote {fn}")

    manifest = dict(
        name="kerr_lensing", n=n, res=res, dist=DIST, fov=FOV, az=AZ, r_out=r_out,
        r_in="ISCO(a)", param_names=["a", "incl_deg"],
        a_range=[0.0, 0.99], incl_range=[15.0, 85.0],
        channels_f16=CHANNELS, otype={"0": "fly", "1": "horizon", "2": "disk", "3": "sky"},
        shards=shards, seed=seed,
        note="push a disk emission field through (hit_r, hit_ph) to make an "
             "observed image; see emission_to_image in lensing.py",
    )
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    progress(f"done: {len(shards)} shard(s) + manifest in {out_dir} "
             f"({(time.time() - t0) / 60:.1f} min)")
    return manifest


def emission_to_image(otype, hit_r, hit_ph, emission_fn):
    """Form the observed intensity map for one sample by pushing a disk emission
    field through its stored geometry. emission_fn(r, phi) -> intensity (arrays).
    Shadow and sky pixels are 0. This is the forward operator the FNO learns and
    the inverse problem inverts."""
    disk = otype == 2
    img = np.zeros(hit_r.shape, np.float32)
    if disk.any():
        img[disk] = emission_fn(hit_r[disk].astype(np.float32),
                                hit_ph[disk].astype(np.float32))
    return img
