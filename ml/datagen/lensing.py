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

Many holes are traced TOGETHER in one batch so the GPU is actually busy. One hole
is only ~res*res rays, far too few to fill a T4 (it sits near-idle and the CPU
loop dominates). Concatenating ~1M rays' worth of holes per call, with the spin
and disk radii carried per ray, keeps the device saturated. Runs on a GPU; falls
back to CPU for a small smoke test.
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


def trace_batch(holes, res, r_out, device, n_steps=6000, C0=0.013, use_compile=True,
                check_every=32, compact_below=0.85):
    """Trace several black holes at once and return the per-ray fields for the whole
    batch (one long [B*res*res] tensor per field, holes laid out back to back).

    The spin and the horizon / disk radii are carried as per-ray tensors, so every
    hole in the batch rides the same fused, compiled step; that is what fills the
    GPU. Mirrors trace_frame's loop (pole reflection, horizon / disk / sky tests,
    compaction), generalised to per-ray parameters.
    """
    H = W = res
    M = 1.0
    step = G.get_step_fn(use_compile)

    Ys, a_l, rcap_l, rin_l, rout_l, resc_l = [], [], [], [], [], []
    for a, incl in holes:
        cam = make_cam(a, incl, (H, W), r_out)
        Y = G.camera_rays(cam, device)
        P = Y.shape[0]
        rp = 1.01 * (M + math.sqrt(max(M * M - a * a, 0.0)))
        Ys.append(Y)
        a_l.append(torch.full((P,), float(a), device=device))
        rcap_l.append(torch.full((P,), rp, device=device))
        rin_l.append(torch.full((P,), float(cam["r_in"]), device=device))
        rout_l.append(torch.full((P,), float(cam["r_out"]), device=device))
        resc_l.append(torch.full((P,), cam["dist"] * 1.4, device=device))
    Y = torch.cat(Ys, 0)
    a_all, r_cap, r_in, r_out_t, r_esc = (torch.cat(a_l), torch.cat(rcap_l),
                                          torch.cat(rin_l), torch.cat(rout_l), torch.cat(resc_l))
    N = Y.shape[0]

    names = ("otype", "hit_r", "hit_ph", "hit_g", "sky_th", "sky_ph")
    out = dict(otype=torch.zeros(N, dtype=torch.int8, device=device),
               hit_r=torch.zeros(N, device=device), hit_ph=torch.zeros(N, device=device),
               hit_g=torch.zeros(N, device=device), sky_th=torch.zeros(N, device=device),
               sky_ph=torch.zeros(N, device=device))
    work = [out[n].clone() for n in names]
    active = torch.ones(N, dtype=torch.bool, device=device)
    idx = None

    def write_back():
        for n, w in zip(names, work):
            if idx is None:
                out[n].copy_(w)
            else:
                out[n][idx] = w

    for i in range(n_steps):
        if i > 0 and i % check_every == 0:
            n_active = int(active.sum())
            if n_active <= max(1, N // 2000):
                break
            if n_active < compact_below * active.shape[0]:
                write_back()
                keep = active.nonzero().squeeze(1)
                idx = keep if idx is None else idx[keep]
                Y, active = Y[keep], active[keep]
                work = [w[keep] for w in work]
                a_all, r_cap, r_in, r_out_t, r_esc = (a_all[keep], r_cap[keep], r_in[keep],
                                                      r_out_t[keep], r_esc[keep])
        Y, active, *work = step(Y, active, *work, r_esc, a_all, M, C0, r_cap, r_in, r_out_t)
    write_back()
    return out


def generate(out_dir, n=2000, res=128, r_out=18.0, seed=0, shard=250, batch=None,
             device=None, n_steps=6000, C0=0.013, use_compile=True, progress=print):
    """Trace n black holes (in GPU-saturating batches) and write npz shards plus a
    manifest to out_dir. `batch` is how many holes share one GPU call; the default
    aims for about a million rays per call."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    if batch is None:
        batch = max(1, 1_000_000 // (res * res))
    params = sample_params(n, seed)
    split = assign_splits(n, seed)
    H = W = res
    P = H * W

    t0 = time.time()
    shards = []
    for s0 in range(0, n, shard):
        s1 = min(s0 + shard, n)
        otype = np.zeros((s1 - s0, H, W), np.int8)
        ch = {c: np.zeros((s1 - s0, H, W), np.float16) for c in CHANNELS}
        for b0 in range(s0, s1, batch):
            b1 = min(b0 + batch, s1)
            holes = [(float(params[i, 0]), float(params[i, 1])) for i in range(b0, b1)]
            out = trace_batch(holes, res, r_out, device, n_steps=n_steps, C0=C0,
                              use_compile=use_compile)
            for j, i in enumerate(range(b0, b1)):
                sl = slice(j * P, (j + 1) * P)
                otype[i - s0] = out["otype"][sl].reshape(H, W).cpu().numpy().astype(np.int8)
                for c in CHANNELS:
                    ch[c][i - s0] = out[c][sl].reshape(H, W).cpu().numpy().astype(np.float16)
            progress(f"  {b1}/{n}   {(time.time() - t0) / b1:.2f}s/hole")
        fn = os.path.join(out_dir, f"shard_{s0:05d}.npz")
        np.savez_compressed(fn, otype=otype, params=params[s0:s1], split=split[s0:s1], **ch)
        shards.append(os.path.basename(fn))
        progress(f"wrote {fn}")

    manifest = dict(
        name="kerr_lensing", n=n, res=res, dist=DIST, fov=FOV, az=AZ, r_out=r_out,
        r_in="ISCO(a)", param_names=["a", "incl_deg"],
        a_range=[0.0, 0.99], incl_range=[15.0, 85.0], batch=batch,
        channels_f16=CHANNELS, otype={"0": "fly", "1": "horizon", "2": "disk", "3": "sky"},
        shards=shards, seed=seed,
        note="push a disk emission field through (hit_r, hit_ph) to make an "
             "observed image; see emission_to_image in lensing.py",
    )
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    progress(f"done: {len(shards)} shard(s) + manifest in {out_dir} "
             f"({(time.time() - t0) / 60:.1f} min)")
    return manifest


def emission_to_image(otype, hit_r, hit_ph, emission_fn, hit_g=None, g_pow=4.0):
    """Form the observed intensity map for one sample by pushing a disk emission
    field through its stored geometry. emission_fn(r, phi) -> intensity (arrays),
    as emitted in the gas's own frame. Pass the sample's hit_g to include the
    relativistic shift: the observed (bolometric) intensity is g^4 times the
    emitted one, so gas moving toward the camera is boosted and gas moving away is
    dimmed. Without hit_g the map is purely geometric, with no Doppler or
    gravitational shift. Shadow and sky pixels are 0. This is the forward operator
    the FNO learns and the inverse problem inverts."""
    disk = otype == 2
    img = np.zeros(hit_r.shape, np.float32)
    if disk.any():
        e = emission_fn(hit_r[disk].astype(np.float32), hit_ph[disk].astype(np.float32))
        if hit_g is not None:
            e = e * hit_g[disk].astype(np.float32) ** g_pow
        img[disk] = e
    return img
