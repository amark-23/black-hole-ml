"""Data for the emission -> image operator (the FNO task), built on the Kerr
lensing dataset written by ml/datagen/lensing.py.

That dataset stores each black hole as the geometry of its light rays: for every
pixel, what the ray hit (`otype`), where it crossed the disk (`hit_r`, `hit_ph`)
and the redshift factor there (`hit_g`). Any disk emission profile E(r, phi) then
gives the observed image directly,

    I = g^4 E(hit_r, hit_ph)   on disk pixels, 0 on the shadow and the sky,

so one trace yields unlimited (emission, image) pairs.

The operator's input is the same emission as the camera would see it with gravity
switched off: a straight-line projection of the disk onto the same image grid.
Input and output then share one image domain at every resolution, and the network
learns exactly what gravity adds: the bending, the far side lifted over the
shadow, the photon ring, the shadow itself, and the g^4 Doppler beaming.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

GEOMETRY = ("otype", "hit_r", "hit_ph", "hit_g")
N_CHANNELS = 5          # [emission seen with gravity off, spin, inclination, x, y]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass
class LensingSet:
    """Ray geometry of a set of black holes at one resolution (numpy, on the CPU)."""
    otype: np.ndarray                   # (n, N, N) int8: 0 unfinished, 1 horizon, 2 disk, 3 sky
    hit_r: np.ndarray                   # (n, N, N) float16
    hit_ph: np.ndarray                  # (n, N, N) float16
    hit_g: np.ndarray                   # (n, N, N) float16
    params: np.ndarray                  # (n, 2) float32: spin a, inclination in degrees
    split: np.ndarray                   # (n,) "train" / "val" / "test"
    index: np.ndarray                   # (n,) hole index in the full set (same at every res)
    manifest: dict = field(default_factory=dict)

    @property
    def res(self) -> int:
        return self.otype.shape[-1]

    def __len__(self) -> int:
        return len(self.index)

    def subset(self, split: str) -> "LensingSet":
        keep = self.split == split
        return LensingSet(self.otype[keep], self.hit_r[keep], self.hit_ph[keep],
                          self.hit_g[keep], self.params[keep], self.split[keep],
                          self.index[keep], self.manifest)


def load_lensing(path: str) -> LensingSet:
    """Load every shard under `path` (searched recursively, so a Kaggle input
    folder works as is), keeping only the geometry the operator needs."""
    files = sorted(glob.glob(os.path.join(path, "**", "shard_*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no shard_*.npz under {path}")
    manifests = glob.glob(os.path.join(path, "**", "manifest.json"), recursive=True)
    manifest = json.load(open(manifests[0])) if manifests else {}
    parts: dict[str, list] = {k: [] for k in GEOMETRY + ("params", "split", "index")}
    start = 0
    for f in files:
        z = np.load(f)
        n = len(z["params"])
        for k in GEOMETRY + ("params", "split"):
            parts[k].append(z[k])
        # Shards written before holes carried their index list them in order.
        parts["index"].append(z["index"] if "index" in z.files else np.arange(start, start + n))
        start += n
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return LensingSet(cat["otype"], cat["hit_r"], cat["hit_ph"], cat["hit_g"],
                      cat["params"].astype(np.float32), cat["split"].astype(str),
                      cat["index"].astype(np.int64), manifest)


# --------------------------------------------------------------------------- #
# Geometry: the ISCO and the straight-line ("gravity off") view of the disk
# --------------------------------------------------------------------------- #

def isco_radius(a):
    """Prograde ISCO for spin a (M = 1); same formula as gpu_render.isco_radius,
    vectorized over numpy arrays or torch tensors."""
    lib = torch if isinstance(a, torch.Tensor) else np
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = lib.sqrt(3 * a * a + z1 * z1)
    return 3 + z2 - lib.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def pixel_grid(res: int, device=None):
    """Image-plane coordinates in [-1, 1], laid out exactly like the ray tracer's
    pixels (gpu_render.camera_rays): x left to right, y top (+1) to bottom (-1)."""
    xs = torch.linspace(-1.0, 1.0, res, device=device)
    ys = torch.linspace(1.0, -1.0, res, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")          # each (res, res)
    return gx, gy


def flat_disk_view(incl_deg: torch.Tensor, res: int, dist: float, fov_deg: float,
                   az_deg: float = 0.0):
    """Where each pixel's straight line of sight meets the equatorial plane.

    The same camera as the ray tracer (gpu_render.camera_rays), but with gravity
    switched off. Returns r and phi of the crossing, and whether there is one in
    front of the camera, each (B, res, res).
    """
    dev = incl_deg.device
    i = torch.deg2rad(incl_deg.float())                      # (B,)
    az = math.radians(az_deg)
    C = torch.stack([dist * torch.sin(i) * math.cos(az),
                     dist * torch.sin(i) * math.sin(az),
                     dist * torch.cos(i)], dim=-1)            # (B, 3) camera position
    f = -C / C.norm(dim=-1, keepdim=True)
    up_w = torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(f)
    right = torch.linalg.cross(f, up_w, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True)
    up = torch.linalg.cross(right, f, dim=-1)
    tanh = math.tan(0.5 * math.radians(fov_deg))
    gx, gy = pixel_grid(res, dev)
    d = (f[:, None, None, :] + (gx * tanh)[None, ..., None] * right[:, None, None, :]
         + (gy * tanh)[None, ..., None] * up[:, None, None, :])
    d = d / d.norm(dim=-1, keepdim=True)                     # (B, res, res, 3)
    t = -C[:, None, None, 2] / d[..., 2]                     # distance to the plane z = 0
    hit = t > 0
    t = torch.where(hit, t, torch.zeros_like(t))
    qx = C[:, None, None, 0] + t * d[..., 0]
    qy = C[:, None, None, 1] + t * d[..., 1]
    return torch.hypot(qx, qy), torch.atan2(qy, qx), hit


# --------------------------------------------------------------------------- #
# Emission profiles
# --------------------------------------------------------------------------- #

@dataclass
class Emissions:
    """A batch of random disk emission profiles

        E(r, phi) = (r_in / r)^p * exp( sum_j amp_j cos(m_j phi + k_j ln(r / r_in) + phase_j) )

    smooth, positive and 2 pi-periodic in phi: a brighter inner disk with
    streaks, spiral arms and hot spots on top."""
    p: torch.Tensor          # (B,)
    m: torch.Tensor          # (B, J) integer azimuthal wavenumbers
    k: torch.Tensor          # (B, J) radial wavenumbers in ln r
    amp: torch.Tensor        # (B, J)
    phase: torch.Tensor      # (B, J)

    def to(self, device) -> "Emissions":
        return Emissions(*(t.to(device) for t in (self.p, self.m, self.k, self.amp, self.phase)))

    def take(self, idx) -> "Emissions":
        return Emissions(*(t[idx] for t in (self.p, self.m, self.k, self.amp, self.phase)))

    def __call__(self, r: torch.Tensor, phi: torch.Tensor, r_in: torch.Tensor) -> torch.Tensor:
        """Evaluate at (B, ...) radii and angles; r_in is (B,)."""
        B, ones = r.shape[0], (1,) * (r.dim() - 1)
        per_sample = lambda t: t.view(B, *ones)                  # noqa: E731
        per_wave = lambda t: t.view(B, *ones, -1)                # noqa: E731
        rr = torch.clamp(r / per_sample(r_in), min=1e-3)
        arg = (per_wave(self.m) * phi[..., None] + per_wave(self.k) * torch.log(rr)[..., None]
               + per_wave(self.phase))
        tex = (per_wave(self.amp) * torch.cos(arg)).sum(-1)
        return rr ** (-per_sample(self.p)) * torch.exp(tex)


def random_emissions(n: int, generator: torch.Generator, n_waves: int = 6,
                     m_max: int = 8, k_max: float = 8.0) -> Emissions:
    """n random profiles. Wavenumbers stay low enough to be resolved at 64x64."""
    g = generator
    p = 1.5 + 1.5 * torch.rand(n, generator=g)
    m = torch.randint(0, m_max + 1, (n, n_waves), generator=g).float()
    k = (2 * torch.rand(n, n_waves, generator=g) - 1) * k_max
    u = torch.randn(n, n_waves, generator=g)
    sigma = torch.rand(n, 1, generator=g)                    # texture strength, 0 = smooth
    amp = sigma * u / u.norm(dim=1, keepdim=True)
    phase = 2 * math.pi * torch.rand(n, n_waves, generator=g)
    return Emissions(p, m, k, amp, phase)


def fixed_emissions(n_holes: int, seed: int = 1234) -> Emissions:
    """One fixed profile per hole index, for validation and testing: the same
    function is used at every resolution, so errors are directly comparable."""
    return random_emissions(n_holes, torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #

def build_inputs(a: torch.Tensor, incl: torch.Tensor, emis: Emissions, res: int,
                 manifest: dict | None = None):
    """Operator inputs for black holes (a, incl) and emission profiles `emis`: no
    ray tracing needed, which is the point of the operator.

    Returns x (B, N, N, 5) = [emission seen with gravity off, spin, inclination,
    x, y], and the per-sample scale the emission channel was divided by (its peak),
    so every sample is O(1); multiply the model's output by it for the true image.
    """
    man = manifest or {}
    dist, fov, az = man.get("dist", 40.0), man.get("fov", 30.0), man.get("az", 0.0)
    r_out = float(man.get("r_out", 18.0))
    device = a.device
    emis = emis.to(device)
    r_in = isco_radius(a)
    r_f, ph_f, front = flat_disk_view(incl, res, dist, fov, az)
    on_flat_disk = front & (r_f >= r_in[:, None, None]) & (r_f <= r_out)
    x_field = torch.where(on_flat_disk, emis(r_f, ph_f, r_in), torch.zeros_like(r_f))
    scale = x_field.flatten(1).amax(1).clamp(min=1e-12)
    B = len(a)
    gx, gy = pixel_grid(res, device)
    x = torch.stack([
        x_field / scale[:, None, None],
        (2 * a - 1)[:, None, None].expand(B, res, res),          # spin 0..0.99 -> about -1..1
        ((incl - 50) / 35)[:, None, None].expand(B, res, res),  # 15..85 deg -> -1..1
        gx.expand(B, res, res),
        gy.expand(B, res, res),
    ], dim=-1)
    return x, scale


def observed_images(ds: LensingSet, rows, emis: Emissions, device, g_pow: float = 4.0):
    """The images a telescope would see for the holes ds[rows], one emission
    profile per row in `emis`: g^4 E(hit_r, hit_ph) on disk pixels, 0 on the shadow
    and the sky. (B, N, N), in the emission's own (arbitrary) brightness units."""
    rows = np.asarray(rows)

    def as_t(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    a = as_t(ds.params[rows, 0]).float()
    otype = as_t(ds.otype[rows])
    hit_r = as_t(ds.hit_r[rows]).float()
    hit_ph = as_t(ds.hit_ph[rows]).float()
    hit_g = as_t(ds.hit_g[rows]).float()
    return torch.where(otype == 2,
                       emis.to(device)(hit_r, hit_ph, isco_radius(a))
                       * hit_g.clamp(min=0) ** g_pow,
                       torch.zeros_like(hit_r))


def make_batch(ds: LensingSet, rows, emis: Emissions, device, g_pow: float = 4.0):
    """Inputs, targets and per-sample scales for the holes ds[rows], with one
    emission profile per row in `emis` (on any device).

    x: (B, N, N, 5) see build_inputs
    y: (B, N, N, 1) the observed image (observed_images), divided by the same
                    per-sample scale as the input (the true image is y * scale)
    """
    rows = np.asarray(rows)
    a = torch.from_numpy(ds.params[rows, 0]).float().to(device)
    incl = torch.from_numpy(ds.params[rows, 1]).float().to(device)
    x, scale = build_inputs(a, incl, emis, ds.res, ds.manifest)
    y = observed_images(ds, rows, emis, device, g_pow)
    return x, (y / scale[:, None, None])[..., None], scale
