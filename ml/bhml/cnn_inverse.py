"""Read a black hole's spin and inclination off its image, and find how much blur
and noise it takes before that fails.

Driven by ml/notebooks/cnn_inverse_kaggle.ipynb on a GPU. The pieces:

  degrade          a telescope's view of an image: Gaussian blur (the beam), then
                   Gaussian noise (the detector), in the image's own units
  prepare          what the network sees: brightness relative to the image's own
                   bright level (real data has no absolute luminosity), an asinh
                   stretch so the faint disk and the beamed spot both register,
                   and the pixel coordinates
  train_model      fit a ParamCNN on clean images, or on randomly degraded ones
  predict / maes   spin and inclination errors at a given degradation
  failure_level    the degradation at which the error reaches half of guessing
  plot_*           the figures for the README

Blur is measured as the beam's full width at half maximum over the diameter of
the black hole's shadow, so it means the same thing at any resolution; the Event
Horizon Telescope's view of M87* is about 0.5 on this scale.
"""

from __future__ import annotations

import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bhml.lensing_data import Emissions, LensingSet, observed_images, pixel_grid, random_emissions
from bhml.models import ParamCNN, count_params

A_RANGE = (0.0, 0.99)            # spin range of the dataset
I_RANGE = (15.0, 85.0)           # inclination range, degrees
EHT_BLUR = 0.5                   # beam FWHM / shadow diameter for the EHT image of M87*
FWHM = 2 * math.sqrt(2 * math.log(2))

DEFAULTS = dict(
    width=32, n_stages=5,
    epochs=120, batch_size=64, lr=1e-3, weight_decay=1e-4, lr_step=40, lr_gamma=0.5,
    seed=0, eval_every=5,
    # augmentation (the "degraded" model): each is applied with probability p_aug
    p_aug=0.6, max_blur=1.0, max_noise=0.3,
)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #

def normalize_params(p: torch.Tensor) -> torch.Tensor:
    """(B, 2) spin, inclination -> both scaled to [-1, 1]."""
    lo = torch.tensor([A_RANGE[0], I_RANGE[0]], device=p.device)
    hi = torch.tensor([A_RANGE[1], I_RANGE[1]], device=p.device)
    return 2 * (p - lo) / (hi - lo) - 1


def denormalize_params(z: torch.Tensor) -> torch.Tensor:
    lo = torch.tensor([A_RANGE[0], I_RANGE[0]], device=z.device)
    hi = torch.tensor([A_RANGE[1], I_RANGE[1]], device=z.device)
    return lo + (z + 1) * (hi - lo) / 2


# --------------------------------------------------------------------------- #
# Degradations and the network's input
# --------------------------------------------------------------------------- #

def shadow_diameter_px(manifest: dict, res: int) -> float:
    """The shadow's diameter in pixels (Schwarzschild value, 2 sqrt(27) M; spin
    changes it by under 10%), for the dataset's camera framing."""
    dist, fov = manifest.get("dist", 40.0), manifest.get("fov", 30.0)
    width_m = 2 * dist * math.tan(0.5 * math.radians(fov))   # image width at the hole, in M
    return 2 * math.sqrt(27.0) * res / width_m


def gaussian_blur(img: torch.Tensor, sigma_px) -> torch.Tensor:
    """Blur (B, N, N) images with a Gaussian beam, one width (pixels) per image;
    a width of 0 leaves the image unchanged. Light blurred past the frame is lost,
    as it would be for a real detector."""
    B = img.shape[0]
    sigma = torch.as_tensor(sigma_px, dtype=img.dtype, device=img.device).reshape(-1).expand(B)
    smax = float(sigma.max())
    if smax <= 0:
        return img
    r = int(math.ceil(3 * smax))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x[None] / sigma.clamp(min=1e-6)[:, None]) ** 2)
    k = torch.where(sigma[:, None] > 0, k, (x[None] == 0).to(img.dtype))   # a delta for 0
    k = k / k.sum(1, keepdim=True)
    t = img[None]                                                           # (1, B, N, N)
    t = F.conv2d(F.pad(t, (r, r, 0, 0)), k[:, None, None, :], groups=B)
    t = F.conv2d(F.pad(t, (0, 0, r, r)), k[:, None, :, None], groups=B)
    return t[0]


def degrade(img: torch.Tensor, blur, noise, shadow_px: float,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """A telescope's view of (B, N, N) images: blur with a beam of FWHM = blur x
    the shadow's diameter, then add Gaussian noise with a standard deviation of
    noise x the blurred image's peak. blur and noise are scalars or (B,) tensors."""
    sigma = torch.as_tensor(blur, dtype=img.dtype, device=img.device) * shadow_px / FWHM
    out = gaussian_blur(img, sigma)
    noise = torch.as_tensor(noise, dtype=img.dtype, device=img.device).reshape(-1).expand(len(img))
    if float(noise.max()) > 0:
        peak = out.flatten(1).amax(1)
        eps = torch.randn(out.shape, generator=generator).to(out.device, out.dtype)
        out = out + (noise * peak)[:, None, None] * eps
    return out


def prepare(img: torch.Tensor, soft: float = 0.05) -> torch.Tensor:
    """The network's input from (B, N, N) images: brightness relative to each
    image's own bright level (the 99.5th percentile, robust to noise; an absolute
    luminosity is not observable), an asinh stretch, and pixel coordinates.
    Returns (B, N, N, 3)."""
    B, N, _ = img.shape
    level = torch.quantile(img.flatten(1), 0.995, dim=1).clamp(min=1e-12)
    rel = img / level[:, None, None]
    x = torch.asinh(rel / soft) / math.asinh(1 / soft)
    gx, gy = pixel_grid(N, img.device)
    return torch.stack([x, gx.expand(B, N, N), gy.expand(B, N, N)], dim=-1)


def _degradation_draw(n: int, cfg: dict, generator: torch.Generator):
    """Random per-image blur and noise for augmented training."""
    on_b = torch.rand(n, generator=generator) < cfg["p_aug"]
    on_n = torch.rand(n, generator=generator) < cfg["p_aug"]
    blur = torch.where(on_b, torch.rand(n, generator=generator) * cfg["max_blur"],
                       torch.zeros(n))
    noise = torch.where(on_n, torch.rand(n, generator=generator) * cfg["max_noise"],
                        torch.zeros(n))
    return blur, noise


def make_batch(ds: LensingSet, rows, emis: Emissions, device, blur=0.0, noise=0.0,
               generator: torch.Generator | None = None):
    """Network inputs (B, N, N, 3) and normalized targets (B, 2) for ds[rows]."""
    rows = np.asarray(rows)
    img = observed_images(ds, rows, emis, device)
    shadow = shadow_diameter_px(ds.manifest, ds.res)
    if isinstance(blur, torch.Tensor):
        blur = blur.to(device)
    if isinstance(noise, torch.Tensor):
        noise = noise.to(device)
    x = prepare(degrade(img, blur, noise, shadow, generator))
    y = normalize_params(torch.from_numpy(ds.params[rows]).float().to(device))
    return x, y


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #

def _emis_for(ds: LensingSet, rows, emis_fixed: Emissions) -> Emissions:
    return emis_fixed.take(torch.from_numpy(ds.index[np.asarray(rows)]))


@torch.no_grad()
def predict(model, ds: LensingSet, emis_fixed: Emissions, device, blur=0.0, noise=0.0,
            seed: int = 0, batch_size: int = 64, val_cfg: dict | None = None):
    """Predicted and true (spin, inclination), each (n, 2), for every hole of ds at
    one degradation (or, with val_cfg, a fixed random draw of degradations).
    Predictions are clipped to the dataset's ranges, so a failing model can do no
    worse than a guess inside them."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    preds = []
    for b in range(0, len(ds), batch_size):
        rows = np.arange(b, min(b + batch_size, len(ds)))
        bl, no = (_degradation_draw(len(rows), val_cfg, gen) if val_cfg else (blur, noise))
        x, _ = make_batch(ds, rows, _emis_for(ds, rows, emis_fixed), device, bl, no, gen)
        preds.append(denormalize_params(model(x).clamp(-1, 1)).cpu())
    return torch.cat(preds).numpy(), ds.params.astype(np.float64)


def maes(pred: np.ndarray, true: np.ndarray) -> dict:
    """Mean absolute errors: spin (dimensionless) and inclination (degrees)."""
    err = np.abs(pred - true)
    return dict(spin=float(err[:, 0].mean()), incl=float(err[:, 1].mean()))


def guess_baseline(train: LensingSet, test: LensingSet) -> dict:
    """The errors of always guessing the training set's mean spin and inclination:
    what 'no information' costs."""
    return maes(np.broadcast_to(train.params.mean(0), test.params.shape), test.params)


def train_model(train_set: LensingSet, val_set: LensingSet, emis_fixed: Emissions,
                cfg: dict | None = None, device="cpu", augment: bool = False, log=print):
    """Train a ParamCNN on clean images, or (augment=True) on images with random
    blur and noise. Keeps the epoch with the best validation loss, measured on the
    same kind of images it trains on. Returns (model, history)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    model = ParamCNN(3, 2, cfg["width"], cfg["n_stages"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    n, bs = len(train_set), cfg["batch_size"]
    hist = dict(epoch=[], train=[], val=[], val_spin=[], val_incl=[], seconds=[])
    best, best_state = math.inf, None
    tag = "degraded" if augment else "clean"
    log(f"CNN ({tag} training): {count_params(model) / 1e6:.2f}M parameters, {n} holes "
        f"at {train_set.res}x{train_set.res}")
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).numpy()
        emis = random_emissions(n, gen)
        total = 0.0
        for b in range(0, n, bs):
            rows = perm[b:b + bs]
            bl, no = _degradation_draw(len(rows), cfg, gen) if augment else (0.0, 0.0)
            x, y = make_batch(train_set, rows, emis.take(torch.from_numpy(rows)), device,
                              bl, no, gen)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        sched.step()
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            pred, true = predict(model, val_set, emis_fixed, device, seed=123,
                                 val_cfg=cfg if augment else None)
            zp = normalize_params(torch.from_numpy(pred).float())
            zt = normalize_params(torch.from_numpy(true).float())
            val = float(F.mse_loss(zp, zt))
            m = maes(pred, true)
            for k, v in (("epoch", epoch), ("train", total / n), ("val", val),
                         ("val_spin", m["spin"]), ("val_incl", m["incl"]),
                         ("seconds", time.time() - t0)):
                hist[k].append(v)
            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())
            log(f"  epoch {epoch:4d}  train {total / n:.4f}  val {val:.4f}  "
                f"spin {m['spin']:.3f}  incl {m['incl']:.2f} deg  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, hist


def sweep(model, ds: LensingSet, emis_fixed: Emissions, device, blurs=(), noises=()) -> dict:
    """Errors on ds at each blur level (no noise) and each noise level (no blur)."""
    out = dict(blur=[], noise=[])
    for b in blurs:
        out["blur"].append((float(b), maes(*predict(model, ds, emis_fixed, device, blur=b))))
    for s in noises:
        out["noise"].append((float(s), maes(*predict(model, ds, emis_fixed, device, noise=s))))
    return out


def failure_level(curve, key: str, baseline: dict):
    """The first degradation level at which the error reaches half of guessing,
    interpolated linearly between the swept levels; None if it never does."""
    half = 0.5 * baseline[key]
    pts = [(lvl, m[key]) for lvl, m in curve]
    for (l0, e0), (l1, e1) in zip(pts, pts[1:]):
        if e0 < half <= e1:
            return l0 + (half - e0) * (l1 - l0) / (e1 - e0)
    return pts[0][0] if pts and pts[0][1] >= half else None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_degradations(ds: LensingSet, row: int, emis_fixed: Emissions, blurs, noises, path: str,
                      device="cpu"):
    """One test image across the blur levels (top) and the noise levels (bottom),
    as the network sees it (the stretched channel)."""
    rows = np.array([row])
    img = observed_images(ds, rows, _emis_for(ds, rows, emis_fixed), device)
    shadow = shadow_diameter_px(ds.manifest, ds.res)
    n = max(len(blurs), len(noises))
    fig, ax = plt.subplots(2, n, figsize=(2.1 * n, 5.0), squeeze=False)
    gen = torch.Generator().manual_seed(0)
    for j in range(n):
        for i, (levels, lab) in enumerate(((blurs, "blur"), (noises, "noise"))):
            a_ = ax[i, j]
            a_.set_xticks([])
            a_.set_yticks([])
            if j >= len(levels):
                a_.axis("off")
                continue
            kw = dict(blur=levels[j]) if lab == "blur" else dict(noise=levels[j])
            d = degrade(img, kw.get("blur", 0.0), kw.get("noise", 0.0), shadow, gen)
            a_.imshow(prepare(d)[0, ..., 0].cpu(), cmap="inferno", vmin=-0.2, vmax=1.0)
            title = (f"blur {levels[j]:g}" + (" (EHT)" if abs(levels[j] - EHT_BLUR) < 1e-9 else "")
                     if lab == "blur" else f"noise {levels[j]:g}")
            a_.set_title(title, fontsize=9)
    a, inc = ds.params[row]
    ax[0, 0].set_ylabel("blur", fontsize=10)
    ax[1, 0].set_ylabel("noise", fontsize=10)
    fig.suptitle(f"a = {a:.2f}, inclination {inc:.0f}°: blur = beam FWHM / shadow diameter, "
                 "noise = std / peak", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scatter(results: dict, path: str):
    """Predicted against true spin and inclination on clean test images."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for name, (pred, true) in results.items():
        ax[0].scatter(true[:, 0], pred[:, 0], s=9, alpha=0.6, label=name)
        ax[1].scatter(true[:, 1], pred[:, 1], s=9, alpha=0.6, label=name)
    ax[0].plot(A_RANGE, A_RANGE, "k--", lw=1)
    ax[1].plot(I_RANGE, I_RANGE, "k--", lw=1)
    ax[0].set_xlabel("true spin a")
    ax[0].set_ylabel("predicted spin")
    ax[1].set_xlabel("true inclination (deg)")
    ax[1].set_ylabel("predicted inclination (deg)")
    for a_ in ax:
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_robustness(sweeps: dict, baseline: dict, path: str):
    """Spin and inclination errors against blur and noise, for each model, with
    the cost of guessing (dashed) and half of it (dotted) for reference."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    for c, kind in enumerate(("blur", "noise")):
        for r, (key, unit) in enumerate((("spin", ""), ("incl", " (deg)"))):
            a_ = ax[r, c]
            for name, sw in sweeps.items():
                lv = [lvl for lvl, _ in sw[kind]]
                a_.plot(lv, [m[key] for _, m in sw[kind]], "-o", lw=2, ms=4, label=name)
            a_.axhline(baseline[key], color="k", ls="--", lw=1, label="guessing the mean")
            a_.axhline(0.5 * baseline[key], color="k", ls=":", lw=1, label="half of guessing")
            if kind == "blur":
                a_.axvline(EHT_BLUR, color="tab:gray", lw=1)
                a_.text(EHT_BLUR, a_.get_ylim()[1] * 0.97, " EHT M87*", fontsize=8,
                        va="top", color="tab:gray")
            a_.set_xlabel("blur: beam FWHM / shadow diameter" if kind == "blur"
                          else "noise: std / peak brightness")
            a_.set_ylabel(f"mean abs. error, {'spin' if key == 'spin' else 'inclination'}{unit}")
            a_.set_ylim(bottom=0)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
