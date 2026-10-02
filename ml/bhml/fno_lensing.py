"""Train and evaluate the emission -> image operator: an FNO and a U-Net baseline.

Driven by ml/notebooks/fno_lensing_kaggle.ipynb on a GPU. The pieces:

  train_model        fit one model at one resolution: relative L2 loss, a fresh
                     random emission profile for every hole every epoch, and fixed
                     profiles for validation (the best validation epoch is kept)
  evaluate           per-sample relative L2 error on a set, at any resolution
  time_model         images per second for a model (inputs built from scratch)
  time_ray_tracer    seconds per black hole for the GPU ray tracer, for comparison
  plot_*             the figures for the README
"""

from __future__ import annotations

import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from bhml.lensing_data import (
    N_CHANNELS,
    Emissions,
    LensingSet,
    build_inputs,
    make_batch,
    random_emissions,
)
from bhml.models import FNO2d, UNet2d, count_params

DEFAULTS = dict(
    fno=dict(modes=12, width=32, depth=4, pad_frac=0.125),
    unet=dict(width=34, n_levels=3),          # about the FNO's parameter count
    epochs=300, batch_size=32, lr=1e-3, weight_decay=1e-4, lr_step=100, lr_gamma=0.5,
    seed=0, eval_every=5,
)


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample ||pred - target|| / ||target||, averaged over the batch. Scale-free,
    so it compares across samples and resolutions (the standard FNO metric)."""
    b = pred.shape[0]
    diff = (pred - target).reshape(b, -1).norm(dim=1)
    return (diff / target.reshape(b, -1).norm(dim=1).clamp(min=1e-12)).mean()


def build_model(kind: str, cfg: dict | None = None) -> torch.nn.Module:
    cfg = {**DEFAULTS, **(cfg or {})}
    if kind == "fno":
        c = cfg["fno"]
        return FNO2d(c["modes"], c["modes"], c["width"], c["depth"], in_channels=N_CHANNELS,
                     pad_frac=c["pad_frac"])
    if kind == "unet":
        c = cfg["unet"]
        return UNet2d(N_CHANNELS, 1, c["width"], c["n_levels"])
    raise ValueError(f"unknown model kind {kind!r}")


def _emis_for(ds: LensingSet, rows, emis_fixed: Emissions) -> Emissions:
    """The fixed profile of each hole, looked up by its index in the full set, so a
    hole sees the same emission at every resolution."""
    return emis_fixed.take(torch.from_numpy(ds.index[np.asarray(rows)]))


@torch.no_grad()
def evaluate(model, ds: LensingSet, emis_fixed: Emissions, device, batch_size: int = 32):
    """Per-sample relative L2 error of `model` on every hole of `ds`."""
    model.eval()
    errs = []
    for b in range(0, len(ds), batch_size):
        rows = np.arange(b, min(b + batch_size, len(ds)))
        x, y, _ = make_batch(ds, rows, _emis_for(ds, rows, emis_fixed), device)
        p = model(x)
        n = len(rows)
        errs.append(((p - y).reshape(n, -1).norm(dim=1)
                     / y.reshape(n, -1).norm(dim=1).clamp(min=1e-12)).cpu())
    return torch.cat(errs).numpy()


def train_model(kind: str, train_set: LensingSet, val_set: LensingSet, emis_fixed: Emissions,
                cfg: dict | None = None, device="cpu", log=print):
    """Train an "fno" or "unet" on train_set; keep the epoch with the best mean
    validation error. Returns (model, history)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    model = build_model(kind, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    n, bs = len(train_set), cfg["batch_size"]
    hist = dict(epoch=[], train=[], val=[], seconds=[])
    best_val, best_state = math.inf, None
    log(f"{kind}: {count_params(model) / 1e6:.2f}M parameters, {n} training holes "
        f"at {train_set.res}x{train_set.res}")
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).numpy()
        emis = random_emissions(n, gen)          # fresh profiles every epoch
        total = 0.0
        for b in range(0, n, bs):
            rows = perm[b:b + bs]
            x, y, _ = make_batch(train_set, rows, emis.take(torch.from_numpy(rows)), device)
            loss = relative_l2(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(rows)
        sched.step()
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            val = float(evaluate(model, val_set, emis_fixed, device).mean())
            hist["epoch"].append(epoch)
            hist["train"].append(total / n)
            hist["val"].append(val)
            hist["seconds"].append(time.time() - t0)
            if val < best_val:
                best_val, best_state = val, copy.deepcopy(model.state_dict())
            log(f"  epoch {epoch:4d}  train {total / n:.4f}  val {val:.4f}  "
                f"({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model, hist


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #

def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def time_model(model, res: int, device, batch_size: int = 32, reps: int = 10,
               manifest: dict | None = None) -> float:
    """Seconds per image for `model` at res x res, including building its inputs
    (the gravity-off view and the emission) from (a, inclination) alone."""
    model.eval()
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(batch_size, generator=gen).to(device) * 0.99
    incl = (15 + 70 * torch.rand(batch_size, generator=gen)).to(device)
    emis = random_emissions(batch_size, gen).to(device)

    def run():
        x, _ = build_inputs(a, incl, emis, res, manifest)
        return model(x)

    for _ in range(3):
        run()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    _sync(device)
    return (time.perf_counter() - t0) / (reps * batch_size)


def time_ray_tracer(params: np.ndarray, res: int, device, r_out: float = 18.0) -> float:
    """Seconds per black hole for the GPU ray tracer at res x res, timed on one
    batch of the size the dataset generator uses (about a million rays per call),
    after a warm-up call. Needs ml/datagen and simulation/viz on sys.path."""
    import lensing  # the dataset generator, ml/datagen/lensing.py

    per_call = max(1, 1_000_000 // (res * res))
    holes = [(float(a), float(i)) for a, i in params[:per_call]]
    lensing.trace_batch(holes[:2], res, r_out, device)       # warm-up / compile
    _sync(device)
    t0 = time.perf_counter()
    lensing.trace_batch(holes, res, r_out, device)
    _sync(device)
    return (time.perf_counter() - t0) / len(holes)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

@torch.no_grad()
def plot_examples(models: dict, ds: LensingSet, rows, emis_fixed: Emissions, device, path: str):
    """For a few holes: the gravity-off input, the true image, and each model's
    prediction with its absolute error."""
    rows = np.asarray(rows)
    x, y, _ = make_batch(ds, rows, _emis_for(ds, rows, emis_fixed), device)
    preds = {name: m.eval()(x).cpu() for name, m in models.items()}
    x, y = x.cpu(), y.cpu()
    cols = 2 + 2 * len(models)
    fig, ax = plt.subplots(len(rows), cols, figsize=(2.3 * cols, 2.35 * len(rows)),
                           squeeze=False)
    show = lambda v: v.clamp(min=0) ** 0.5                          # noqa: E731
    for i, r in enumerate(rows):
        a, incl = ds.params[r]
        vmax = float(show(y[i, ..., 0]).max())
        ax[i, 0].imshow(show(x[i, ..., 0]), cmap="inferno")
        ax[i, 1].imshow(show(y[i, ..., 0]), cmap="inferno", vmin=0, vmax=vmax)
        ax[i, 0].set_ylabel(f"a={a:.2f}, i={incl:.0f}°", fontsize=9)
        c = 2
        for name, p in preds.items():
            err = float(((p[i] - y[i]).norm() / y[i].norm()))
            ax[i, c].imshow(show(p[i, ..., 0]), cmap="inferno", vmin=0, vmax=vmax)
            ax[i, c + 1].imshow((p[i, ..., 0] - y[i, ..., 0]).abs(), cmap="magma",
                                vmin=0, vmax=0.5 * float(y[i].max()))
            if i == 0:
                ax[i, c].set_title(name, fontsize=10)
                ax[i, c + 1].set_title(f"|{name} error|", fontsize=10)
            ax[i, c].text(0.03, 0.97, f"{100 * err:.1f}%", color="w", fontsize=8, va="top",
                          transform=ax[i, c].transAxes)
            c += 2
    ax[0, 0].set_title("input: gravity off", fontsize=10)
    ax[0, 1].set_title("truth: ray traced", fontsize=10)
    for a_ in ax.ravel():
        a_.set_xticks([])
        a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_transfer(errors: dict, train_res: int, path: str):
    """Test error against evaluation resolution, one line per model."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, by_res in errors.items():
        res = sorted(by_res)
        ax.plot(res, [100 * by_res[r] for r in res], "-o", lw=2, label=name)
    ax.axvline(train_res, color="k", ls="--", lw=1, label=f"trained at {train_res}²")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r for v in errors.values() for r in v}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}²"))
    ax.set_xlabel("evaluation resolution")
    ax.set_ylabel("test relative L2 error (%)")
    ax.set_title("Resolution transfer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(histories: dict, path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, h in histories.items():
        line, = ax.plot(h["epoch"], [100 * v for v in h["val"]], lw=2, label=f"{name} (val)")
        ax.plot(h["epoch"], [100 * v for v in h["train"]], lw=1, ls="--", color=line.get_color(),
                label=f"{name} (train)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 error (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
