"""Train the black-hole ML models.

Run from the repo root (venv active). Needs torch (CPU is fine):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    python -m bhml.train capture     [data/capture.npz]     # classifier -> recovers b_crit
    python -m bhml.train deflection  [data/deflection.npz]  # regressor + speed benchmark
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from bhml.models import MLP
from bhml.sim import trace_photon

plt.switch_backend("Agg")  # render off-screen; no display needed

SEED = 0


def _split(n: int, frac: float = 0.8):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


# --------------------------------------------------------------------------- #
# Phase 2: capture classifier
# --------------------------------------------------------------------------- #
def train_capture(data_path: str) -> None:
    d = np.load(data_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xn = (X - mu) / sd
    tr, va = _split(len(X))

    torch.manual_seed(SEED)
    Xtr, ytr = torch.tensor(Xn[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(Xn[va]), torch.tensor(y[va])

    model = MLP(in_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(300):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(model(Xva)) > 0.5).float() == yva).float().mean().item()
            print(f"epoch {epoch + 1:3d}  loss {loss.item():.4f}  val_acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        val_acc = ((torch.sigmoid(model(Xva)) > 0.5).float() == yva).float().mean().item()
    print(f"final val accuracy: {val_acc:.4f}")

    b_crit = 3.0 * np.sqrt(3.0)
    b_grid = np.linspace(3.0, 8.0, 2001, dtype=np.float32)

    def prob_vs_b(r0: float) -> np.ndarray:
        feats = np.column_stack([b_grid, np.full_like(b_grid, r0)]).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor((feats - mu) / sd))).numpy()

    p = prob_vs_b(float(np.median(X[:, 1])))
    b_boundary = float(b_grid[np.argmin(np.abs(p - 0.5))])
    print(f"learned boundary b = {b_boundary:.4f}  vs  analytic 3*sqrt(3) = {b_crit:.4f}")

    r0_lo, r0_hi = float(X[:, 1].min()), float(X[:, 1].max())
    crossings = [float(b_grid[np.argmin(np.abs(prob_vs_b(r0) - 0.5))])
                 for r0 in np.linspace(r0_lo, r0_hi, 7)]
    print(f"boundary spread over r0 in [{r0_lo:.0f}, {r0_hi:.0f}]: "
          f"{min(crossings):.4f} to {max(crossings):.4f}  (should be ~flat)")

    os.makedirs(os.path.join("docs", "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(b_grid, p, color="#1f77b4", label="model P(capture)")
    ax.axvline(b_crit, color="black", ls="--", lw=1.0, label=f"analytic 3*sqrt(3) = {b_crit:.3f}")
    ax.axvline(b_boundary, color="#d62728", ls=":", lw=1.5, label=f"learned = {b_boundary:.3f}")
    ax.axhline(0.5, color="gray", lw=0.5)
    ax.set_xlabel("impact parameter  b / M")
    ax.set_ylabel("P(capture)")
    ax.set_title(f"Learned capture boundary  (val acc {val_acc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "capture_boundary.png"), dpi=150)
    print("wrote docs/figures/capture_boundary.png")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "feature_names": list(d["feature_names"])},
               os.path.join("checkpoints", "capture_mlp.pt"))
    print("wrote checkpoints/capture_mlp.pt")


# --------------------------------------------------------------------------- #
# Phase 3: deflection surrogate
# --------------------------------------------------------------------------- #
def train_deflection(data_path: str) -> None:
    d = np.load(data_path, allow_pickle=True)
    X = d["X"].astype(np.float32)          # (n, 1) impact parameter b
    y = d["y"].astype(np.float32)          # (n,) deflection angle

    # standardize both inputs and target (regression trains better this way)
    mu_x, sd_x = X.mean(axis=0), X.std(axis=0)
    mu_y, sd_y = float(y.mean()), float(y.std())
    Xn = (X - mu_x) / sd_x
    yn = (y - mu_y) / sd_y
    tr, va = _split(len(X))

    torch.manual_seed(SEED)
    Xtr, ytr = torch.tensor(Xn[tr]), torch.tensor(yn[tr])
    Xva = torch.tensor(Xn[va])

    model = MLP(in_dim=1, hidden=(64, 64))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(2000):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if (epoch + 1) % 400 == 0:
            print(f"epoch {epoch + 1:4d}  train_mse {loss.item():.5f}")

    # validation error, back in physical units (radians)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva).numpy() * sd_y + mu_y
    true_va = y[va]
    rel_l2 = float(np.linalg.norm(pred_va - true_va) / np.linalg.norm(true_va))
    print(f"val relative L2 error: {100 * rel_l2:.2f}%")

    # figure: model curve over the true deflection, with 4M/b for reference
    b_grid = np.linspace(float(X.min()), float(X.max()), 1000, dtype=np.float32).reshape(-1, 1)
    with torch.no_grad():
        pred_grid = model(torch.tensor((b_grid - mu_x) / sd_x)).numpy() * sd_y + mu_y

    os.makedirs(os.path.join("docs", "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[va, 0], true_va, s=6, alpha=0.3, color="gray", label="integrator (val)")
    ax.plot(b_grid[:, 0], pred_grid, color="#1f77b4", lw=2, label="surrogate")
    ax.plot(b_grid[:, 0], 4.0 / b_grid[:, 0], color="black", ls="--", lw=1, label="4M/b (weak field)")
    ax.set_xlabel("impact parameter  b / M")
    ax.set_ylabel("deflection  delta phi  (rad)")
    ax.set_title(f"Deflection surrogate  (val rel. L2 {100 * rel_l2:.2f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("docs", "figures", "deflection_fit.png"), dpi=150)
    print("wrote docs/figures/deflection_fit.png")

    # speed: surrogate (batched) vs integrator (per call)
    bench = np.random.default_rng(1).uniform(5.25, 30.0, size=2000).astype(np.float32).reshape(-1, 1)
    bt = torch.tensor((bench - mu_x) / sd_x)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(bt).numpy()
    t_nn = (time.perf_counter() - t0) / len(bench)

    t0 = time.perf_counter()
    for i in range(100):
        trace_photon(float(bench[i, 0]), 1.0e4)
    t_int = (time.perf_counter() - t0) / 100

    print(f"surrogate:  {t_nn * 1e6:8.2f} us/sample (batched)")
    print(f"integrator: {t_int * 1e3:8.2f} ms/sample")
    print(f"speedup: {t_int / t_nn:.0f}x")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y, "sd_y": sd_y},
               os.path.join("checkpoints", "deflection_mlp.pt"))
    print("wrote checkpoints/deflection_mlp.pt")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    if mode not in ("capture", "deflection"):
        sys.exit("usage: python -m bhml.train [capture|deflection] [data.npz]")
    data_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", f"{mode}.npz")

    if mode == "capture":
        train_capture(data_path)
    else:
        train_deflection(data_path)


if __name__ == "__main__":
    main()
