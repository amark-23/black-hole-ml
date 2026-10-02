"""Tests for the emission -> image operator task: the data pipeline, the models,
and a short training run. CPU only, a few seconds in total."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation" / "viz"))   # gpu_render: the ray tracer
sys.path.insert(0, str(ROOT / "ml" / "datagen"))       # lensing: the dataset generator

import gpu_render as G  # noqa: E402
import lensing as L  # noqa: E402

from bhml import fno_lensing as F  # noqa: E402
from bhml.lensing_data import (  # noqa: E402
    LensingSet,
    fixed_emissions,
    flat_disk_view,
    isco_radius,
    make_batch,
    random_emissions,
)
from bhml.models import FNO2d, SpectralConv2d, UNet2d, count_params  # noqa: E402

torch.set_num_threads(1)     # tiny tensors: extra threads only add overhead (and contention)


def test_spectral_conv_is_resolution_invariant():
    """A band-limited field sampled on 32^2 and 64^2 grids gives the same output
    at the shared points: the layer acts on frequencies, not pixels."""
    torch.manual_seed(0)
    layer = SpectralConv2d(2, 3, 4, 4)

    def field(n):
        t = torch.arange(n) / n
        x, y = torch.meshgrid(t, t, indexing="ij")
        f1 = torch.sin(2 * math.pi * (x + 2 * y)) + 0.5 * torch.cos(2 * math.pi * 3 * x)
        f2 = torch.cos(2 * math.pi * (2 * x - y))
        return torch.stack([f1, f2])[None]

    with torch.no_grad():
        coarse, fine = layer(field(32)), layer(field(64))
    assert torch.allclose(fine[..., ::2, ::2], coarse, atol=1e-5)


def test_models_run_at_any_resolution():
    for model in (FNO2d(4, 4, 8, 2), UNet2d(5, 1, 8, 2)):
        for n in (32, 64):
            assert model(torch.randn(2, n, n, 5)).shape == (2, n, n, 1)


def test_complex_weights_count_twice():
    assert count_params(SpectralConv2d(2, 3, 4, 5)) == 2 * (2 * 2 * 3 * 4 * 5)


def test_isco_matches_the_ray_tracer():
    for a in (0.0, 0.5, 0.9, 0.99):
        assert isco_radius(np.float64(a)) == pytest.approx(G.isco_radius(a), rel=1e-12)


@pytest.mark.parametrize("incl", [20.0, 55.0, 80.0])
def test_gravity_off_view_matches_a_weak_field_trace(incl):
    """Trace a hole of mass 1e-6 (rays go straight): it must hit the disk at exactly
    the pixels, radii and angles the gravity-off projection predicts. This pins the
    projection to the ray tracer's camera, pixel layout and phi convention."""
    res, dist, fov, M, a, r_in, r_out = 12, 40.0, 30.0, 1e-6, 0.0, 0.5, 18.0
    cam = dict(a=a, incl=math.radians(incl), az=0.0, dist=dist, fov=math.radians(fov),
               res=(res, res), r_in=r_in, r_out=r_out)
    Y = G.camera_rays(cam, "cpu")
    n = Y.shape[0]
    work = [torch.zeros(n, dtype=torch.int8)] + [torch.zeros(n) for _ in range(5)]
    active = torch.ones(n, dtype=torch.bool)
    for _ in range(3000):
        Y, active, *work = G.trace_step(Y, active, *work, torch.tensor(1.4 * dist), a, M, 0.013,
                                        2.02 * M, r_in, r_out)
        if not active.any():
            break
    otype, hit_r, hit_ph = (w.reshape(res, res) for w in work[:3])
    r_f, ph_f, front = flat_disk_view(torch.tensor([incl]), res, dist, fov)
    flat_disk = (front & (r_f >= r_in) & (r_f <= r_out))[0]
    assert torch.equal(flat_disk, otype == 2)
    assert (hit_r - r_f[0])[flat_disk].abs().max() < 2e-3
    dphi = torch.remainder(hit_ph - ph_f[0] + math.pi, 2 * math.pi) - math.pi
    assert dphi[flat_disk].abs().max() < 1e-4


def test_emissions_are_positive_and_periodic():
    emis = random_emissions(4, torch.Generator().manual_seed(0))
    r = torch.linspace(1.5, 18, 50).expand(4, 50)
    ph = torch.linspace(-3, 3, 50).expand(4, 50)
    r_in = torch.full((4,), 1.5)
    e = emis(r, ph, r_in)
    assert (e > 0).all()
    assert torch.allclose(e, emis(r, ph + 2 * math.pi, r_in), rtol=1e-5)


def _synthetic_set(n=8, res=16, seed=0):
    """A small fake geometry set (random but valid fields) for pipeline tests."""
    rng = np.random.default_rng(seed)
    params = np.stack([rng.uniform(0, 0.99, n), rng.uniform(15, 85, n)], 1).astype(np.float32)
    r_in = isco_radius(params[:, 0].astype(np.float64))
    hit_r = (r_in[:, None, None] + rng.uniform(0, 1, (n, res, res)) * (18 - r_in[:, None, None]))
    return LensingSet(
        otype=rng.integers(1, 4, (n, res, res)).astype(np.int8),
        hit_r=hit_r.astype(np.float16),
        hit_ph=rng.uniform(-np.pi, np.pi, (n, res, res)).astype(np.float16),
        hit_g=rng.uniform(0.6, 1.4, (n, res, res)).astype(np.float16),
        params=params, split=np.array(["train"] * n), index=np.arange(n),
        manifest=dict(dist=40.0, fov=30.0, az=0.0, r_out=18.0))


def test_targets_match_the_generators_forward_operator():
    """make_batch's target is exactly lensing.emission_to_image with the g^4 shift."""
    ds = _synthetic_set()
    emis = fixed_emissions(len(ds))
    rows = np.arange(len(ds))
    _, y, scale = make_batch(ds, rows, emis.take(torch.from_numpy(rows)), "cpu")
    for i in rows:
        one = emis.take(torch.tensor([i]))
        r_in = isco_radius(torch.tensor([float(ds.params[i, 0])]))

        def emission(r, ph):
            return one(torch.from_numpy(r)[None], torch.from_numpy(ph)[None], r_in)[0].numpy()

        ref = L.emission_to_image(ds.otype[i], ds.hit_r[i].astype("f4"), ds.hit_ph[i].astype("f4"),
                                  emission, hit_g=ds.hit_g[i])
        np.testing.assert_allclose(y[i, ..., 0].numpy() * scale[i].item(), ref,
                                   rtol=1e-4, atol=1e-6)


def test_a_short_training_run_learns():
    """With straight-line geometry (gravity off on both sides, g = 1) the operator
    is the identity on the disk: a tiny FNO must get much better at it quickly."""
    n, res = 12, 16
    ds = _synthetic_set(n, res)
    a = torch.from_numpy(ds.params[:, 0])
    incl = torch.from_numpy(ds.params[:, 1])
    r_f, ph_f, front = flat_disk_view(incl, res, 40.0, 30.0)
    on = front & (r_f >= isco_radius(a)[:, None, None]) & (r_f <= 18.0)
    ds.otype = np.where(on.numpy(), 2, 3).astype(np.int8)
    ds.hit_r, ds.hit_ph = r_f.numpy().astype(np.float16), ph_f.numpy().astype(np.float16)
    ds.hit_g = np.ones_like(ds.hit_r)
    emis = fixed_emissions(n)
    cfg = dict(fno=dict(modes=4, width=12, depth=2, pad_frac=0.125), epochs=40,
               batch_size=6, lr=3e-3, eval_every=40, lr_step=100)
    torch.manual_seed(cfg.get("seed", 0))
    before = F.evaluate(F.build_model("fno", cfg), ds, emis, "cpu").mean()
    model, hist = F.train_model("fno", ds, ds, emis, cfg, "cpu", log=lambda s: None)
    after = F.evaluate(model, ds, emis, "cpu").mean()
    assert after < 0.5 * before
