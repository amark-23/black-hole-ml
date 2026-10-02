"""Tests for the inverse task (image -> spin, inclination): targets, degradations,
the network's input, and a short training run. CPU only, a few seconds."""

import numpy as np
import torch

from bhml import cnn_inverse as C
from bhml.lensing_data import LensingSet, fixed_emissions, flat_disk_view, isco_radius

torch.set_num_threads(1)     # tiny tensors: extra threads only add overhead (and contention)


def test_parameter_scaling_round_trips():
    p = torch.tensor([[0.0, 15.0], [0.99, 85.0], [0.5, 50.0]])
    z = C.normalize_params(p)
    assert torch.allclose(z[0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(z[1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(C.denormalize_params(z), p, atol=1e-5)


def test_blur_keeps_the_light_and_zero_width_is_a_no_op():
    blob = torch.zeros(2, 64, 64)
    blob[:, 28:36, 28:36] = 1.0
    out = C.gaussian_blur(blob, torch.tensor([0.0, 3.0]))
    assert torch.equal(out[0], blob[0])
    assert abs(float(out[1].sum() / blob[1].sum()) - 1) < 1e-5
    assert float(out[1].max()) < 1.0                         # it did spread


def test_noise_has_the_requested_level():
    img = torch.rand(8, 64, 64)
    noisy = C.degrade(img, 0.0, 0.1, shadow_px=30.0, generator=torch.Generator().manual_seed(0))
    level = ((noisy - img).flatten(1).std(1) / img.flatten(1).amax(1)).numpy()
    assert np.allclose(level, 0.1, rtol=0.05)


def test_input_ignores_absolute_brightness():
    img = torch.rand(3, 32, 32)
    assert torch.allclose(C.prepare(img), C.prepare(7.5 * img), atol=1e-5)
    assert C.prepare(img).shape == (3, 32, 32, 3)


def test_failure_level_interpolates():
    curve = [(0.0, dict(spin=0.02)), (0.5, dict(spin=0.08)), (1.0, dict(spin=0.16))]
    base = dict(spin=0.24)                                   # half of guessing: 0.12
    assert abs(C.failure_level(curve, "spin", base) - 0.75) < 1e-9
    assert C.failure_level(curve[:2], "spin", base) is None


def test_a_short_training_run_learns_inclination():
    """Gravity-off disks: the only signal is how tilted the disk looks, which a
    tiny CNN must pick up quickly."""
    n, res = 24, 32
    rng = np.random.default_rng(0)
    params = np.stack([rng.uniform(0, 0.99, n), rng.uniform(15, 85, n)], 1).astype(np.float32)
    a, incl = torch.from_numpy(params[:, 0]), torch.from_numpy(params[:, 1])
    r_f, ph_f, front = flat_disk_view(incl, res, 40.0, 30.0)
    on = front & (r_f >= isco_radius(a)[:, None, None]) & (r_f <= 18.0)
    ds = LensingSet(otype=np.where(on.numpy(), 2, 3).astype(np.int8),
                    hit_r=r_f.numpy().astype(np.float16), hit_ph=ph_f.numpy().astype(np.float16),
                    hit_g=np.ones((n, res, res), np.float16), params=params,
                    split=np.array(["train"] * n), index=np.arange(n),
                    manifest=dict(dist=40.0, fov=30.0, r_out=18.0, res=res))
    emis = fixed_emissions(n)
    cfg = dict(width=8, n_stages=3, epochs=30, batch_size=8, lr=3e-3, lr_step=100,
               eval_every=30)
    base = C.guess_baseline(ds, ds)["incl"]
    model, _ = C.train_model(ds, ds, emis, cfg, "cpu", log=lambda s: None)
    err = C.maes(*C.predict(model, ds, emis, "cpu"))["incl"]
    assert err < 0.5 * base
