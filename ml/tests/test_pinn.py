"""Tests for the orbit PINNs: the reference solutions, the built-in initial
conditions, and short forward and inverse fits. CPU only, under a minute."""

import math

import numpy as np
import torch

from bhml import pinn as P

torch.set_num_threads(1)     # tiny tensors: extra threads only add overhead (and contention)


def test_periapsis_solves_the_cubic():
    b = np.array([5.25, 6.0, 10.0, 100.0])
    r = P.periapsis(b)
    assert np.allclose(r ** 3 - b ** 2 * (r - 2), 0, atol=1e-8 * b ** 3)
    assert np.all(r > 3)                                     # outside the photon sphere


def test_elliptic_integral_limits():
    assert abs(P.ellip_f(math.pi / 2, 0.0) - math.pi / 2) < 1e-12
    assert abs(P.ellip_f(0.3, 0.0) - 0.3) < 1e-12
    # F(phi | 1) = artanh(sin phi)
    assert abs(P.ellip_f(1.0, 1.0) - math.atanh(math.sin(1.0))) < 1e-10


def test_exact_deflection_against_quadrature_and_weak_field():
    # adaptive quadrature of 2 * integral du / sqrt(1/b^2 - u^2 + 2u^3) - pi (scipy, 1e-13)
    for b, quad in ((5.25, 4.194799708235089), (8.0, 0.8587300180777371),
                    (20.0, 0.2361359953879112)):
        assert abs(P.exact_deflection(b) - quad) < 1e-10 * quad
    assert abs(P.exact_deflection(1e5) * 1e5 / 4 - 1) < 1e-3   # Einstein's 4M/b


def test_rk4_finds_the_exact_turning_angle():
    b = 7.0
    phi, u, du = P.rk4_binet(0.0, 1 / b, 3.0, h=1e-3)
    assert abs(P.first_downturn(phi, du) - P.turning_angle(b)) < 1e-5


def test_first_downturn():
    phi = np.linspace(0, 3, 301)
    assert abs(P.first_downturn(phi, np.cos(phi)) - math.pi / 2) < 1e-4
    assert P.first_downturn(phi, np.ones_like(phi)) is None


def test_initial_conditions_are_built_in():
    win = P.Window(1.0, 2.0, 0.1, 0.05, 8.0, P.FORWARD)
    u, du, _ = P.derivatives(win, torch.tensor([[1.0]], dtype=P.DTYPE))
    assert abs(u.item() - 0.1) < 1e-14 and abs(du.item() - 0.05) < 1e-14
    model = P.ParamRay(P.PARAM)
    b = torch.tensor([[5.5], [12.0], [29.0]], dtype=P.DTYPE)
    u, du, _ = P.derivatives(model, torch.zeros(3, 1, dtype=P.DTYPE), b)
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-14)
    assert torch.allclose(du, 1 / b, atol=1e-14)


def test_short_forward_fit_gets_the_deflection():
    r = P.solve_ray(10.0, "single", dict(adam=300, lbfgs=200, n_fix=1024))
    assert r["rel_err"] < 1e-2


def test_shooting_fit_recovers_f_without_noise():
    obs = P.make_orbit(f=1.0, noise=0.0, n_orbits=2, n_obs=20)
    fit = P.fit_shooting(obs, f_inits=(0.5,))
    assert abs(fit["f"] - 1) < 1e-4 and abs(fit["p"] / 20 - 1) < 1e-4


def test_inverse_pinn_runs():
    obs = P.make_orbit(f=1.0, noise=0.01, n_orbits=1, n_obs=15)
    _, fit = P.fit_inverse(obs, dict(width=16, depth=2, adam=200, lbfgs=50, n_fix=512))
    assert np.isfinite(fit["f"]) and np.isfinite(fit["p"]) and fit["p"] > 0
