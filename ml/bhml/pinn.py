"""Physics-informed neural networks for Schwarzschild orbits: the orbit equation is
the loss.

Equatorial orbits are written in Binet form, u = M/r as a function of the orbital
angle phi (units M = 1):

  light          u'' + u = 3 u^2
  a massive body u'' + u = 1/p + 3 f u^2      (f = 1 is Einstein, f = 0 is Newton)

Driven by ml/notebooks/pinn_geodesics_kaggle.ipynb on CPU. The pieces:

  exact_deflection     a light ray's deflection in closed form (Darwin's elliptic
                       integral), the truth the forward PINNs are graded on
  rk4_binet            a plain RK4 integrator of the same equations, for orbits
  solve_ray            forward PINN for one ray, from the equation alone (no data):
                       one network over the whole ray, or marched window by window
  ParamRay / train_param
                       one network u(phi, b) for every ray b in [b_lo, b_hi], a
                       deflection surrogate trained on physics alone
  make_orbit / fit_inverse
                       inverse PINN: from a few noisy positions of a star on a
                       close orbit, recover the orbit size p and the strength f of
                       the relativistic term
  fit_shooting         the classical answer to the same inverse problem: shoot the
                       ODE and fit its initial conditions, p and f by least squares
  run_parallel / call  the experiments' jobs across CPU cores
  plot_*               the figures for the README

Everything runs in float64: the residual of a second derivative is small, and
float32 rounding sets a floor on it well above where the solution stops improving.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import matplotlib.pyplot as plt
import numpy as np
import torch

B_CRIT = 3 * math.sqrt(3)       # critical impact parameter: rays with b < B_CRIT fall in
DTYPE = torch.float64

FORWARD = dict(width=32, depth=3, adam=3000, lbfgs=1000, lr=2e-3, n_col=512, n_fix=2048,
               window=1.0, max_windows=60, seed=0)
PARAM = dict(b_lo=5.25, b_hi=30.0, phi_max=2 * math.pi, enc="log", width=64, depth=4,
             adam=10000, lbfgs=2000, lr=2e-3, n_col=1024, n_fix=4096, seed=0)
INVERSE = dict(width=64, depth=4, adam=3000, lbfgs=1000, lr=2e-3, n_col=1024, n_fix=4096,
               w_phys=1.0, f_init=0.5, seed=0)


# --------------------------------------------------------------------------- #
# Reference solutions
# --------------------------------------------------------------------------- #

def periapsis(b):
    """Closest approach r0 of a light ray with impact parameter b > B_CRIT, the
    largest root of r^3 - b^2 (r - 2) = 0."""
    b = np.asarray(b, dtype=float)
    return 2 * b / math.sqrt(3) * np.cos(np.arccos(-B_CRIT / b) / 3)


def _carlson_rf(x, y, z, tol=1e-12):
    """Carlson's symmetric elliptic integral R_F(x, y, z), by duplication."""
    x, y, z = (np.array(v, dtype=float) for v in np.broadcast_arrays(x, y, z))
    for _ in range(100):
        mu = (x + y + z) / 3
        if np.all(np.maximum.reduce([abs(x - mu), abs(y - mu), abs(z - mu)]) <= tol * abs(mu)):
            break
        sx, sy, sz = np.sqrt(x), np.sqrt(y), np.sqrt(z)
        lam = sx * sy + sy * sz + sz * sx
        x, y, z = (x + lam) / 4, (y + lam) / 4, (z + lam) / 4
    mu = (x + y + z) / 3
    dx, dy, dz = 1 - x / mu, 1 - y / mu, 1 - z / mu
    e2, e3 = dx * dy - dz * dz, dx * dy * dz
    return (1 - e2 / 10 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44) / np.sqrt(mu)


def ellip_f(phi, k2):
    """Incomplete elliptic integral of the first kind F(phi | k^2), 0 <= phi <= pi/2."""
    s, c = np.sin(phi), np.cos(phi)
    return s * _carlson_rf(c * c, 1 - k2 * s * s, 1.0)


def exact_deflection(b):
    """Total deflection of a light ray with impact parameter b > B_CRIT (Darwin 1959):
    4 sqrt(r0 / Q) (K(k) - F(zeta, k)) - pi, with Q^2 = (r0 - 2)(r0 + 6)."""
    r0 = periapsis(b)
    q = np.sqrt((r0 - 2) * (r0 + 6))
    k2 = (q - r0 + 6) / (2 * q)
    zeta = np.arcsin(np.sqrt((q - r0 + 2) / (q - r0 + 6)))
    return 4 * np.sqrt(r0 / q) * (ellip_f(math.pi / 2, k2) - ellip_f(zeta, k2)) - math.pi


def turning_angle(b):
    """The angle phi at which a ray coming in from infinity reaches periapsis."""
    return (math.pi + exact_deflection(b)) / 2


def rk4_binet(u0, v0, phi_max, p=math.inf, f=1.0, h=1e-3):
    """Integrate u'' + u = 1/p + 3 f u^2 from phi = 0 (p = inf is light). Returns the
    grid phi and u, u' on it."""
    n = max(1, int(math.ceil(phi_max / h)))
    h = phi_max / n
    inv_p = 0.0 if math.isinf(p) else 1.0 / p
    out = np.empty((n + 1, 2))
    y = np.array([u0, v0], dtype=float)
    out[0] = y

    def rhs(y):
        return np.array([y[1], inv_p + 3 * f * y[0] ** 2 - y[0]])

    for i in range(n):
        k1 = rhs(y)
        k2 = rhs(y + h / 2 * k1)
        k3 = rhs(y + h / 2 * k2)
        k4 = rhs(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y
    return np.linspace(0, phi_max, n + 1), out[:, 0], out[:, 1]


def first_downturn(phi: np.ndarray, du: np.ndarray):
    """The first angle where u' falls through zero (linear interpolation), or None."""
    idx = np.flatnonzero(du <= 0)
    if len(idx) == 0 or idx[0] == 0:
        return None
    i = idx[0]
    return float(phi[i - 1] + du[i - 1] / (du[i - 1] - du[i]) * (phi[i] - phi[i - 1]))


# --------------------------------------------------------------------------- #
# Networks and derivatives
# --------------------------------------------------------------------------- #

def mlp(in_dim: int, width: int, depth: int) -> torch.nn.Sequential:
    """tanh MLP: smooth, so its second derivative is too."""
    layers, k = [], in_dim
    for _ in range(depth):
        layers += [torch.nn.Linear(k, width), torch.nn.Tanh()]
        k = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(k, 1)).to(DTYPE)
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return net


def derivatives(u_fn, phi: torch.Tensor, *args):
    """u, du/dphi and d2u/dphi2 at phi, by autograd (kept in the graph for training)."""
    phi = phi.detach().requires_grad_(True)
    u = u_fn(phi, *args)
    du = torch.autograd.grad(u.sum(), phi, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), phi, create_graph=True)[0]
    return u, du, d2u


def _lbfgs(params, closure, iters: int):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, max_eval=int(1.25 * iters),
                            history_size=100, tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")
    opt.step(closure)


def _adam(params, loss_fn, steps: int, lr: float):
    """Adam with a cosine schedule down to lr/100; loss_fn draws its own points."""
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), eta_min=lr / 100)
    for _ in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


# --------------------------------------------------------------------------- #
# Forward: one light ray, from the equation alone
# --------------------------------------------------------------------------- #

class Window(torch.nn.Module):
    """u on [phi0, phi0 + W] with the initial conditions built in:

        u = u0 + v0 s + (s / W)^2 N(2 s / W - 1) / b,   s = phi - phi0

    so u(phi0) = u0 and u'(phi0) = v0 whatever the network does, and the loss is the
    orbit equation's residual alone."""

    def __init__(self, phi0, width, u0, v0, b, cfg):
        super().__init__()
        self.phi0, self.W, self.u0, self.v0, self.b = phi0, width, u0, v0, b
        self.net = mlp(1, cfg["width"], cfg["depth"])

    def forward(self, phi):
        s = phi - self.phi0
        return self.u0 + self.v0 * s + (s / self.W) ** 2 * self.net(2 * s / self.W - 1) / self.b

    def residual(self, phi):
        u, _, d2u = derivatives(self, phi)
        return self.b * (d2u + u - 3 * u ** 2)       # scaled by b: O(1) at every b


def fit_window(win: Window, cfg: dict, gen: torch.Generator) -> float:
    def draw(n):
        return win.phi0 + win.W * torch.rand(n, 1, generator=gen, dtype=DTYPE)

    _adam(win.net.parameters(), lambda: win.residual(draw(cfg["n_col"])).pow(2).mean(),
          cfg["adam"], cfg["lr"])
    fixed = draw(cfg["n_fix"])

    def closure():
        win.net.zero_grad()
        loss = win.residual(fixed).pow(2).mean()
        loss.backward()
        return loss

    if cfg["lbfgs"]:
        _lbfgs(win.net.parameters(), closure, cfg["lbfgs"])
    return float(win.residual(fixed).pow(2).mean().detach())


def _scan(win: Window, n: int = 2001):
    """u and u' on a fine grid across a window (no graph)."""
    phi = torch.linspace(win.phi0, win.phi0 + win.W, n, dtype=DTYPE)[:, None]
    u, du, _ = derivatives(win, phi)
    return phi[:, 0].numpy(), u.detach()[:, 0].numpy(), du.detach()[:, 0].numpy()


def solve_ray(b: float, mode: str = "march", cfg: dict | None = None, phi_max: float | None = None):
    """Forward PINN for the light ray of impact parameter b, coming in from infinity
    (u = 0, u' = 1/b at phi = 0). The deflection is read off where u' first turns
    negative, the periapsis: deflection = 2 phi_turn - pi.

    mode "single": one network over [0, phi_max]. phi_max defaults to the exact
        turning angle plus one radian, a domain that is just long enough, which is
        as generous to the plain PINN as it can be.
    mode "march": windows of cfg["window"] radians, each starting from where the
        last one ended, until u' turns. It needs no knowledge of the answer.
    """
    cfg = {**FORWARD, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    t0 = time.time()
    u0, v0, phi0 = 0.0, 1.0 / b, 0.0
    if mode == "single":
        widths = [phi_max if phi_max is not None else float(turning_angle(b)) + 1.0]
    elif mode == "march":
        widths = [cfg["window"]] * cfg["max_windows"]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    windows, losses, phi_turn = [], [], None
    for w in widths:
        win = Window(phi0, w, u0, v0, b, cfg)
        losses.append(fit_window(win, cfg, gen))
        windows.append(win)
        ph, u, du = _scan(win)
        phi_turn = first_downturn(ph, du)
        if phi_turn is not None or u.max() > 0.5:   # periapsis, or it fell in
            break
        u0, v0, phi0 = float(u[-1]), float(du[-1]), phi0 + w
    exact = float(exact_deflection(b))
    defl = 2 * phi_turn - math.pi if phi_turn is not None else math.nan
    # the ray's shape against an RK4 solution, up to periapsis
    ph = np.concatenate([_scan(w, 401)[0] for w in windows])
    u = np.concatenate([_scan(w, 401)[1] for w in windows])
    end = phi_turn if phi_turn is not None else ph[-1]
    grid, u_ref, _ = rk4_binet(0.0, 1.0 / b, float(ph[-1]), h=1e-3)
    keep = ph <= end
    u_err = float(np.max(np.abs(np.interp(ph[keep], grid, u_ref) - u[keep])) * b)
    return dict(b=b, mode=mode, deflection=defl, exact=exact,
                rel_err=abs(defl - exact) / exact if phi_turn is not None else math.nan,
                u_err=u_err, n_windows=len(windows), loss=max(losses), time=time.time() - t0,
                phi=ph[keep][::4].tolist(), u=u[keep][::4].tolist())


# --------------------------------------------------------------------------- #
# Forward, parametric: every ray at once
# --------------------------------------------------------------------------- #

class ParamRay(torch.nn.Module):
    """u(phi, b) for every ray b in [b_lo, b_hi]:

        u = (sin phi + (phi / Phi)^2 N(phi, b)) / b

    sin(phi)/b is the straight line, so u(0) = 0 and u'(0) = 1/b for every b. The
    network sees b through log(b - B_CRIT) ("log"), which spreads out the rays near
    the photon sphere where the deflection changes fastest, or linearly ("lin")."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.net = mlp(2, cfg["width"], cfg["depth"])

    def encode(self, b):
        lo, hi = self.cfg["b_lo"], self.cfg["b_hi"]
        if self.cfg["enc"] == "log":
            lo, hi, b = math.log(lo - B_CRIT), math.log(hi - B_CRIT), torch.log(b - B_CRIT)
        return 2 * (b - lo) / (hi - lo) - 1

    def forward(self, phi, b):
        big = self.cfg["phi_max"]
        x = torch.cat([2 * phi / big - 1, self.encode(b)], dim=1)
        return (torch.sin(phi) + (phi / big) ** 2 * self.net(x)) / b

    def residual(self, phi, b):
        u, _, d2u = derivatives(self, phi, b)
        return b * (d2u + u - 3 * u ** 2)

    def draw_b(self, n, gen):
        """b uniform in the encoding, so near-critical rays get their share."""
        z = torch.rand(n, 1, generator=gen, dtype=DTYPE)
        lo, hi = self.cfg["b_lo"], self.cfg["b_hi"]
        if self.cfg["enc"] == "log":
            llo, lhi = math.log(lo - B_CRIT), math.log(hi - B_CRIT)
            return B_CRIT + torch.exp(llo + z * (lhi - llo))
        return lo + z * (hi - lo)


def train_param(cfg: dict | None = None):
    cfg = {**PARAM, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    model = ParamRay(cfg)
    big = cfg["phi_max"]

    def draw(n):
        return big * torch.rand(n, 1, generator=gen, dtype=DTYPE), model.draw_b(n, gen)

    t0 = time.time()
    _adam(model.parameters(), lambda: model.residual(*draw(cfg["n_col"])).pow(2).mean(),
          cfg["adam"], cfg["lr"])
    fixed = draw(cfg["n_fix"])

    def closure():
        model.zero_grad()
        loss = model.residual(*fixed).pow(2).mean()
        loss.backward()
        return loss

    if cfg["lbfgs"]:
        _lbfgs(model.parameters(), closure, cfg["lbfgs"])
    loss = float(model.residual(*fixed).pow(2).mean().detach())
    return model, dict(loss=loss, train_time=time.time() - t0)


def param_deflection(model: ParamRay, bs, n_phi: int = 4001) -> np.ndarray:
    """Deflection read off the parametric network for each b (nan if u' never turns
    within phi_max)."""
    phi = torch.linspace(0, model.cfg["phi_max"], n_phi, dtype=DTYPE)
    out = []
    for b in np.asarray(bs, dtype=float):
        _, du, _ = derivatives(model, phi[:, None], torch.full((n_phi, 1), b, dtype=DTYPE))
        t = first_downturn(phi.numpy(), du.detach()[:, 0].numpy())
        out.append(2 * t - math.pi if t is not None else math.nan)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Inverse: weigh the relativistic term from a few positions on an orbit
# --------------------------------------------------------------------------- #

def make_orbit(p=20.0, e=0.5, f=1.0, n_orbits=3, n_obs=30, noise=0.01, seed=0):
    """A star starting at periapsis, u(0) = (1 + e)/p, u'(0) = 0, observed at n_obs
    random angles over n_orbits turns. The noise is Gaussian, its standard
    deviation a fraction `noise` of 1/p, the orbit's mean u."""
    rng = np.random.default_rng(seed)
    phi_max = 2 * math.pi * n_orbits
    grid, u, _ = rk4_binet((1 + e) / p, 0.0, phi_max, p=p, f=f, h=2e-3)
    phi = np.sort(rng.uniform(0, phi_max, n_obs))
    u_obs = np.interp(phi, grid, u) + noise / p * rng.standard_normal(n_obs)
    return dict(phi=phi, u=u_obs, phi_max=phi_max, p=p, e=e, f=f, noise=noise,
                n_orbits=n_orbits, n_obs=n_obs, seed=seed, grid=grid, u_true=u)


class OrbitPINN(torch.nn.Module):
    """u(phi) = ubar (1 + N(phi)) over the observed span, with p and f trainable.
    ubar, the mean of the observations, sets the scale."""

    def __init__(self, obs: dict, cfg: dict):
        super().__init__()
        self.phi_max = obs["phi_max"]
        self.ubar = float(np.mean(obs["u"]))
        self.net = mlp(1, cfg["width"], cfg["depth"])
        self.log_p = torch.nn.Parameter(torch.tensor(-math.log(self.ubar), dtype=DTYPE))
        self.f = torch.nn.Parameter(torch.tensor(cfg["f_init"], dtype=DTYPE))

    def forward(self, phi):
        return self.ubar * (1 + self.net(2 * phi / self.phi_max - 1))

    def residual(self, phi):
        u, _, d2u = derivatives(self, phi)
        return (d2u + u - torch.exp(-self.log_p) - 3 * self.f * u ** 2) / self.ubar


def fit_inverse(obs: dict, cfg: dict | None = None):
    """Inverse PINN: loss = data misfit + w_phys * orbit-equation residual, both
    relative to ubar. Starts from p = 1/ubar (the Newtonian mean) and f = f_init."""
    cfg = {**INVERSE, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    model = OrbitPINN(obs, cfg)
    phi_obs = torch.tensor(obs["phi"], dtype=DTYPE)[:, None]
    u_obs = torch.tensor(obs["u"], dtype=DTYPE)[:, None]

    def loss_at(phi_col):
        data = ((model(phi_obs) - u_obs) / model.ubar).pow(2).mean()
        return data + cfg["w_phys"] * model.residual(phi_col).pow(2).mean()

    def draw(n):
        return model.phi_max * torch.rand(n, 1, generator=gen, dtype=DTYPE)

    t0 = time.time()
    _adam(model.parameters(), lambda: loss_at(draw(cfg["n_col"])), cfg["adam"], cfg["lr"])
    fixed = draw(cfg["n_fix"])

    def closure():
        model.zero_grad()
        loss = loss_at(fixed)
        loss.backward()
        return loss

    if cfg["lbfgs"]:
        _lbfgs(model.parameters(), closure, cfg["lbfgs"])
    return model, dict(f=float(model.f.detach()), p=float(torch.exp(model.log_p.detach())),
                       loss=float(loss_at(fixed).detach()), time=time.time() - t0)


def _rk4_batch(x, span: float, n: int):
    """RK4 of u'' + u = 1/p + 3 f u^2 for a batch of parameter rows x = (u0, u0', log p,
    f); returns u, u' at the n + 1 nodes, shape (n + 1, batch)."""
    u, v = x[:, 0].copy(), x[:, 1].copy()
    inv_p, f, h = np.exp(-x[:, 2]), x[:, 3], span / n
    us, vs = [u], [v]

    def acc(u):
        return inv_p + 3 * f * u * u - u

    for _ in range(n):
        a1, b1 = v, acc(u)
        a2, b2 = v + h / 2 * b1, acc(u + h / 2 * a1)
        a3, b3 = v + h / 2 * b2, acc(u + h / 2 * a2)
        a4, b4 = v + h * b3, acc(u + h * a3)
        u = u + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        v = v + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        us.append(u)
        vs.append(v)
    return np.array(us), np.array(vs)


def _hermite(us, vs, h, phi):
    """Cubic Hermite interpolation of the RK4 nodes (values and slopes) at phi."""
    i = np.clip((phi / h).astype(int), 0, len(us) - 2)[:, None]
    t = (phi / h)[:, None] - i
    t2, t3 = t * t, t * t * t
    pick = np.take_along_axis
    ib = np.broadcast_to(i, (len(phi), us.shape[1]))
    return ((2 * t3 - 3 * t2 + 1) * pick(us, ib, 0) + (t3 - 2 * t2 + t) * h * pick(vs, ib, 0)
            + (-2 * t3 + 3 * t2) * pick(us, ib + 1, 0) + (t3 - t2) * h * pick(vs, ib + 1, 0))


def fit_shooting(obs: dict, f_inits=(0.0, 0.5, 1.0, 1.5), steps_per_orbit: int = 100,
                 iters: int = 60):
    """Classical least squares: integrate the ODE from (u0, u0') with (p, f) and fit
    all four to the observations (Levenberg-Marquardt, Jacobian by central
    differences). The fit is grown one orbit at a time, since a single orbit pins
    down the phase that later ones would otherwise trap in a wrong winding, from
    each starting f; the best fit is kept."""
    t0 = time.time()
    phi, u_obs = np.asarray(obs["phi"]), np.asarray(obs["u"])
    ubar = float(u_obs.mean())
    first = phi < 2 * math.pi
    a = np.stack([np.ones(int(first.sum())), np.cos(phi[first]), np.sin(phi[first])], 1)
    c0, c1, c2 = np.linalg.lstsq(a, u_obs[first], rcond=None)[0]   # a Kepler ellipse to start
    step = np.array([1e-7, 1e-7, 1e-6, 1e-6])
    best = None
    for f0 in f_inits:
        x = np.array([c0 + c1, c2, -math.log(max(c0, 1e-3)), f0])
        for k in range(1, obs["n_orbits"] + 1):
            span, n = 2 * math.pi * k, steps_per_orbit * k
            sel = phi <= span

            def resid(rows):              # (batch, 4) -> (n_obs, batch)
                us, vs = _rk4_batch(rows, span, n)
                return (_hermite(us, vs, span / n, phi[sel]) - u_obs[sel, None]) / ubar

            lam = 1e-3
            for _ in range(iters):
                rows = np.concatenate([x[None], x + np.diag(step), x - np.diag(step)])
                r = resid(rows)
                r0, jac = r[:, 0], (r[:, 1:5] - r[:, 5:9]) / (2 * step)
                g, hess = jac.T @ r0, jac.T @ jac
                cost = r0 @ r0
                while lam < 1e12:
                    dx = np.linalg.solve(hess + lam * np.diag(np.diag(hess) + 1e-30), -g)
                    r_new = resid((x + dx)[None])[:, 0]
                    if np.isfinite(r_new).all() and r_new @ r_new < cost:
                        x, lam = x + dx, max(lam / 3, 1e-9)
                        break
                    lam *= 4
                else:
                    break
                if np.abs(dx).max() < 1e-12:
                    break
        r0 = resid(x[None])[:, 0]
        loss = float(np.mean(r0 ** 2))
        if np.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, x)
    loss, x = best
    return dict(f=float(x[3]), p=float(math.exp(x[2])), u0=float(x[0]), v0=float(x[1]),
                loss=loss, time=time.time() - t0)


# --------------------------------------------------------------------------- #
# Experiment jobs, run across CPU cores
# --------------------------------------------------------------------------- #

def forward_job(kw: dict) -> dict:
    kw = dict(kw)
    return solve_ray(kw.pop("b"), kw.pop("mode", "march"), kw.pop("cfg", None),
                     kw.pop("phi_max", None))


def param_job(kw: dict) -> dict:
    model, info = train_param(kw.get("cfg"))
    cfg = model.cfg
    bs = np.concatenate([B_CRIT + np.geomspace(cfg["b_lo"] - B_CRIT, 2.0, 60, endpoint=False),
                         np.linspace(B_CRIT + 2.0, cfg["b_hi"], 140)])
    pred, exact = param_deflection(model, bs), exact_deflection(bs)
    rel = np.abs(pred - exact) / exact
    return dict(cfg=cfg, **info, b=bs.tolist(), pred=pred.tolist(), exact=exact.tolist(),
                rel_l2=float(np.linalg.norm(np.nan_to_num(pred - exact, nan=1e3))
                             / np.linalg.norm(exact)),
                max_rel=float(np.nanmax(rel)), n_nan=int(np.isnan(pred).sum()),
                state={k: v.tolist() for k, v in model.state_dict().items()})


def inverse_job(kw: dict) -> dict:
    kw = dict(kw)
    cfg = kw.pop("cfg", None)
    shoot = kw.pop("shooting", True)
    obs = make_orbit(**kw)
    _, pinn = fit_inverse(obs, cfg)
    out = dict(**{k: obs[k] for k in ("p", "e", "f", "noise", "n_orbits", "n_obs", "seed")},
               pinn=pinn)
    if shoot:
        out["shooting"] = fit_shooting(obs)
    return out


def call(job: tuple) -> dict:
    """(job function, its arguments) -> result; lets one pool run mixed jobs."""
    fn, kw = job
    return fn(kw)


def _init_worker():
    torch.set_num_threads(1)


def run_parallel(fn, jobs, workers: int = 4, label=None):
    """Run fn over jobs on `workers` processes, one thread each (the networks are tiny,
    so a core per job beats several cores per job). Results come back in job order."""
    results = [None] * len(jobs)
    t0 = time.time()
    ctx = get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        for n, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            if label:
                print(f"  {label}: {n}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_forward(results: list, path: str):
    """Left: rays from the marching PINN against RK4. Right: deflection error against
    b - b_crit for one network and for marching."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))
    march = sorted([r for r in results if r["mode"] == "march"], key=lambda r: r["b"])
    shown = [r for r in march if np.isfinite(r["deflection"])]
    shown = shown[:: max(1, len(shown) // 4)][:5]
    th = np.linspace(0, 2 * math.pi, 200)
    ax0.fill(np.cos(th) * 2, np.sin(th) * 2, color="black")
    ax0.plot(3 * np.cos(th), 3 * np.sin(th), color="0.6", lw=0.8, ls="--")
    for r in shown:
        b = r["b"]
        phi_t = (math.pi + r["exact"]) / 2
        grid, u, _ = rk4_binet(0.0, 1.0 / b, 2 * phi_t, h=1e-3)
        keep = u > 1 / 40
        ax0.plot(np.cos(grid[keep] - math.pi) / u[keep], np.sin(grid[keep] - math.pi) / u[keep],
                 color="0.75", lw=3)
        ph, uu = np.array(r["phi"]), np.array(r["u"])
        keep = uu > 1 / 40
        ax0.plot(np.cos(ph[keep] - math.pi) / uu[keep], np.sin(ph[keep] - math.pi) / uu[keep],
                 lw=1.3, label=f"b - b_crit = {b - B_CRIT:.2g}")
    ax0.set(xlim=(-25, 15), ylim=(-20, 20), aspect="equal", xlabel="x / M", ylabel="y / M",
            title="Incoming half of each ray: PINN (colour) over RK4 (grey)")
    ax0.legend(fontsize=7, loc="lower left")
    for mode, style in (("single", "o--"), ("march", "s-")):
        rs = sorted([r for r in results if r["mode"] == mode], key=lambda r: r["b"])
        x = np.array([r["b"] - B_CRIT for r in rs])
        y = np.array([r["rel_err"] if np.isfinite(r["rel_err"]) else 1.0 for r in rs])
        lab = "one network over the ray" if mode == "single" else "marching, 1 rad windows"
        ax1.loglog(x, y, style, label=lab)
    ax1.axhline(0.01, color="0.6", lw=0.8, ls=":")
    ax1.set(xlabel="b - b_crit", ylabel="relative deflection error",
            title="How close to the photon sphere")
    ax1.invert_xaxis()
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_param(runs: dict, path: str):
    """Deflection against b for each parametric network, and its relative error."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    first = next(iter(runs.values()))
    b, exact = np.array(first["b"]), np.array(first["exact"])
    ax0.plot(b, exact, color="0.3", lw=3, alpha=0.4, label="exact")
    for name, r in runs.items():
        pred = np.array(r["pred"])
        ax0.plot(b, pred, lw=1.2, label=name)
        ax1.semilogy(b - B_CRIT, np.abs(pred - exact) / exact, lw=1.2, label=name)
    ax0.set(xscale="log", xlabel="b / M", ylabel="deflection (rad)",
            title="One network u(phi, b), physics only")
    ax0.legend(fontsize=8)
    ax1.set(xscale="log", xlabel="b - b_crit", ylabel="relative error",
            title="Deflection error across b")
    ax1.axhline(0.014, color="0.6", lw=0.8, ls=":")
    ax1.text(ax1.get_xlim()[0] * 1.2, 0.016, "data-trained surrogate (1.4%)", fontsize=7,
             color="0.4")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _stats(rows, key, method):
    v = np.array([r[method][key] for r in rows])
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def summarize_inverse(results: list, by: str) -> list:
    """Mean and spread of f and p for each value of `by`, per method and true f."""
    out = []
    keys = sorted({(r["f"], r[by]) for r in results})
    for f_true, val in keys:
        rows = [r for r in results if r["f"] == f_true and r[by] == val]
        row = dict(f_true=f_true, **{by: val}, n=len(rows))
        for m in ("pinn", "shooting"):
            if m in rows[0]:
                row[m] = dict(zip(("f_mean", "f_std"), _stats(rows, "f", m)))
                row[m]["p_err"] = float(np.mean([abs(r[m]["p"] - r["p"]) / r["p"] for r in rows]))
        out.append(row)
    return out


def plot_inverse(example_obs: dict, newton_obs: dict, sweeps: dict, path: str):
    """Left: the observed orbits (GR and Newton). Right: recovered f against noise,
    for both methods, with the spread over noise draws."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))
    for obs, col, lab in ((example_obs, "C0", "GR (f = 1)"), (newton_obs, "C3", "Newton (f = 0)")):
        g, u = obs["grid"], obs["u_true"]
        ax0.plot(np.cos(g) / u, np.sin(g) / u, color=col, lw=1, alpha=0.7, label=lab)
        ax0.plot(np.cos(obs["phi"]) / obs["u"], np.sin(obs["phi"]) / obs["u"], ".", color=col, ms=5)
    ax0.plot(0, 0, "ko", ms=4)
    ax0.set(aspect="equal", xlabel="x / M", ylabel="y / M",
            title=f"{example_obs['n_orbits']} orbits, {example_obs['n_obs']} noisy positions")
    ax0.legend(fontsize=8)
    for (name, rows), col in zip(sweeps.items(), ("C0", "C3", "C2", "C1")):
        x = np.array([r["noise"] for r in rows])
        for m, mk, dx in (("pinn", "o", 1.0), ("shooting", "s", 1.08)):
            if m not in rows[0]:
                continue
            y = np.array([r[m]["f_mean"] for r in rows])
            e = np.array([r[m]["f_std"] for r in rows])
            ax1.errorbar(x * dx, y, e, fmt=mk + ("-" if m == "pinn" else ":"), color=col, ms=4,
                         capsize=2, label=f"{name}, {'PINN' if m == 'pinn' else 'shooting fit'}")
    for f in (0, 1):
        ax1.axhline(f, color="0.6", lw=0.8, ls="--")
    ax1.set(xscale="log", xlabel="noise (fraction of the mean u)", ylabel="recovered f",
            title="Einstein (f = 1) or Newton (f = 0)?")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
