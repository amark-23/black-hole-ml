"""Vectorized Kerr ray tracer in PyTorch, for GPU rendering of flythrough frames.

This is a tensor port of the C++ engine: the Kerr inverse-metric terms (generated
by matlab/derive_kerr.m), the Hamiltonian right-hand side, and a backward ray
tracer, all rewritten so that every pixel of a frame is one row of a big state
tensor and a single GPU steps them together. The physics matches simulation/:
the same null geodesics, the same disk with its g^4 Doppler + gravitational
shading, plus a gravitationally lensed starfield behind the hole.
"""

import math

import torch

# State columns, matching the C++ KerrIndex enum.
KT, KR, KTH, KPHI, KPT, KPR, KPTH, KPPH = range(8)


def kerr_terms(r, th, a, M):
    """Inverse-metric components and their r, theta derivatives, elementwise.

    A direct transcription of include/kerr_generated.hpp: same expressions, with
    torch ops in place of std::sin/cos/pow so they run on whole tensors at once.
    Returns a dict of the 15 fields the RHS needs.
    """
    sin, cos = torch.sin, torch.cos
    s2 = sin(th) ** 2
    c2 = cos(th) ** 2
    cs = cos(th) * sin(th)
    Sig = a * a * c2 + r * r                 # Sigma = r^2 + a^2 cos^2 th
    Del = -2.0 * M * r + a * a + r * r       # Delta = r^2 - 2Mr + a^2
    a2r2 = a * a + r * r

    k = {}
    k["gtt"] = -(a2r2 ** 2 - a * a * s2 * Del) / (Sig * Del)
    k["d_r_gtt"] = (
        -(r * a2r2 * 4.0 + a * a * s2 * (M * 2.0 - r * 2.0)) / (Sig * Del)
        + (r * (a2r2 ** 2 - a * a * s2 * Del) * 2.0 / Sig ** 2) / Del
        - ((a2r2 ** 2 - a * a * s2 * Del) * (M * 2.0 - r * 2.0) / Del ** 2) / Sig
    )
    k["d_th_gtt"] = (
        (a * a * cs * 2.0) / Sig
        - (a * a * cs * (a2r2 ** 2 - a * a * s2 * Del) * 2.0 / Sig ** 2) / Del
    )
    k["gtph"] = (-2.0 * M * a * r) / (Sig * Del)
    k["d_r_gtph"] = (
        M * a / Sig ** 2 / Del ** 2
        * (-(a ** 4) * c2 - M * r ** 3 * 4.0 + r ** 4 * 3.0 + a * a * r * r + a * a * r * r * c2)
        * 2.0
    )
    k["d_th_gtph"] = (M * a ** 3 * r * cs * -4.0 / Sig ** 2) / Del
    k["gphph"] = -((1.0 / s2) * (M * r * 2.0 + a * a * s2 - a * a - r * r)) / (Sig * Del)
    k["d_r_gphph"] = (
        (1.0 / s2) / Sig ** 2 / Del ** 2
        * (
            M * r ** 4 * -4.0 + a ** 4 * r + r ** 5 + M * M * r ** 3 * 4.0
            + a * a * r ** 3 * 2.0 - a * a * r ** 3 * s2 * 2.0 - M * a * a * r * r * 4.0
            - a ** 4 * r * s2 - a ** 4 * r * c2 * s2 + M * a * a * r * r * s2 * 3.0
            + M * a ** 4 * c2 * s2
        )
        * -2.0
    )
    k["d_th_gphph"] = (
        (cos(th) / sin(th) ** 3 / Sig ** 2)
        * (
            (a ** 4 * cos(th * 2.0)) / 2.0 - M * r ** 3 * 2.0
            + (a ** 4 * cos(th * 2.0) ** 2) / 4.0 + a ** 4 / 4.0 + r ** 4
            + a * a * r * r + a * a * r * r * cos(th * 2.0) - M * a * a * r * cos(th * 2.0) * 2.0
        )
        * -2.0
    ) / Del
    k["grr"] = Del / Sig
    k["d_r_grr"] = (
        1.0 / Sig ** 2
        * (M * r * r - a * a * r - M * a * a * c2 + a * a * r * c2) * 2.0
    )
    k["d_th_grr"] = a * a * cs / Sig ** 2 * Del * 2.0
    k["gthth"] = 1.0 / Sig
    k["d_r_gthth"] = r / Sig ** 2 * -2.0
    k["d_th_gthth"] = a * a * cs / Sig ** 2 * 2.0
    return k


def kerr_rhs(Y, a, M):
    """dY/dlambda for a batch of Kerr geodesics; mirrors kerr_rhs in kerr.cpp."""
    r, th = Y[:, KR], Y[:, KTH]
    pt, pr, pth, pph = Y[:, KPT], Y[:, KPR], Y[:, KPTH], Y[:, KPPH]
    k = kerr_terms(r, th, a, M)

    dY = torch.zeros_like(Y)
    dY[:, KT] = k["gtt"] * pt + k["gtph"] * pph
    dY[:, KR] = k["grr"] * pr
    dY[:, KTH] = k["gthth"] * pth
    dY[:, KPHI] = k["gtph"] * pt + k["gphph"] * pph

    def quad(a_tt, a_tph, a_phph, a_rr, a_thth):
        return (a_tt * pt * pt + 2.0 * a_tph * pt * pph + a_phph * pph * pph
                + a_rr * pr * pr + a_thth * pth * pth)

    dY[:, KPR] = -0.5 * quad(k["d_r_gtt"], k["d_r_gtph"], k["d_r_gphph"],
                             k["d_r_grr"], k["d_r_gthth"])
    dY[:, KPTH] = -0.5 * quad(k["d_th_gtt"], k["d_th_gtph"], k["d_th_gphph"],
                              k["d_th_grr"], k["d_th_gthth"])
    return dY


def rk4(Y, h, a, M):
    """One classical RK4 step with a per-ray step size h (shape [P, 1])."""
    k1 = kerr_rhs(Y, a, M)
    k2 = kerr_rhs(Y + 0.5 * h * k1, a, M)
    k3 = kerr_rhs(Y + 0.5 * h * k2, a, M)
    k4 = kerr_rhs(Y + h * k3, a, M)
    return Y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# --- disk shading, camera, starfield, and the frame renderer ----------------- #

def disk_redshift(r, p_t, p_ph, a, M=1.0):
    """g = nu_obs / nu_emit for the orbiting disk gas; mirrors disk_redshift()
    in raytrace.cpp. Returns 0 inside the ISCO (no stable circular orbit)."""
    sM = math.sqrt(M)
    r32 = r ** 1.5
    denom = r ** 3 - 3.0 * M * r * r + 2.0 * a * sM * r32
    omega = sM / (r32 + a * sM)
    ut = (r32 + a * sM) / torch.sqrt(torch.clamp(denom, min=1e-9))
    nu_emit = -ut * (p_t + omega * p_ph)
    g = torch.where((denom > 0) & (nu_emit > 0), (-p_t) / nu_emit, torch.zeros_like(r))
    return g


def camera_rays(cam, device):
    """Build the initial Kerr state for every pixel of a perspective camera.

    cam: dict with a, incl (rad), az (rad), dist, fov (rad), res (H, W),
         r_in, r_out. Returns the state tensor Y [P, 8] (P = H*W).
    """
    H, W = cam["res"]
    D, incl, az = cam["dist"], cam["incl"], cam["az"]
    # Camera position and an orthonormal frame looking at the origin.
    C = torch.tensor([D * math.sin(incl) * math.cos(az),
                      D * math.sin(incl) * math.sin(az),
                      D * math.cos(incl)], device=device)
    f = -C / C.norm()
    up_w = torch.tensor([0.0, 0.0, 1.0], device=device)
    right = torch.cross(f, up_w, dim=0)
    right = right / right.norm()
    up = torch.cross(right, f, dim=0)

    tanh = math.tan(0.5 * cam["fov"])
    aspect = W / H
    ys = torch.linspace(1.0, -1.0, H, device=device) * tanh          # top to bottom
    xs = torch.linspace(-1.0, 1.0, W, device=device) * tanh * aspect
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")                   # [H, W]
    d = (f[None, None, :] + gx[..., None] * right[None, None, :]
         + gy[..., None] * up[None, None, :])
    d = d / d.norm(dim=-1, keepdim=True)
    d = d.reshape(-1, 3)                                             # [P, 3]

    r = C.norm()
    th = torch.acos(C[2] / r)
    ph = torch.atan2(C[1], C[0])
    sth, cth, sph, cph = math.sin(th), math.cos(th), math.sin(ph), math.cos(ph)
    rhat = torch.tensor([sth * cph, sth * sph, cth], device=device)
    thhat = torch.tensor([cth * cph, cth * sph, -sth], device=device)
    phhat = torch.tensor([-sph, cph, 0.0], device=device)

    drdl = d @ rhat
    r_dth = d @ thhat
    rs_dph = d @ phhat

    P = d.shape[0]
    Y = torch.zeros(P, 8, device=device)
    Y[:, KR] = r
    Y[:, KTH] = th
    Y[:, KPHI] = ph
    Y[:, KPT] = -1.0
    Y[:, KPR] = drdl
    Y[:, KPTH] = r * r_dth
    Y[:, KPPH] = r * sth * rs_dph
    return Y


def make_starfield(n_stars=22000, th_res=1024, ph_res=2048, seed=7, device="cpu"):
    """A fixed lat-long sky of stars, sampled later by ray direction.

    Each star is splatted a little wider than a pixel so it survives bilinear
    lookup and smears into an arc when the hole lenses that part of the sky.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    sky = torch.zeros(th_res, ph_res)
    ti = torch.randint(1, th_res - 1, (n_stars,), generator=g)
    pj = torch.randint(0, ph_res, (n_stars,), generator=g)
    mag = 0.35 + 0.65 * torch.rand(n_stars, generator=g) ** 3        # mostly faint, few bright
    for dv, du, w in [(0, 0, 1.0), (1, 0, 0.4), (-1, 0, 0.4), (0, 1, 0.4), (0, -1, 0.4)]:
        vi = (ti + dv).clamp(0, th_res - 1)
        ui = (pj + du) % ph_res
        sky.index_put_((vi, ui), mag * w, accumulate=True)
    return sky.clamp(0, 1).to(device)


def sample_starfield(sky, th, ph):
    """Bilinearly sample the sky at ray directions, wrapping in phi."""
    th_res, ph_res = sky.shape
    ph = torch.nan_to_num(ph, nan=0.0, posinf=0.0, neginf=0.0)
    th = torch.nan_to_num(th, nan=0.0, posinf=0.0, neginf=0.0)
    u = ((ph / (2 * math.pi)) % 1.0) * ph_res
    v = torch.clamp(th / math.pi, 0, 1) * (th_res - 1)
    u0 = torch.floor(u).long() % ph_res
    u1 = (u0 + 1) % ph_res
    v0 = torch.clamp(torch.floor(v).long(), 0, th_res - 1)
    v1 = torch.clamp(v0 + 1, 0, th_res - 1)
    fu = u - torch.floor(u)
    fv = v - v0.float()
    return (sky[v0, u0] * (1 - fu) * (1 - fv) + sky[v0, u1] * fu * (1 - fv)
            + sky[v1, u0] * (1 - fu) * fv + sky[v1, u1] * fu * fv)


def inferno_lut(device):
    import matplotlib
    lut = matplotlib.colormaps["inferno"](torch.linspace(0, 1, 256).numpy())[:, :3]
    return torch.tensor(lut, dtype=torch.float32, device=device)


def render_frame(cam, a, sky, lut, disk_scale, device, n_steps=900, C0=0.02,
                 return_raw=False):
    """Trace one frame and return an [H, W, 3] RGB tensor in [0, 1].

    With return_raw=True, also return the raw disk-brightness tensor and the
    per-ray outcome codes, used once to calibrate disk_scale for the whole clip.
    """
    M = 1.0
    r_cap = 1.01 * (M + math.sqrt(max(M * M - a * a, 0.0)))
    r_esc = cam["dist"] * 1.4
    r_in, r_out = cam["r_in"], cam["r_out"]
    PI = math.pi

    Y = camera_rays(cam, device)
    P = Y.shape[0]
    active = torch.ones(P, dtype=torch.bool, device=device)
    otype = torch.zeros(P, dtype=torch.int8, device=device)   # 0 fly 1 horizon 2 disk 3 sky
    bright = torch.zeros(P, device=device)
    dir_th = Y[:, KTH].clone()
    dir_ph = Y[:, KPHI].clone()

    for _ in range(n_steps):
        if int(active.sum()) <= max(1, P // 2000):
            break  # only a few near-critical stragglers left; call them dark
        r_before = Y[:, KR].clone()
        th_before = Y[:, KTH].clone()
        h = torch.clamp(C0 * Y[:, KR:KR + 1], max=0.6)
        Yn = rk4(Y, h, a, M)
        Y = torch.where(active[:, None], Yn, Y)

        r_now = Y[:, KR]
        # horizon
        hit_h = active & (r_now <= r_cap)
        otype[hit_h] = 1
        active[hit_h] = False
        # equatorial crossing into the disk annulus
        f0 = th_before - PI / 2.0
        f1 = Y[:, KTH] - PI / 2.0
        crossed = active & (f0 * f1 < 0)
        if bool(crossed.any()):
            frac = f0 / (f0 - f1)
            r_cross = r_before + frac * (r_now - r_before)
            on_disk = crossed & (r_cross >= r_in) & (r_cross <= r_out)
            if bool(on_disk.any()):
                g = disk_redshift(r_cross, Y[:, KPT], Y[:, KPPH], a, M)
                emis = (r_in / torch.clamp(r_cross, min=1e-3)) ** 2
                b = emis * g ** 4
                bright = torch.where(on_disk, b, bright)
                otype[on_disk] = 2
                active[on_disk] = False
        # escape to the sky
        esc = active & (r_now > r_esc) & (Y[:, KPR] > 0)
        if bool(esc.any()):
            dir_th = torch.where(esc, Y[:, KTH], dir_th)
            dir_ph = torch.where(esc, Y[:, KPHI], dir_ph)
            otype[esc] = 3
            active[esc] = False

    # rays that never finished are still deep in the strong field: call them dark
    # (they belong to the shadow / photon ring), not sky, to avoid stray specks.
    H, W = cam["res"]
    rgb = torch.zeros(P, 3, device=device)
    sky_mask = otype == 3
    star = sample_starfield(sky, dir_th, dir_ph)
    star_rgb = star[:, None] * torch.tensor([0.9, 0.93, 1.0], device=device)  # cool white
    rgb = torch.where(sky_mask[:, None], star_rgb, rgb)
    # disk (inferno by normalized brightness)
    disk_mask = otype == 2
    idx = torch.clamp((bright / disk_scale) * 255, 0, 255).long()
    rgb = torch.where(disk_mask[:, None], lut[idx], rgb)
    frame = rgb.reshape(H, W, 3).clamp(0, 1)
    if return_raw:
        return frame, bright, otype
    return frame
