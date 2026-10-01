#include "raytrace.hpp"

#include <cmath>

#include "integrator.hpp"

// Implementation of the backward ray tracer declared in raytrace.hpp. The heavy
// lifting (the Kerr equations of motion and the adaptive stepper) is reused as
// is from kerr.cpp and integrator.hpp; this file only builds the starting ray
// for each pixel and decides where that ray ends up.

namespace bhsim {

KerrState camera_ray(const Camera& cam, double alpha, double beta) {
    const double si = std::sin(cam.incl), ci = std::cos(cam.incl);

    // View geometry in flat Cartesian coordinates, valid because the camera sits
    // far out where the metric is almost flat. w points from the hole out to the
    // camera; right and up span the image plane.
    //   w     = (sin i, 0, cos i)
    //   right = normalize(zhat x w) = (0, 1, 0)
    //   up    = w x right           = (-cos i, 0, sin i)
    const double wx = si, wz = ci;
    const double ux = -ci, uz = si;

    // Pixel's starting point on the image plane, and its direction (all rays run
    // antiparallel to w, i.e. straight toward the hole).
    const double Px = cam.r_cam * wx + beta * ux;
    const double Py = alpha;
    const double Pz = cam.r_cam * wz + beta * uz;
    const double dx = -wx, dy = 0.0, dz = -wz;

    // Flat-space spherical coordinates of the start point.
    const double r  = std::sqrt(Px * Px + Py * Py + Pz * Pz);
    const double th = std::acos(Pz / r);
    const double ph = std::atan2(Py, Px);
    const double sth = std::sin(th), cth = std::cos(th);
    const double sph = std::sin(ph), cph = std::cos(ph);

    // Project the Cartesian direction onto the spherical basis at the start:
    //   dr/dl        = d . rhat
    //   r dth/dl     = d . thhat
    //   r sinth dph/dl = d . phhat
    const double drdl   = dx * sth * cph + dy * sth * sph + dz * cth;
    const double r_dth  = dx * cth * cph + dy * cth * sph - dz * sth;
    const double rs_dph = -dx * sph + dy * cph;

    // Covariant momenta in the flat far field (g ~ diag(-1, 1, r^2, r^2 sin^2 th)),
    // normalised to energy E = 1 so that p_t = -1.
    KerrState y{};
    y[KT] = 0.0; y[KR] = r; y[KTH] = th; y[KPHI] = ph;
    y[KPT]  = -1.0;
    y[KPR]  = drdl;              // g_rr dr/dl, g_rr ~ 1
    y[KPTH] = r * r_dth;         // r^2 dth/dl
    y[KPPH] = r * sth * rs_dph;  // r^2 sin^2 th dph/dl
    return y;
}

// Redshift factor g = nu_observed / nu_emitted for light leaving the disk at
// radius r (equatorial) and reaching a static observer at infinity. The gas
// there moves on a prograde circular geodesic, so g folds together the Doppler
// shift of that orbital motion with the gravitational shift out of the well.
//
//   static observer at infinity:  nu_obs   = -p_t = E
//   orbiting emitter (u^phi = Omega u^t):  nu_emit = -u^t (p_t + Omega p_phi)
//
// with the Kerr equatorial circular-orbit values (Bardeen 1972), in M units:
//   Omega = sqrt(M) / (r^3/2 + a sqrt(M))
//   u^t   = (r^3/2 + a sqrt(M)) / sqrt(r^3 - 3 M r^2 + 2 a sqrt(M) r^3/2)
// The square root is real down to the circular photon orbit (3M at a = 0), not the
// ISCO (6M); inside that there is no circular orbit and we return 0. The disk
// starts at or outside the ISCO, so in practice r stays well clear of this.
static double disk_redshift(const Kerr& bh, double r, double p_t, double p_phi) {
    const double a = bh.a, M = bh.M;
    const double sM = std::sqrt(M);
    const double r32 = std::pow(r, 1.5);
    const double denom = r * r * r - 3.0 * M * r * r + 2.0 * a * sM * r32;
    if (denom <= 0.0) return 0.0;
    const double omega = sM / (r32 + a * sM);
    const double ut = (r32 + a * sM) / std::sqrt(denom);
    const double nu_emit = -ut * (p_t + omega * p_phi);
    if (nu_emit <= 0.0) return 0.0;
    return (-p_t) / nu_emit;
}

double trace_pixel(const Kerr& bh, const Camera& cam, double alpha, double beta) {
    const double PI = std::acos(-1.0);
    const auto rhs = [&](const KerrState& s) { return kerr_rhs(s, bh); };

    KerrState y = camera_ray(cam, alpha, beta);
    const double r_capture = 1.01 * bh.horizon();
    const double atol = 1e-7, rtol = 1e-7;  // loose: images do not need 1e-10
    const long max_steps = 100000;
    const double r_in = (cam.r_in > 0.0) ? cam.r_in : bh.isco();
    double h = 1.0;

    for (long i = 0; i < max_steps; ++i) {
        const double r_before  = y[KR];
        const double th_before = y[KTH];

        adaptive_step<8>(y, h, rhs, atol, rtol);

        // Crossed the spin axis (theta left [0, pi])? Boyer-Lindquist coordinates
        // are singular there, so a ray that flies over the pole comes out with
        // theta < 0 (or > pi), where the far-side equator sits at -pi/2 and the
        // disk test below would never see it. Map it back onto the same point of
        // space: theta -> -theta, phi -> phi + pi, p_theta -> -p_theta. This is the
        // exact continuation of the geodesic through the axis.
        if (y[KTH] < 0.0 || y[KTH] > PI) {
            y[KTH] = (y[KTH] < 0.0) ? -y[KTH] : 2.0 * PI - y[KTH];
            y[KPHI] += PI;
            y[KPTH] = -y[KPTH];
        }

        // Fell through the horizon: this pixel is in the shadow.
        if (y[KR] <= r_capture) return 0.0;

        // Crossed the equatorial plane? If the disk sits at the crossing radius,
        // the ray stops there (the disk is opaque, and tracing backward the first
        // crossing is the nearest surface the camera sees).
        double f0 = th_before - PI / 2.0;
        const double f1 = y[KTH] - PI / 2.0;
        if (f0 == 0.0) f0 = -1e-30;
        if (f0 * f1 < 0.0) {
            const double frac = f0 / (f0 - f1);
            const double r_cross = r_before + frac * (y[KR] - r_before);
            if (r_cross >= r_in && r_cross <= cam.r_out) {
                const double v = r_in / r_cross;      // in (0, 1], 1 at the edge
                const double emis = v * v;            // inner disk glows brighter
                // p_t and p_phi are conserved, so y still holds their disk values.
                const double g = disk_redshift(bh, r_cross, y[KPT], y[KPPH]);
                return emis * g * g * g * g;          // observed brightness ~ g^4
            }
        }

        // Climbed back out past the camera and is now outbound: empty sky. The
        // p_r > 0 guard is essential: every ray STARTS on the image plane at
        // r = sqrt(r_cam^2 + alpha^2 + beta^2), which is already > r_cam. Without
        // the guard a ray far from the image centre is called sky on entry, before
        // it travels inward, and a wide field silently loses its outer edge.
        if (y[KR] > cam.r_cam && y[KPR] > 0.0) return 0.0;
    }
    return 0.0;  // ran out of steps near the hole: treat as dark
}

std::vector<double> render_image(const Camera& cam) {
    const Kerr bh{1.0, cam.a};
    const int n = cam.res;
    std::vector<double> img(static_cast<std::size_t>(n) * n, 0.0);
    const double step = (n > 1) ? 2.0 * cam.half_width / (n - 1) : 0.0;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n; ++j) {
        const double beta = cam.half_width - j * step;  // row 0 = top of image
        for (int i = 0; i < n; ++i) {
            const double alpha = -cam.half_width + i * step;
            img[static_cast<std::size_t>(j) * n + i] = trace_pixel(bh, cam, alpha, beta);
        }
    }
    return img;
}

}  // namespace bhsim
