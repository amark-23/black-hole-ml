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

double trace_pixel(const Kerr& bh, const Camera& cam, double alpha, double beta) {
    const double PI = std::acos(-1.0);
    const auto rhs = [&](const KerrState& s) { return kerr_rhs(s, bh); };

    KerrState y = camera_ray(cam, alpha, beta);
    const double r_capture = 1.01 * bh.horizon();
    const double atol = 1e-7, rtol = 1e-7;  // loose: images do not need 1e-10
    const long max_steps = 100000;
    double h = 1.0;

    for (long i = 0; i < max_steps; ++i) {
        const double r_before  = y[KR];
        const double th_before = y[KTH];

        adaptive_step<8>(y, h, rhs, atol, rtol);

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
            if (r_cross >= cam.r_in && r_cross <= cam.r_out) {
                const double v = cam.r_in / r_cross;  // in (0, 1], 1 at the edge
                return v * v;                         // inner disk glows brighter
            }
        }

        // Climbed back out past the camera: empty sky.
        if (y[KR] > cam.r_cam) return 0.0;
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
