#pragma once
#include <cmath>

#include "kerr_generated.hpp"  // auto-generated kerr_terms(), from matlab/derive_kerr.m

// The spacetime geometry. For Phase 1 this is just Schwarzschild, described by a
// single number (the mass M) and the metric potential f(r) = 1 - 2M/r that
// appears throughout the equations of motion. Per-orbit quantities (E, L, the
// particle type) live in geodesic.hpp, not here.

namespace bhsim {

struct Schwarzschild {
    double M = 1.0;  // black-hole mass; sets the length/time scale (G = c = 1)

    // Metric potential f(r) = 1 - 2M/r.
    //   f > 0 outside the horizon, f = 0 at r = 2M, f < 0 inside.
    // Appears in g_tt = -f, g_rr = 1/f, and in dt/dl = E / f.
    double f(double r) const { return 1.0 - 2.0 * M / r; }

    // Event horizon radius, where f(r) = 0.
    double horizon() const { return 2.0 * M; }
};

// Rotating (Kerr) black hole: mass M and spin a = J/M, with 0 <= a <= M.
// The heavy algebra (inverse metric + derivatives) lives in kerr_terms();
// this struct just carries the parameters and the basic geometry.
struct Kerr {
    double M = 1.0;
    double a = 0.0;  // spin; a = 0 recovers Schwarzschild

    double Sigma(double r, double th) const {
        const double c = std::cos(th);
        return r * r + a * a * c * c;
    }
    double Delta(double r) const { return r * r - 2.0 * M * r + a * a; }

    // Outer event horizon r_+ = M + sqrt(M^2 - a^2).
    double horizon() const { return M + std::sqrt(M * M - a * a); }

    // Prograde innermost stable circular orbit (Bardeen, Press & Teukolsky 1972):
    // 6M at a = 0, shrinking toward M as a -> M. Where an accretion disk ends.
    double isco() const {
        const double x = a / M;
        const double z1 = 1.0 + std::cbrt(1.0 - x * x)
                                * (std::cbrt(1.0 + x) + std::cbrt(1.0 - x));
        const double z2 = std::sqrt(3.0 * x * x + z1 * z1);
        return M * (3.0 + z2 - std::sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
    }
};

}  // namespace bhsim
