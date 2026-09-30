#pragma once

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

}  // namespace bhsim
