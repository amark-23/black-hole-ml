#pragma once
#include <array>
#include <cstddef>

#include "metric.hpp"

// A geodesic in the equatorial plane: the phase-space state plus the constants
// of motion. The actual equations of motion (the RHS) and the initial-condition
// helper are implemented in schwarzschild.cpp.

namespace bhsim {

// State y = (t, r, phi, p_r), with p_r = dr/dlambda. Four doubles.
using State = std::array<double, 4>;

// Named indices, so we write y[R] instead of y[1] and never mix them up.
enum StateIndex : std::size_t { T = 0, R = 1, PHI = 2, PR = 3 };

// Constants of motion, fixed along a geodesic:
//   E   : conserved energy          ( = f(r) * dt/dlambda )
//   L   : conserved angular momentum ( = r^2 * dphi/dlambda )
//   eps : normalization, 1 = massive (timelike), 0 = photon (null)
struct Constants {
    double E;
    double L;
    double eps;
};

// Impact parameter b = L / E (the physically meaningful quantity for photons).
inline double impact_parameter(const Constants& c) { return c.L / c.E; }

// Photon constants for a given impact parameter: fix the affine scale with
// E = 1, so L = b and eps = 0.
inline Constants photon_constants(double b) { return {1.0, b, 0.0}; }

// --- implemented in schwarzschild.cpp ---

// Right-hand side dy/dlambda of the geodesic equations, from THEORY.md.
State geodesic_rhs(const State& y, const Schwarzschild& metric, const Constants& c);

// Initial state for a photon incoming from radius r0 with impact parameter b.
// Starts at phi = 0, t = 0, with p_r < 0 (moving inward).
State photon_initial_state(const Schwarzschild& metric, double r0, double b);

}  // namespace bhsim
