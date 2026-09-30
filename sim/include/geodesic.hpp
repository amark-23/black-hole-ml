#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>

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

// One sampled point along a trajectory: affine parameter, polar (r, phi), and
// Cartesian (x, y) = (r cos phi, r sin phi) for plotting.
struct Row {
    double lambda, r, phi, x, y;
};

// --- Kerr: full 8-D phase-space state (motion is no longer planar) --------- //

// y = (t, r, theta, phi, p_t, p_r, p_theta, p_phi)
using KerrState = std::array<double, 8>;

enum KerrIndex : std::size_t {
    KT = 0, KR = 1, KTH = 2, KPHI = 3, KPT = 4, KPR = 5, KPTH = 6, KPPH = 7
};

// Hamiltonian RHS dy/dlambda for a Kerr geodesic (implemented in kerr.cpp).
KerrState kerr_rhs(const KerrState& y, const Kerr& metric);

// Equatorial photon incoming from r0 with impact parameter b (E = 1, L_z = b).
// Positive b co-rotates (prograde), negative counter-rotates (retrograde).
KerrState kerr_equatorial_photon(const Kerr& metric, double r0, double b);

// Integrate a Kerr trajectory, sampling every accepted step (x, y in the
// equatorial plane). Same stop conditions as trace(): captured / escaped /
// lambda_max / max_steps.
std::vector<Row> trace_kerr(const Kerr& metric, KerrState y0, double r_escape,
                            double lambda_max, std::string& outcome,
                            double atol = 1e-10, double rtol = 1e-10,
                            long max_steps = 2000000);

// --- implemented in schwarzschild.cpp ---

// Right-hand side dy/dlambda of the geodesic equations, from THEORY.md.
State geodesic_rhs(const State& y, const Schwarzschild& metric, const Constants& c);

// Initial state at radius r0 for the given constants, from the constraint
// p_r^2 = E^2 - V(r0). Starts at t = phi = 0; inward gives p_r < 0.
State initial_state(const Schwarzschild& metric, const Constants& c, double r0, bool inward);

// Convenience: initial state for a photon incoming with impact parameter b.
State photon_initial_state(const Schwarzschild& metric, double r0, double b);

// Integrate from y0 with adaptive RK45, sampling every accepted step. Stops when
// the orbit is captured (reaches the horizon), escapes (passes periapsis and
// climbs back past r_escape), exceeds lambda_max, or hits max_steps. The reason
// is written to `outcome`.
std::vector<Row> trace(const Schwarzschild& metric, const Constants& c, State y0,
                       double r_escape, double lambda_max, std::string& outcome,
                       double atol = 1e-10, double rtol = 1e-10, long max_steps = 2000000);

}  // namespace bhsim
