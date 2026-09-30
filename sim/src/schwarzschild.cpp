#include "geodesic.hpp"

#include <cmath>

#include "metric.hpp"

// Equatorial Schwarzschild geodesics. Implements the two functions declared in
// geodesic.hpp, exactly as written in THEORY.md ("Geodesic equations" and
// "State vector and initial conditions").

namespace bhsim {

// dy/dlambda for y = (t, r, phi, p_r).
State geodesic_rhs(const State& y, const Schwarzschild& metric, const Constants& c) {
    const double r   = y[R];
    const double M   = metric.M;
    const double f   = metric.f(r);   // 1 - 2M/r
    const double L2  = c.L * c.L;

    State dydl{};
    dydl[T]   = c.E / f;              // dt/dlambda   = E / (1 - 2M/r)
    dydl[R]   = y[PR];               // dr/dlambda   = p_r
    dydl[PHI] = c.L / (r * r);       // dphi/dlambda = L / r^2
    dydl[PR]  = -c.eps * M / (r * r) // dp_r/dlambda = -(1/2) dV/dr
              + L2 / (r * r * r)
              - 3.0 * M * L2 / (r * r * r * r);
    return dydl;
}

// Photon coming in from r0 with impact parameter b: E = 1, L = b, eps = 0,
// starting at t = phi = 0 and moving inward (p_r < 0). The magnitude of p_r
// comes from the constraint p_r^2 = E^2 - V(r0), with V = f * L^2 / r^2 (eps=0).
State photon_initial_state(const Schwarzschild& metric, double r0, double b) {
    const Constants c = photon_constants(b);         // {E=1, L=b, eps=0}
    const double f    = metric.f(r0);
    const double pr2  = c.E * c.E - f * (c.L * c.L) / (r0 * r0);
    const double pr   = -std::sqrt(pr2 > 0.0 ? pr2 : 0.0);  // inward; clamp tiny negatives

    State y{};
    y[T]   = 0.0;
    y[R]   = r0;
    y[PHI] = 0.0;
    y[PR]  = pr;
    return y;
}

}  // namespace bhsim
