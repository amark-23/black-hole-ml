#include "geodesic.hpp"

#include <cmath>

#include "integrator.hpp"
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

// General initial state at r0 from the constraint p_r^2 = E^2 - V(r0), with
// V = f(r0) * (eps + L^2/r0^2). `inward` sets the sign of p_r. Works for photons
// (eps = 0) and massive particles (eps = 1) alike.
State initial_state(const Schwarzschild& metric, const Constants& c, double r0, bool inward) {
    const double f   = metric.f(r0);
    const double V   = f * (c.eps + c.L * c.L / (r0 * r0));
    const double pr2 = c.E * c.E - V;
    double pr = std::sqrt(pr2 > 0.0 ? pr2 : 0.0);  // clamp tiny negatives at turning points
    if (inward) pr = -pr;

    State y{};
    y[T]   = 0.0;
    y[R]   = r0;
    y[PHI] = 0.0;
    y[PR]  = pr;
    return y;
}

// Photon coming in from r0 with impact parameter b: E = 1, L = b, eps = 0.
State photon_initial_state(const Schwarzschild& metric, double r0, double b) {
    return initial_state(metric, photon_constants(b), r0, /*inward=*/true);
}

// Integrate a trajectory, sampling every accepted adaptive step.
std::vector<Row> trace(const Schwarzschild& metric, const Constants& c, State y,
                       double r_escape, double lambda_max, std::string& outcome,
                       double atol, double rtol, long max_steps) {
    const auto rhs = [&](const State& s) { return geodesic_rhs(s, metric, c); };
    const double r_horizon = metric.horizon() + 1e-3;

    std::vector<Row> rows;
    auto push = [&](double lam, const State& s) {
        rows.push_back({lam, s[R], s[PHI], s[R] * std::cos(s[PHI]), s[R] * std::sin(s[PHI])});
    };

    // Only unbound orbits (E >= 1, includes photons) can escape to infinity.
    // A bound orbit (E < 1) oscillates forever, so it runs until lambda_max.
    const bool unbound = (c.E >= 1.0);

    double lambda = 0.0, h = 0.1;
    push(lambda, y);
    outcome = "max steps reached";

    for (long i = 0; i < max_steps; ++i) {
        lambda += adaptive_step<4>(y, h, rhs, atol, rtol);
        push(lambda, y);
        if (y[R] <= r_horizon)                          { outcome = "captured";  break; }
        if (unbound && y[PR] > 0.0 && y[R] >= r_escape) { outcome = "escaped";   break; }
        if (lambda >= lambda_max)                       { outcome = "max lambda"; break; }
    }
    return rows;
}

}  // namespace bhsim
