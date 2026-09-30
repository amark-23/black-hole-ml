// Phase 1 tests: integrate real Schwarzschild photon orbits and check them
// against the analytic checkpoints in THEORY.md. Returns nonzero on any
// failure, which CTest reports as a failed test.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "geodesic.hpp"
#include "integrator.hpp"
#include "metric.hpp"

using namespace bhsim;

// Integrate a photon of impact parameter b inward from r0.
// Returns true if it reaches the horizon (captured), false if it climbs back
// out past r0 (escaped).
static bool photon_captured(const Schwarzschild& bh, double b, double r0 = 50.0) {
    const Constants c = photon_constants(b);
    State y = photon_initial_state(bh, r0, b);
    const auto rhs = [&](const State& s) { return geodesic_rhs(s, bh, c); };

    const double h         = 0.01;
    const double r_capture = 2.0 * bh.M + 0.01;  // just outside the horizon
    const double r_escape  = r0 + 1.0;

    for (int step = 0; step < 2000000; ++step) {
        y = rk4_step<4>(y, h, rhs);
        if (!std::isfinite(y[R])) return true;   // blew up near the horizon
        if (y[R] <= r_capture) return true;      // fell in
        if (y[R] >= r_escape)  return false;     // came back out
    }
    return false;  // unresolved (asymptotes to photon sphere): treat as escape
}

static int failures = 0;
static void check(bool cond, const char* msg) {
    std::printf("%-6s %s\n", cond ? "ok:" : "FAIL:", msg);
    if (!cond) ++failures;
}

int main() {
    const Schwarzschild bh{1.0};
    const double b_crit = 3.0 * std::sqrt(3.0);  // = 3*sqrt(3)*M ~ 5.196

    // 1. Capture / escape on either side of the critical impact parameter.
    check(photon_captured(bh, b_crit - 0.2),  "b < b_crit is captured");
    check(!photon_captured(bh, b_crit + 0.2), "b > b_crit escapes");

    // 2. Bisect the capture/escape boundary and compare to 3*sqrt(3)*M.
    double lo = 4.0, hi = 7.0;  // lo captured, hi escapes
    for (int i = 0; i < 40; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (photon_captured(bh, mid)) lo = mid; else hi = mid;
    }
    const double b_found = 0.5 * (lo + hi);
    std::printf("       b_crit found = %.5f, expected = %.5f\n", b_found, b_crit);
    check(std::fabs(b_found - b_crit) < 1e-2, "numerical b_crit matches 3*sqrt(3) M");

    // 3. Hamiltonian constraint p_r^2 + V(r) - E^2 stays ~ 0 along an orbit.
    {
        const double b = 7.0;  // an escaping photon
        const Constants c = photon_constants(b);
        State y = photon_initial_state(bh, 50.0, b);
        const auto rhs = [&](const State& s) { return geodesic_rhs(s, bh, c); };
        double max_drift = 0.0;
        for (int i = 0; i < 20000; ++i) {
            y = rk4_step<4>(y, 0.01, rhs);
            if (y[R] < 3.0 || y[R] > 60.0) break;
            const double f = bh.f(y[R]);
            const double V = f * (c.L * c.L) / (y[R] * y[R]);  // eps = 0
            const double H = y[PR] * y[PR] + V - c.E * c.E;
            max_drift = std::max(max_drift, std::fabs(H));
        }
        std::printf("       max |constraint drift| = %.2e\n", max_drift);
        check(max_drift < 1e-6, "Hamiltonian constraint conserved");
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
