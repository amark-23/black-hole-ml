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

// dp_r/dlambda at radius r for given L and eps. Independent of p_r and E, so it
// isolates the radial force term of the RHS. Circular orbits are its zeros.
static double dpr_dlambda(const Schwarzschild& bh, double r, double L, double eps) {
    State y{};
    y[R]  = r;
    y[PR] = 0.0;
    const Constants c{1.0, L, eps};
    return geodesic_rhs(y, bh, c)[PR];
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

    // 4. Photon sphere: unstable circular null orbit at r = 3M.
    {
        const double L = 3.0 * std::sqrt(3.0);  // b_crit for E = 1
        // (a) circular condition: dp_r/dlambda = 0 at r = 3M
        const double g3 = dpr_dlambda(bh, 3.0, L, 0.0);
        std::printf("       dp_r/dl at r=3M (photon) = %.2e\n", g3);
        check(std::fabs(g3) < 1e-12, "photon sphere: circular condition at r=3M");
        // (b) a photon launched there with p_r = 0 lingers near r = 3M
        const Constants c = photon_constants(L);
        State y{};
        y[R] = 3.0;
        const auto rhs = [&](const State& s) { return geodesic_rhs(s, bh, c); };
        double max_dev = 0.0;
        for (int i = 0; i < 3000; ++i) {
            y = rk4_step<4>(y, 0.01, rhs);
            max_dev = std::max(max_dev, std::fabs(y[R] - 3.0));
        }
        std::printf("       max |r-3M| over ~2.75 orbits = %.2e\n", max_dev);
        check(max_dev < 1e-3, "photon sphere: orbit stays near r=3M (short integration)");
    }

    // 5. ISCO for massive particles at r = 6M.
    {
        const double eps = 1.0;
        // circular-orbit angular momentum at radius r: L^2 = M r^2 / (r - 3M)
        const auto Lcirc = [&](double r) { return std::sqrt(bh.M * r * r / (r - 3.0 * bh.M)); };
        // stability function: d/dr (dp_r/dlambda) with L fixed at its circular value.
        // Its sign gives orbit stability; it vanishes at the ISCO.
        const auto stab = [&](double r) {
            const double L = Lcirc(r), dr = 1e-4;
            return (dpr_dlambda(bh, r + dr, L, eps) - dpr_dlambda(bh, r - dr, L, eps)) / (2 * dr);
        };

        // (a) circular orbit exists at r = 6M
        const double g6 = dpr_dlambda(bh, 6.0, Lcirc(6.0), eps);
        check(std::fabs(g6) < 1e-12, "ISCO: circular condition at r=6M");
        // (b) marginal stability exactly at r = 6M
        std::printf("       stability(6M) = %.2e\n", stab(6.0));
        check(std::fabs(stab(6.0)) < 1e-5, "ISCO: marginally stable at r=6M");
        // (c) it brackets the ISCO: unstable inside (r=5M), stable outside (r=7M)
        check(stab(5.0) > 0.0 && stab(7.0) < 0.0,
              "ISCO brackets: unstable at r=5M, stable at r=7M");
    }

    // 6. Adaptive RK45: an escaping photon integrated efficiently and accurately.
    {
        const double b = 7.0, r0 = 50.0;
        const Constants c = photon_constants(b);
        State y = photon_initial_state(bh, r0, b);
        const auto rhs = [&](const State& s) { return geodesic_rhs(s, bh, c); };

        double h = 0.1, hmin = 1e30, hmax = 0.0, max_drift = 0.0;
        long steps = 0;
        bool went_in = false;
        for (int i = 0; i < 200000; ++i) {
            const double ht = adaptive_step<4>(y, h, rhs, 1e-11, 1e-11);
            ++steps;
            hmin = std::min(hmin, ht);
            hmax = std::max(hmax, ht);
            const double f = bh.f(y[R]);
            const double V = f * (c.L * c.L) / (y[R] * y[R]);
            max_drift = std::max(max_drift, std::fabs(y[PR] * y[PR] + V - c.E * c.E));
            if (y[R] < 10.0) went_in = true;
            if (went_in && y[R] >= r0) break;
        }
        std::printf("       RK45 steps=%ld, h in [%.2e, %.2e], drift=%.2e\n",
                    steps, hmin, hmax, max_drift);
        check(max_drift < 1e-8,       "RK45: constraint held to tolerance");
        check(steps < 2000,           "RK45: efficient step count");
        check(hmax > 5.0 * hmin,      "RK45: step size adapts");
    }

    // 7. Weak-field light deflection -> 4M/b (Einstein), with the GR 2nd-order term.
    {
        const double PI = std::acos(-1.0);
        const auto deflection = [&](double b, double r0) {
            const Constants c = photon_constants(b);
            State y = photon_initial_state(bh, r0, b);
            const auto rhs = [&](const State& s) { return geodesic_rhs(s, bh, c); };
            double h = 1.0;
            bool went_in = false;
            for (long i = 0; i < 5000000; ++i) {
                adaptive_step<4>(y, h, rhs, 1e-12, 1e-12);
                if (y[R] < 3.0 * b) went_in = true;
                if (went_in && y[R] >= r0) break;
            }
            // subtract the straight-line angle between the two finite-r0 points
            return y[PHI] - PI + 2.0 * std::asin(b / r0);
        };

        const double r0 = 1e6;
        // (a) b = 100: matches the two-term weak-field series to < 1%
        const double d1 = deflection(100.0, r0);
        const double series = 4.0 * bh.M / 100.0 + (15.0 * PI / 4.0) * (bh.M / 100.0) * (bh.M / 100.0);
        std::printf("       deflection(b=100) = %.6e, series = %.6e\n", d1, series);
        check(std::fabs(d1 / series - 1.0) < 0.01, "deflection matches weak-field series (b=100)");
        // (b) b = 500: approaches Einstein's 4M/b to < 2%
        const double d2 = deflection(500.0, r0);
        check(std::fabs(d2 / (4.0 * bh.M / 500.0) - 1.0) < 0.02, "deflection -> 4M/b (b=500)");
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
