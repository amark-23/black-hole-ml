// Command-line trajectory generator: integrate one photon orbit and print it as
// CSV to standard output. Redirect to a file and plot it in Python.
//
//   bhsim [b] [r0]
//     b   impact parameter   (default 6.0)   ; b < 3*sqrt(3) ~ 5.196 is captured
//     r0  starting radius     (default 30.0)
//
//   example:  bhsim 6 30 > orbit.csv

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "geodesic.hpp"
#include "integrator.hpp"
#include "metric.hpp"

using namespace bhsim;

int main(int argc, char** argv) {
    const double b  = (argc > 1) ? std::atof(argv[1]) : 6.0;
    const double r0 = (argc > 2) ? std::atof(argv[2]) : 30.0;

    const Schwarzschild bh{1.0};
    const Constants     c   = photon_constants(b);        // E=1, L=b, eps=0
    State               y   = photon_initial_state(bh, r0, b);
    const auto          rhs = [&](const State& s) { return geodesic_rhs(s, bh, c); };

    const double r_horizon = bh.horizon() + 1e-3;  // stop just outside 2M
    const double r_escape  = r0 + 1.0;
    const double atol = 1e-10, rtol = 1e-10;

    // one row per step; the black hole is at the origin
    auto emit = [](double lam, const State& s) {
        const double x = s[R] * std::cos(s[PHI]);
        const double y = s[R] * std::sin(s[PHI]);
        std::printf("%.8f,%.8f,%.8f,%.8f,%.8f\n", lam, s[R], s[PHI], x, y);
    };

    std::printf("lambda,r,phi,x,y\n");
    double lambda = 0.0, h = 0.1;
    emit(lambda, y);

    const char* outcome = "max steps reached";
    for (long i = 0; i < 2000000; ++i) {
        lambda += adaptive_step<4>(y, h, rhs, atol, rtol);
        emit(lambda, y);
        if (y[R] <= r_horizon)               { outcome = "captured"; break; }
        if (y[PR] > 0.0 && y[R] >= r_escape) { outcome = "escaped";  break; }
    }

    // status goes to stderr so it doesn't pollute the CSV on stdout
    std::fprintf(stderr, "b=%.4f  r0=%.1f  ->  %s\n", b, r0, outcome);
    return 0;
}
