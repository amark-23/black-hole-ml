// Command-line trajectory generator: integrate one geodesic and print it as CSV
// to standard output. Redirect to a file and plot it in Python.
//
//   bhsim                          photon, b=6, r0=30   (default)
//   bhsim photon  <b> <r0>         Schwarzschild null geodesic
//   bhsim massive <E> <L> <r0> [lambda_max]   Schwarzschild timelike geodesic
//   bhsim kerr    <a> <b> <r0>     equatorial photon around spin a
//                                  (b > 0 prograde, b < 0 retrograde)
//
//   example:  bhsim kerr 0.9 6 30 > kerr_orbit.csv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "geodesic.hpp"
#include "metric.hpp"

using namespace bhsim;

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "";
    std::string outcome;
    std::vector<Row> rows;
    char status[128];

    if (mode == "kerr") {
        const double a  = (argc > 2) ? std::atof(argv[2]) : 0.9;
        const double b  = (argc > 3) ? std::atof(argv[3]) : 6.0;
        const double r0 = (argc > 4) ? std::atof(argv[4]) : 30.0;
        const Kerr bh{1.0, a};
        KerrState y0 = kerr_equatorial_photon(bh, r0, b);
        rows = trace_kerr(bh, y0, r0 + 1.0, 1e9, outcome);
        std::snprintf(status, sizeof status, "kerr a=%.3f b=%.4f r0=%.1f", a, b, r0);

    } else if (mode == "massive") {
        const double E  = (argc > 2) ? std::atof(argv[2]) : 0.97;
        const double L  = (argc > 3) ? std::atof(argv[3]) : 4.0;
        const double r0 = (argc > 4) ? std::atof(argv[4]) : 20.0;
        const double lambda_max = (argc > 5) ? std::atof(argv[5]) : 1500.0;
        const Schwarzschild bh{1.0};
        const Constants c{E, L, 1.0};
        const State y0 = initial_state(bh, c, r0, /*inward=*/true);
        rows = trace(bh, c, y0, r0 + 1.0, lambda_max, outcome);
        std::snprintf(status, sizeof status, "massive E=%.4f L=%.4f r0=%.1f", E, L, r0);

    } else {
        // photon (with or without the explicit "photon" keyword)
        const int off = (mode == "photon") ? 1 : 0;
        const double b  = (argc > 1 + off) ? std::atof(argv[1 + off]) : 6.0;
        const double r0 = (argc > 2 + off) ? std::atof(argv[2 + off]) : 30.0;
        const Schwarzschild bh{1.0};
        const Constants c = photon_constants(b);
        const State y0 = initial_state(bh, c, r0, /*inward=*/true);
        rows = trace(bh, c, y0, r0 + 1.0, 1e9, outcome);
        std::snprintf(status, sizeof status, "photon b=%.4f r0=%.1f", b, r0);
    }

    std::printf("lambda,r,phi,x,y\n");
    for (const Row& p : rows)
        std::printf("%.8f,%.8f,%.8f,%.8f,%.8f\n", p.lambda, p.r, p.phi, p.x, p.y);

    std::fprintf(stderr, "%s  ->  %s  (%zu points)\n", status, outcome.c_str(), rows.size());
    return 0;
}
