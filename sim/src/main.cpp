// Command-line trajectory generator: integrate one geodesic and print it as CSV
// to standard output. Redirect to a file and plot it in Python.
//
//   bhsim                          photon, b=6, r0=30   (default)
//   bhsim photon  <b> <r0>         null geodesic; b < 3*sqrt(3) ~ 5.196 is captured
//   bhsim massive <E> <L> <r0> [lambda_max]
//                                  timelike geodesic (E, L per unit mass);
//                                  bound orbits precess, so lambda_max caps the run
//
//   example:  bhsim photon 6 30 > orbit.csv
//             bhsim massive 0.97 4.0 20 1500 > rosette.csv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "geodesic.hpp"
#include "metric.hpp"

using namespace bhsim;

int main(int argc, char** argv) {
    const Schwarzschild bh{1.0};

    Constants c{};
    double    r0         = 30.0;
    double    lambda_max = 1e9;  // effectively unlimited for unbound orbits

    const bool massive = (argc > 1) && std::strcmp(argv[1], "massive") == 0;
    if (massive) {
        const double E = (argc > 2) ? std::atof(argv[2]) : 0.97;
        const double L = (argc > 3) ? std::atof(argv[3]) : 4.0;
        r0             = (argc > 4) ? std::atof(argv[4]) : 20.0;
        lambda_max     = (argc > 5) ? std::atof(argv[5]) : 1500.0;
        c = Constants{E, L, 1.0};
    } else {
        // photon mode (with or without the explicit "photon" keyword)
        const int off = (argc > 1 && std::strcmp(argv[1], "photon") == 0) ? 1 : 0;
        const double b = (argc > 1 + off) ? std::atof(argv[1 + off]) : 6.0;
        r0             = (argc > 2 + off) ? std::atof(argv[2 + off]) : 30.0;
        c = photon_constants(b);
    }

    const State y0 = initial_state(bh, c, r0, /*inward=*/true);

    std::string outcome;
    const std::vector<Row> rows = trace(bh, c, y0, r0 + 1.0, lambda_max, outcome);

    std::printf("lambda,r,phi,x,y\n");
    for (const Row& p : rows)
        std::printf("%.8f,%.8f,%.8f,%.8f,%.8f\n", p.lambda, p.r, p.phi, p.x, p.y);

    std::fprintf(stderr, "%s  E=%.4f L=%.4f eps=%.0f  r0=%.1f  ->  %s  (%zu points)\n",
                 massive ? "massive" : "photon", c.E, c.L, c.eps, r0, outcome.c_str(), rows.size());
    return 0;
}
