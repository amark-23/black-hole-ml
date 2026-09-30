// Python bindings for the Schwarzschild geodesic integrator.
// Built as the extension module `_bhsim` when CMake is configured with
// -DBHSIM_PYTHON=ON (see sim/CMakeLists.txt). Exposes two trace functions that
// return the trajectory as an (N, 5) numpy array with columns lambda,r,phi,x,y.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "geodesic.hpp"
#include "metric.hpp"

namespace py = pybind11;
using namespace bhsim;

static py::dict run_trace(const Constants& c, double r0, double lambda_max) {
    const Schwarzschild bh{1.0};
    const State y0 = initial_state(bh, c, r0, /*inward=*/true);

    std::string outcome;
    const std::vector<Row> rows = trace(bh, c, y0, r0 + 1.0, lambda_max, outcome);

    // copy into an (N, 5) numpy array: columns lambda, r, phi, x, y
    py::array_t<double> arr({static_cast<py::ssize_t>(rows.size()), static_cast<py::ssize_t>(5)});
    auto m = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(rows.size()); ++i) {
        m(i, 0) = rows[i].lambda;
        m(i, 1) = rows[i].r;
        m(i, 2) = rows[i].phi;
        m(i, 3) = rows[i].x;
        m(i, 4) = rows[i].y;
    }

    py::dict d;
    d["data"] = arr;
    d["columns"] = py::make_tuple("lambda", "r", "phi", "x", "y");
    d["outcome"] = outcome;
    return d;
}

PYBIND11_MODULE(_bhsim, m) {
    m.doc() = "Schwarzschild geodesic integrator (C++ core)";

    m.def(
        "trace_photon",
        [](double b, double r0, double lambda_max) {
            return run_trace(photon_constants(b), r0, lambda_max);
        },
        py::arg("b"), py::arg("r0") = 30.0, py::arg("lambda_max") = 1e9,
        "Trace a photon of impact parameter b from radius r0.");

    m.def(
        "trace_massive",
        [](double E, double L, double r0, double lambda_max) {
            return run_trace(Constants{E, L, 1.0}, r0, lambda_max);
        },
        py::arg("E"), py::arg("L"), py::arg("r0"), py::arg("lambda_max") = 1500.0,
        "Trace a massive particle with energy E and angular momentum L from r0.");
}
