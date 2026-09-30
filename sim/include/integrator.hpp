#pragma once
#include <array>
#include <cstddef>

// Generic ODE integrator. Knows nothing about black holes: it advances any
// autonomous first-order system dy/dl = f(y) by one step. The physics lives
// elsewhere (schwarzschild.cpp) and is passed in as `f`.

namespace bhsim {

// One classical fourth-order Runge-Kutta (RK4) step.
//
//   y : current state (N components)
//   h : step size in the affine parameter lambda
//   f : callable (const std::array<double,N>&) -> std::array<double,N>
//       returning dy/dl at a given state.
//
// Returns the state advanced by h. The system is assumed autonomous
// (f has no explicit lambda dependence), which is true for the geodesic
// equations, so f takes only the state.
template <std::size_t N, class F>
std::array<double, N> rk4_step(const std::array<double, N>& y, double h, F&& f) {
    // helper: componentwise  a + s * b
    const auto axpy = [](const std::array<double, N>& a,
                         const std::array<double, N>& b, double s) {
        std::array<double, N> out{};
        for (std::size_t i = 0; i < N; ++i) out[i] = a[i] + s * b[i];
        return out;
    };

    const std::array<double, N> k1 = f(y);                    // slope at start
    const std::array<double, N> k2 = f(axpy(y, k1, 0.5 * h)); // slope at midpoint
    const std::array<double, N> k3 = f(axpy(y, k2, 0.5 * h)); // corrected midpointz
    const std::array<double, N> k4 = f(axpy(y, k3, h));       // slope at end

    std::array<double, N> out{};
    for (std::size_t i = 0; i < N; ++i)
        out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    return out;
}

}  // namespace bhsim
