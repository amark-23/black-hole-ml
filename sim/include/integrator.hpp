#pragma once
#include <algorithm>
#include <array>
#include <cmath>
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
    const std::array<double, N> k3 = f(axpy(y, k2, 0.5 * h)); // corrected midpoint
    const std::array<double, N> k4 = f(axpy(y, k3, h));       // slope at end

    std::array<double, N> out{};
    for (std::size_t i = 0; i < N; ++i)
        out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    return out;
}

// --- Adaptive Dormand-Prince RK5(4) --------------------------------------- //

// One embedded Dormand-Prince step. Advances y by h and writes the 5th-order
// result into y5 and the error estimate (y5 - y4) into err. Seven stages; the
// last reuses the new point (FSAL: k7 = f(y5)).
template <std::size_t N, class F>
void dopri54(const std::array<double, N>& y, double h, F&& f,
             std::array<double, N>& y5, std::array<double, N>& err) {
    // Dormand-Prince coefficients.
    constexpr double a21 = 1.0 / 5;
    constexpr double a31 = 3.0 / 40, a32 = 9.0 / 40;
    constexpr double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
    constexpr double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187,
                     a53 = 64448.0 / 6561, a54 = -212.0 / 729;
    constexpr double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247,
                     a64 = 49.0 / 176, a65 = -5103.0 / 18656;
    // 5th-order solution weights (b2 = 0).
    constexpr double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192,
                     b5 = -2187.0 / 6784, b6 = 11.0 / 84;
    // Error weights d = b - b* (5th minus 4th order).
    constexpr double d1 = 71.0 / 57600, d3 = -71.0 / 16695, d4 = 71.0 / 1920,
                     d5 = -17253.0 / 339200, d6 = 22.0 / 525, d7 = -1.0 / 40;

    std::array<double, N> t{};
    const std::array<double, N> k1 = f(y);
    for (std::size_t i = 0; i < N; ++i) t[i] = y[i] + h * (a21 * k1[i]);
    const std::array<double, N> k2 = f(t);
    for (std::size_t i = 0; i < N; ++i) t[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    const std::array<double, N> k3 = f(t);
    for (std::size_t i = 0; i < N; ++i)
        t[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    const std::array<double, N> k4 = f(t);
    for (std::size_t i = 0; i < N; ++i)
        t[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    const std::array<double, N> k5 = f(t);
    for (std::size_t i = 0; i < N; ++i)
        t[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    const std::array<double, N> k6 = f(t);
    for (std::size_t i = 0; i < N; ++i)
        y5[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
    const std::array<double, N> k7 = f(y5);  // FSAL
    for (std::size_t i = 0; i < N; ++i)
        err[i] = h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i] + d7 * k7[i]);
}

// Take one adaptive step from y with trial size *h. If the scaled error norm is
// <= 1 the step is accepted and y advances; otherwise *h is shrunk and the step
// retried. On return *h holds the size suggested for the NEXT step, and the
// return value is the step size actually taken (the lambda advanced).
template <std::size_t N, class F>
double adaptive_step(std::array<double, N>& y, double& h, F&& f,
                     double atol = 1e-10, double rtol = 1e-10) {
    constexpr double SAFETY = 0.9, MIN_FAC = 0.2, MAX_FAC = 5.0, H_FLOOR = 1e-14;
    std::array<double, N> y5{}, err{};

    for (;;) {
        dopri54<N>(y, h, f, y5, err);

        // scaled RMS error norm
        double sum = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double sc = atol + rtol * std::max(std::fabs(y[i]), std::fabs(y5[i]));
            const double ratio = err[i] / sc;
            sum += ratio * ratio;
        }
        const double errn = std::sqrt(sum / static_cast<double>(N));
        const double h_taken = h;

        double fac = (errn > 0.0) ? SAFETY * std::pow(errn, -0.2) : MAX_FAC;
        fac = std::min(MAX_FAC, std::max(MIN_FAC, fac));

        if (errn <= 1.0 || h_taken <= H_FLOOR) {
            y = y5;         // accept
            h = h * fac;    // suggested size for the next step
            return h_taken;
        }
        h = h * fac;        // reject: shrink and retry
    }
}

}  // namespace bhsim
