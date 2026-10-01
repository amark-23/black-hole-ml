// Phase 4 tests: validate the Kerr Hamiltonian integrator three ways —
//   1. a = 0 reproduces the Schwarzschild integrator (equatorial photon),
//   2. it matches the MATLAB ode113 reference fixture (a = 0.5),
//   3. the Hamiltonian H and the constants E, L_z stay conserved,
//   4. the ISCO formula hits its known values,
//   5. a ray flying over the spin axis still finds the disk behind the hole.
// Returns nonzero on any failure (CTest reports it as failed).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "geodesic.hpp"
#include "integrator.hpp"
#include "metric.hpp"
#include "raytrace.hpp"

using namespace bhsim;

static int failures = 0;
static void check(bool cond, const char* msg) {
    std::printf("%-6s %s\n", cond ? "ok:" : "FAIL:", msg);
    if (!cond) ++failures;
}

// super-Hamiltonian H = 1/2 g^{ab} p_a p_b (0 for photons, -1/2 for massive)
static double kerr_H(const KerrState& y, const Kerr& bh) {
    const KerrTerms k = kerr_terms(y[KR], y[KTH], bh.a, bh.M);
    const double pt = y[KPT], pr = y[KPR], pth = y[KPTH], pph = y[KPPH];
    return 0.5 * (k.gtt * pt * pt + 2.0 * k.gtph * pt * pph + k.gphph * pph * pph
                  + k.grr * pr * pr + k.gthth * pth * pth);
}

// equatorial photon (theta = pi/2, p_theta = 0), inward, from the H = 0 constraint
static KerrState kerr_photon_eq(const Kerr& bh, double r0, double b) {
    const double PI = std::acos(-1.0);
    KerrState y{};
    y[KT] = 0.0; y[KR] = r0; y[KTH] = PI / 2; y[KPHI] = 0.0;
    y[KPT] = -1.0; y[KPPH] = b; y[KPTH] = 0.0;
    const KerrTerms k = kerr_terms(r0, PI / 2, bh.a, bh.M);
    const double pt = y[KPT], pph = y[KPPH];
    const double pr2 = -(k.gtt * pt * pt + 2.0 * k.gtph * pt * pph + k.gphph * pph * pph) / k.grr;
    y[KPR] = -std::sqrt(pr2 > 0.0 ? pr2 : 0.0);
    return y;
}

int main() {
    // ---- 1. a = 0 reduces to Schwarzschild (equatorial photon) ------------ //
    {
        const Kerr kerr0{1.0, 0.0};
        const Schwarzschild schw{1.0};
        const double b = 6.0, r0 = 20.0;

        KerrState yk = kerr_photon_eq(kerr0, r0, b);
        const Constants c = photon_constants(b);
        State ys = photon_initial_state(schw, r0, b);

        const auto rhs_k = [&](const KerrState& s) { return kerr_rhs(s, kerr0); };
        const auto rhs_s = [&](const State& s) { return geodesic_rhs(s, schw, c); };

        double max_diff = 0.0;
        for (int i = 0; i < 3000; ++i) {
            yk = rk4_step<8>(yk, 0.01, rhs_k);
            ys = rk4_step<4>(ys, 0.01, rhs_s);
            max_diff = std::max(max_diff, std::fabs(yk[KR] - ys[R]));
            max_diff = std::max(max_diff, std::fabs(yk[KPHI] - ys[PHI]));
            if (yk[KR] < 2.1 || ys[R] < 2.1) break;
        }
        std::printf("       max |Kerr(a=0) - Schwarzschild| = %.2e\n", max_diff);
        check(max_diff < 1e-6, "Kerr a=0 matches Schwarzschild (equatorial photon)");
    }

    // ---- load the MATLAB reference fixture (a = 0.5) ---------------------- //
    std::vector<std::array<double, 9>> rows;  // lambda, t, r, th, phi, pt, pr, pth, pphi
    {
        std::ifstream f(KERR_FIXTURE);
        check(static_cast<bool>(f), "opened Kerr fixture");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#' || line.rfind("lambda", 0) == 0) continue;
            std::array<double, 9> row{};
            std::stringstream ss(line);
            std::string cell;
            int i = 0;
            while (i < 9 && std::getline(ss, cell, ',')) row[i++] = std::stod(cell);
            if (i == 9) rows.push_back(row);
        }
    }

    if (rows.size() < 2) {
        std::printf("FAIL:  fixture has too few rows\n");
        return 1;
    }

    const Kerr bh{1.0, 0.5};
    const auto& first = rows.front();
    const auto& last = rows.back();
    KerrState y0{};
    for (int i = 0; i < 8; ++i) y0[i] = first[i + 1];  // t,r,th,phi,pt,pr,pth,pphi
    const double lam_final = last[0];

    // ---- 2. match the ode113 reference at lambda_final -------------------- //
    {
        KerrState y = y0;
        const auto rhs = [&](const KerrState& s) { return kerr_rhs(s, bh); };
        double lam = 0.0, h = 0.1;
        while (lam < lam_final - 1e-12) {
            const double hmax = lam_final - lam;
            if (h > hmax) h = hmax;
            lam += adaptive_step<8>(y, h, rhs, 1e-11, 1e-11);
            if (h > hmax) h = hmax;
        }
        const double dr = std::fabs(y[KR] - last[2]);
        const double dth = std::fabs(y[KTH] - last[3]);
        const double dphi = std::fabs(y[KPHI] - last[4]);
        std::printf("       fixture diff at lambda=%.1f: dr=%.2e dtheta=%.2e dphi=%.2e\n",
                    lam_final, dr, dth, dphi);
        check(dr < 1e-4 && dth < 1e-4 && dphi < 1e-4, "matches MATLAB ode113 reference (a=0.5)");
    }

    // ---- 3. conserved quantities along the orbit ------------------------- //
    {
        KerrState y = y0;
        const auto rhs = [&](const KerrState& s) { return kerr_rhs(s, bh); };
        const double H0 = kerr_H(y, bh);
        const double E0 = -y[KPT], Lz0 = y[KPPH];
        double lam = 0.0, h = 0.1, max_dH = 0.0, max_dE = 0.0, max_dLz = 0.0;
        while (lam < lam_final - 1e-12) {
            const double hmax = lam_final - lam;
            if (h > hmax) h = hmax;
            lam += adaptive_step<8>(y, h, rhs, 1e-11, 1e-11);
            if (h > hmax) h = hmax;
            max_dH = std::max(max_dH, std::fabs(kerr_H(y, bh) - H0));
            max_dE = std::max(max_dE, std::fabs(-y[KPT] - E0));
            max_dLz = std::max(max_dLz, std::fabs(y[KPPH] - Lz0));
        }
        std::printf("       drift: |dH|=%.2e |dE|=%.2e |dLz|=%.2e\n", max_dH, max_dE, max_dLz);
        check(max_dH < 1e-6, "Kerr Hamiltonian conserved");
        check(max_dE < 1e-12 && max_dLz < 1e-12, "E and L_z exactly conserved");
    }

    {
        // 4. ISCO: 6M without spin, M at extremal spin, ~2.3209M at a = 0.9.
        check(std::fabs(Kerr{1.0, 0.0}.isco() - 6.0) < 1e-12, "ISCO = 6M at a = 0");
        check(std::fabs(Kerr{1.0, 1.0}.isco() - 1.0) < 1e-12, "ISCO = M at a = M");
        check(std::fabs(Kerr{1.0, 0.9}.isco() - 2.320883) < 1e-6, "ISCO = 2.3209M at a = 0.9");
    }

    {
        // 5. The pixel column alpha = 0 lies in the plane of the spin axis, so its
        //    rays above the hole pass straight over the pole. They must still see
        //    the far side of the disk lensed up over the shadow, like their
        //    neighbours, rather than a black line through the image.
        Camera cam;
        cam.a = 0.9;
        const Kerr bh{1.0, cam.a};
        bool ok = true;
        for (double beta : {5.0, 6.0, 7.0, 8.0}) {
            const double on_axis = trace_pixel(bh, cam, 0.0, beta);
            const double beside  = trace_pixel(bh, cam, 0.06, beta);
            if (!(beside > 0.0 && std::fabs(on_axis - beside) < 0.05 * beside)) ok = false;
        }
        check(ok, "ray over the spin axis sees the lensed disk (no axis seam)");
    }

    std::printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
