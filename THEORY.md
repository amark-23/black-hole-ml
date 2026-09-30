# Theory

Reference for the physics behind the simulator: metrics, equations of motion, and the
analytic results the code is tested against. Filled in step by step as each phase lands.

## Conventions

- Geometrized units: G = c = 1; lengths and times in units of the black-hole mass M.
- Metric signature: (− + + +).
- Coordinates noted per section.

## Schwarzschild

Non-rotating, uncharged black hole of mass $M$. Coordinates
$(t, r, \theta, \varphi)$; event horizon at $r = 2M$.

### Metric

$$
ds^2 = -\left(1 - \frac{2M}{r}\right)dt^2
       + \left(1 - \frac{2M}{r}\right)^{-1}dr^2
       + r^2\,d\theta^2 + r^2\sin^2\theta\,d\varphi^2
$$

Spherical symmetry means every geodesic lies in a plane, so we lose no
generality by fixing the **equatorial plane** $\theta = \pi/2$. The integrator
works there; a general orientation is recovered afterward by a rotation.

### Conserved quantities

The Killing vectors $\partial_t$ and $\partial_\varphi$ give two constants of
motion along each geodesic:

$$
E = \left(1 - \frac{2M}{r}\right)\frac{dt}{d\lambda},
\qquad
L = r^2 \frac{d\varphi}{d\lambda}
$$

The normalization $g_{\mu\nu}u^\mu u^\nu = -\varepsilon$ sets the particle type:
$\varepsilon = 1$ for timelike (massive; $\lambda$ is proper time $\tau$),
$\varepsilon = 0$ for null (photons).

### Geodesic equations (integrator form)

Rather than integrate $(dr/d\lambda)^2 = E^2 - V(r)$ directly — which needs a
sign flip and special handling at each turning point where the square root
vanishes — we integrate the **second-order radial equation**, which is smooth
everywhere. State vector $y = (t, r, \varphi, p_r)$ with $p_r \equiv dr/d\lambda$:

$$
\frac{dt}{d\lambda} = \frac{E}{1 - 2M/r},
\qquad
\frac{dr}{d\lambda} = p_r,
\qquad
\frac{d\varphi}{d\lambda} = \frac{L}{r^2},
$$

$$
\frac{dp_r}{d\lambda} = -\frac{1}{2}\frac{dV}{dr}
= -\frac{\varepsilon M}{r^2} + \frac{L^2}{r^3} - \frac{3ML^2}{r^4},
\qquad
V(r) = \left(1 - \frac{2M}{r}\right)\!\left(\varepsilon + \frac{L^2}{r^2}\right).
$$

This is the exact system the C++ integrator (RK4, then adaptive RK45) advances.
$E$ and $L$ are fixed from the initial conditions and held constant; watching
their drift is the integrator's sanity check.

### State vector and initial conditions

We represent a geodesic by its phase-space **state**

$$
y = (t, r, \varphi, p_r), \qquad p_r \equiv \frac{dr}{d\lambda},
$$

together with the **constants** $(E, L, \varepsilon)$ that stay fixed along the
orbit. Given a state and its constants, the RHS above returns $dy/d\lambda$, so
that pair is everything the integrator needs.

**Photon from an impact parameter.** For null geodesics ($\varepsilon = 0$) the
affine parameter $\lambda$ can be rescaled freely, so only the ratio
$b \equiv L/E$ is physical. We fix the scale by choosing $E = 1$, hence $L = b$.
A photon incoming from far away at $r_0 \gg M$ starts moving inward
($p_r < 0$), with magnitude set by the constraint $p_r^2 = E^2 - V(r_0)$:

$$
p_r(r_0) = -\sqrt{\,E^2 - \left(1 - \frac{2M}{r_0}\right)\frac{L^2}{r_0^2}\,}.
$$

The sign is negative on the way in and flips at a turning point — which the
second-order integrator handles on its own, without us tracking the branch.

**Massive particle.** For $\varepsilon = 1$, both $E$ and $L$ are physical
(energy and angular momentum per unit rest mass) and are set from the orbit we
want.

### Orbit-shape equation (cross-check)

An independent form, useful for verification, drops $\lambda$ and uses
$u \equiv 1/r$ as a function of $\varphi$:

$$
\text{null:}\quad \frac{d^2u}{d\varphi^2} + u = 3M u^2,
\qquad
\text{timelike:}\quad \frac{d^2u}{d\varphi^2} + u = \frac{M}{L^2} + 3M u^2.
$$

The $3Mu^2$ term is the general-relativistic correction; without it these reduce
to the Newtonian conic sections.

### Analytic checkpoints (test targets)

| Quantity | Value | Used to test |
| --- | --- | --- |
| Photon sphere | $r = 3M$ | unstable circular null orbit ($p_r = 0$, $\dot p_r = 0$, $\varepsilon = 0$) |
| Critical impact parameter | $b_\text{crit} = 3\sqrt{3}\,M \approx 5.196\,M$ | $b = L/E$; $b < b_\text{crit}$ captured, $b > b_\text{crit}$ escapes |
| ISCO (massive) | $r = 6M$ | innermost stable circular orbit |
| Weak-field deflection | $\Delta\varphi \to 4M/b$ | large-$b$ photon bending (Einstein) |
| $E$, $L$ | constant | bounded drift over long integration |

## Numerical integration

The geodesic system $dy/d\lambda = f(y)$ has no closed-form solution in general,
so we advance it numerically with explicit Runge–Kutta methods.

**Euler (baseline, for contrast).** Take the current slope and walk straight:

$$
y_{n+1} = y_n + h\,f(y_n).
$$

Simple, but the slope changes *during* the step, so error piles up: local error
$\mathcal{O}(h^2)$, global error $\mathcal{O}(h)$. Too crude for long orbits.

**RK4 (fourth-order Runge–Kutta).** Sample the slope four times across the step
and take a weighted average:

$$
k_1 = f(y_n), \qquad
k_2 = f\!\left(y_n + \tfrac{h}{2}k_1\right), \qquad
k_3 = f\!\left(y_n + \tfrac{h}{2}k_2\right), \qquad
k_4 = f\!\left(y_n + h\,k_3\right),
$$

$$
y_{n+1} = y_n + \frac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right).
$$

The midpoint slopes ($k_2, k_3$) carry double weight. Local truncation error is
$\mathcal{O}(h^5)$ and global error $\mathcal{O}(h^4)$, so halving $h$ cuts the
error roughly 16-fold.

Because $f$ has no explicit $\lambda$ dependence (the system is **autonomous**),
each $k_i$ depends only on a trial state, not on $\lambda$.

**Accuracy diagnostics.** RK4 carries no built-in error estimate, so we monitor
quantities that *should* stay fixed. Two are exact first integrals of the motion:
the constants $E$ and $L$, and the constraint

$$
p_r^2 + V(r) = E^2
$$

(the effective-potential relation rearranged). Their drift over an integration
measures the accumulated truncation error, and a nonzero drift that grows with
step size is the fastest way to catch a wrong RHS. In practice, with RK4 at
$h = 0.01$, the constraint holds to $\sim 10^{-14}$ — machine precision.

**Adaptive RK45 (Phase 1 follow-up).** A fixed step is wasteful: orbits need
tiny steps near the black hole, where the field is steep, and can take large
steps far away. RK45 embeds a 4th- and a 5th-order estimate in the same
evaluations, uses their difference as a local error estimate, and resizes $h$ to
hold that error under a tolerance — small steps where the geometry is sharp,
large steps where it is flat.

## Kerr

### Metric (Boyer–Lindquist)

### Christoffel symbols

### Frame dragging

### Photon orbits vs. spin

## Ray tracing

### Emission model

### Observer geometry

## Neural models

### Capture classifier

### Deflection surrogate

### FNO image map

### Inverse problem
