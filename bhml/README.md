# Machine-learning models (`bhml/`)

Neural models trained on data from the Schwarzschild simulator (see
[../sim/README.md](../sim/README.md)). Data generation, the models, and training
all call the C++ integrator through its Python bindings; the physics is in
[../THEORY.md](../THEORY.md).

## Setup

```
python -m venv .venv
.venv\Scripts\activate                                   # Windows
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The models also need the `_bhsim` extension built (see the sim README's
"Python bindings" section).

## Tasks

### Capture classifier (Phase 2)

Predict whether a photon is captured or escapes, from its initial conditions
`(b, r0)`. Because capture depends only on the impact parameter, the model should
learn a decision boundary at `b_crit = 3√3 M ≈ 5.196` and ignore the decoy `r0`.

```
python -m bhml.data capture      # -> data/capture.npz
python -m bhml.train capture     # -> docs/figures/capture_boundary.png, checkpoint
```

**Result:** ~99.9% validation accuracy; learned boundary ≈ 5.19 vs. analytic
5.196; boundary barely moves with `r0` — the decoy is ignored.

![Learned capture boundary](../docs/figures/capture_boundary.png)

### Deflection surrogate (Phase 3)

Regress the deflection angle `δφ(b)` for escaping photons (`b > b_crit`). The
target spans the weak-field law `δφ → 4M/b` and the strong-field logarithmic
divergence as `b → b_crit`.

```
python -m bhml.data deflection   # -> data/deflection.npz
python -m bhml.train deflection  # -> docs/figures/deflection_fit.png, checkpoint
```

**Result:** ~1.4% validation relative L2 error, and ~200× faster than running the
integrator (batched inference) — the point of a surrogate.

![Deflection surrogate](../docs/figures/deflection_fit.png)
