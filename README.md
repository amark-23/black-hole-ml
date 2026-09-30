# black-hole-ml

A C++ black-hole geodesic simulator paired with machine-learning models trained on its output.

The project models light and matter around Schwarzschild and Kerr black holes with an
efficient C++ core, then trains neural models on the generated data:

- a **capture classifier** (is a photon captured or does it escape?),
- a **deflection-angle surrogate** (fast regression replacing the integrator),
- a **Fourier Neural Operator** mapping accretion-disk emission to ray-traced images, and
- a **CNN inverse model** that recovers spin and inclination from an image.

Physics runs in C++ (OpenMP, optional CUDA); the models and data pipelines are in Python
(PyTorch). GPU-bound work runs in a Kaggle notebook that clones this repo; everything else
runs on CPU. The Fourier Neural Operator is implemented from scratch in this repo; the
companion repository [`fno-pde`](https://github.com/amark-23/fno-pde) is kept only as a
reference for the approach, not as a dependency.

See [`THEORY.md`](THEORY.md) for the physics and the equations.

## Layout

```
sim/        C++ core: metrics, integrators, ray tracing (CMake, OpenMP, optional CUDA)
bhml/       Python package: data generation, models, training
matlab/     symbolic derivations and reference-trajectory fixtures (dev-time only)
configs/    run configurations
notebooks/  Colab notebooks for GPU work
tests/      C++ tests live in sim/tests
```

## Status

Phase 0 — scaffolding. Structure only; no implementation yet.
