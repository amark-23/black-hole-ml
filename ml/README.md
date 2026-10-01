# Machine learning on the black hole

Neural models trained on data from the simulator in [`../simulation`](../simulation).
The idea is to let a network learn the input-to-output map that the physics
defines, then use it either as a fast stand-in for the integrator or to solve
problems the integrator cannot solve directly (like reading spin off an image).

Each task follows the same shape: generate a dataset by running the simulator,
train a small model, and check what it learned against the physics.

## Setup

```
python -m venv .venv
.venv\Scripts\activate                                    # Windows
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The data generators call the simulator through its Python bindings, so build the
`_bhsim` extension once (see the "Calling the simulator from Python" section of
the [simulation README](../simulation/README.md)). CPU PyTorch is plenty; the
models here are tiny.

The package is imported as `bhml` (it lives at `ml/bhml`). Run everything from the
repo root, for example `python -m bhml.data capture`.

## Capture classifier

The first task: from a photon's starting conditions, predict whether it is
captured or escapes. This is a binary classification problem.

For a non-rotating hole the answer depends only on the impact parameter: a photon
is captured exactly when $b < b_\text{crit} = 3\sqrt3\,M$. So a model trained on
$b$ should discover a decision boundary sitting at $3\sqrt3 \approx 5.196$. We
also feed it the start radius $r_0$ as a decoy, since capture does not depend on
it, to see whether the model correctly learns to ignore an irrelevant input.

This task is deliberately easy. Its value is twofold: it exercises the whole
pipeline end to end (simulator, dataset, training, evaluation), and it lets the
physics grade the result. If the learned boundary lands on $3\sqrt3$, both the
integrator and the training loop are working.

```
python -m bhml.data capture      # -> data/capture.npz
python -m bhml.train capture     # -> ml/figures/capture_boundary.png, checkpoint
```

Result: about 99.9% validation accuracy; the learned boundary sits at roughly
5.19 against the analytic 5.196; and the boundary barely moves as $r_0$ varies,
so the decoy is ignored.

<p align="center">
  <img src="figures/capture_boundary.png" alt="Learned capture boundary">
</p>

## Deflection surrogate

The second task: predict a photon's total deflection angle $\delta\varphi$ from
its impact parameter (for escaping photons, $b > b_\text{crit}$). This is a
regression, and the target spans two very different regimes:

- far out ($b \gg b_\text{crit}$) the bending is gentle, $\delta\varphi \to 4M/b$,
  Einstein's weak-field result;
- close to the threshold ($b \to b_\text{crit}$) the deflection diverges like
  $-\ln(b - b_\text{crit})$, because the photon whirls many times around the
  photon sphere before escaping.

A trained surrogate returns the deflection far faster than integrating the
geodesic, which is the point: pay the integration cost once to build the dataset,
then evaluate almost for free afterward.

```
python -m bhml.data deflection   # -> data/deflection.npz
python -m bhml.train deflection  # -> ml/figures/deflection_fit.png, checkpoint, speed benchmark
```

Result: about 1.4% relative L2 error across the whole range, and roughly 200 times
faster than the integrator (batched inference).

<p align="center">
  <img src="figures/deflection_fit.png" alt="Deflection surrogate">
</p>

## Notebooks

`notebooks/kaggle_smoke.ipynb` is a smoke test for running the project on Kaggle:
it clones the repo, builds the C++ core, runs the tests, and imports the package,
all on CPU. It is the template for the GPU-bound training that later tasks will
need.

## What comes next

- An image-to-image model (a Fourier Neural Operator) mapping an accretion-disk
  emission profile to a ray-traced image. tba.
- An inverse model that reads spin and inclination back out of an image. tba.
