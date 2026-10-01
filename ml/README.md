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

## Kerr lensing dataset

The next two tasks share one dataset of ray-traced Kerr black holes, published on
Kaggle as [**kerr-lensing**](https://www.kaggle.com/datasets/markopolo2310/kerr-lensing).

It holds 2,000 black holes, with spin $a$ drawn uniformly from $[0, 0.99]$ and
viewing inclination from $[15°, 85°]$, each traced at 128×128 by the GPU ray tracer
in [`simulation/viz/gpu_render.py`](../simulation/viz/gpu_render.py). The framing
is fixed (camera at $40M$, 30° field of view), so every sample shares one image
domain and only the physics changes. The disk runs from the ISCO out to $18M$.

Rather than finished pictures, it stores where every pixel's light ray ended up:

| field | meaning |
| --- | --- |
| `otype` | outcome of the ray: 1 horizon (the shadow), 2 disk, 3 sky, 0 unfinished |
| `hit_r`, `hit_ph` | where the ray crossed the disk, $(r, \varphi)$ |
| `hit_g` | redshift factor $g$ at that crossing (Doppler + gravitational) |
| `sky_th`, `sky_ph` | the direction an escaping ray heads off to on the sky |
| `params` | $(a, \text{inclination})$ of each black hole |
| `split` | train / val / test, 80/10/10 |

Those fields *are* the black hole's lensing map. Push any disk emission profile
$E(r, \varphi)$ through them (`emission_to_image` in
[`datagen/lensing.py`](datagen/lensing.py)) and out comes the observed image, so a
single trace yields unlimited (emission, image) pairs. Pass `hit_g` as well to
include the relativistic shift: the observed brightness is $g^4$ times the
emitted one, so gas moving toward the camera is boosted and gas moving away is
dimmed.

<p align="center">
  <img src="figures/kerr_lensing_sample.png" alt="One dataset sample: per-pixel ray outcome, disk hit radius, and the observed image from a sample emission">
</p>

*One sample ($a = 0.63$, inclination $83°$): the outcome of every pixel's ray, the
disk radius it hit, and the observed image formed by pushing a sample emission
profile through that geometry.*

The data comes from [`notebooks/dataset_lensing_kaggle.ipynb`](notebooks/dataset_lensing_kaggle.ipynb)
(about two minutes on a Kaggle T4), as eight float16 `npz` shards plus a
`manifest.json` describing the schema. In a Kaggle notebook, use **Add Data →
kerr-lensing**, then:

```python
import glob, numpy as np
shards = sorted(glob.glob("/kaggle/input/kerr-lensing/**/shard_*.npz", recursive=True))
z = np.load(shards[0])   # otype, hit_r, hit_ph, hit_g, sky_th, sky_ph, params, split
```

## Emission to image: a Fourier Neural Operator

Planned, and first up. This is the function-to-function task, and the one place
the [`fno-pde`](https://github.com/amark-23/fno-pde) models apply directly: its
from-scratch FNO and its U-Net baseline carry over, now mapping one image to
another. Resolution transfer is the headline: high-resolution ray tracing is
expensive, so training at 64² and evaluating at 256² is a real payoff, not just a
benchmark.

- **FNO2d**: emission profile → observed image, with spin and inclination as
  constant input channels.
- **(x, y) coordinate channels**: the map is not translation-invariant, since the
  photon ring sits at a fixed place in the image.
- **U-Net baseline** at a matched parameter count.
- **Resolution transfer**: train at 64², evaluate at 128² and 256².
- **Speed** against the ray tracer.

The published set is 128². For the other resolutions, run the same generator with
`res=64` and `res=256`: the same `n` and seed give the same black holes and
splits, and the framing is fixed, so every resolution samples the same image
domain. Taking every other pixel of the 128² grid would not quite: its pixel
centres run edge to edge, so the subsampled grid comes out slightly off-centre.

## Spin and inclination from an image: a CNN

Planned, after the FNO. The inverse problem, on the same dataset:

- **CNN**: image → (spin, inclination).
- **Noise and blur robustness**: degrade the test images step by step, and find
  how much it takes before the estimates fail.

## Notebooks

`notebooks/kaggle_smoke.ipynb` is a smoke test for running the project on Kaggle:
it clones the repo, builds the C++ core, runs the tests, and imports the package,
all on CPU. It is the template for the GPU-bound training that later tasks will
need.

`notebooks/dataset_lensing_kaggle.ipynb` generates the Kerr lensing dataset above
on a Kaggle GPU and previews a sample.
