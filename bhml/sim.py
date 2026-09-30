"""Pythonic wrapper over the compiled `_bhsim` extension.

Requires the extension to be built:
    cmake -S sim -B sim/build -DBHSIM_PYTHON=ON
    cmake --build sim/build --config Release

Then, from the repo root:
    from bhml.sim import trace_photon
    traj, outcome = trace_photon(6.0, r0=30.0)
    x, y = traj["x"], traj["y"]
"""

from __future__ import annotations

try:
    from . import _bhsim  # compiled C++ extension, placed in bhml/ by CMake
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The _bhsim extension is not built. Configure with -DBHSIM_PYTHON=ON and "
        "build (see this module's docstring)."
    ) from exc


def _to_columns(result: dict) -> tuple[dict, str]:
    data = result["data"]
    cols = result["columns"]
    return {name: data[:, i] for i, name in enumerate(cols)}, result["outcome"]


def trace_photon(b: float, r0: float = 30.0, lambda_max: float = 1e9):
    """Trace a photon of impact parameter b from radius r0.

    Returns (columns, outcome) where columns is a dict of numpy arrays keyed by
    lambda, r, phi, x, y, and outcome is 'captured' / 'escaped' / ...
    """
    return _to_columns(_bhsim.trace_photon(b, r0, lambda_max))


def trace_massive(E: float, L: float, r0: float, lambda_max: float = 1500.0):
    """Trace a massive particle with energy E and angular momentum L from r0."""
    return _to_columns(_bhsim.trace_massive(E, L, r0, lambda_max))
