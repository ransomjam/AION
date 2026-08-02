"""Array backend — NumPy on CPU, CuPy on CUDA.

AION's autograd engine is array-library agnostic by construction: ``Tensor``
wraps an array and every operation in :mod:`aion.nn.ops` is expressed as
arithmetic on that array.  Replacing the array library therefore replaces the
compute device without touching the graph machinery, exactly as
:mod:`aion.nn.tensor` documents.

This module is that replacement point.

Selecting a device
------------------
The device is chosen once, at import time, from the ``AION_DEVICE``
environment variable::

    AION_DEVICE=cpu     NumPy on the host           (default)
    AION_DEVICE=cuda    CuPy on an NVIDIA GPU

It is deliberately an environment variable and not a runtime argument: the
choice must be made before any ``Tensor`` exists, because a single process
cannot hold half a graph on each device.  A training script therefore never
decides the device — the operator does, when launching it.

What stays on the host, always
------------------------------
Three things never move to the device, and that is a design decision, not an
oversight:

- **Random number generation.**  All RNG stays on ``numpy.random.Generator``.
  CuPy's generators produce a different stream for the same seed, so keeping
  RNG on the host is what makes a seed mean the same thing on both devices:
  the same weight initialisation, the same batch order, the same dropout
  masks.  Reproducibility is an operating invariant; it outranks the tiny cost
  of generating masks on the host.
- **Token ids and packed corpora.**  Training data is integer arrays streamed
  from disk.  They are indexed and sliced, never multiplied, so the host is
  where they belong.  Only the per-batch slice crosses to the device.
- **Serialization.**  ``.npz`` checkpoints are host arrays.  Everything written
  to disk goes through :func:`to_host` first, so a checkpoint written on a GPU
  pod loads on a CPU laptop and the reverse.  Checkpoints are device-neutral.

Numerical equivalence
---------------------
CPU and GPU runs are *not* bit-identical to each other — cuBLAS and OpenBLAS
reduce in different orders.  Bit-identity is guaranteed only for a given
device: the same code, seed, dataset version, and ``AION_DEVICE`` reproduce
the same result, and a resumed run continues bit-identically to an
uninterrupted one.  ``scripts/gpu_smoke.py`` measures the CPU/GPU agreement
directly rather than assuming it.
"""

from __future__ import annotations

import os

import numpy as _np

__all__ = [
    "DEVICE", "xp", "numpy", "is_gpu", "asarray", "as_index", "to_host",
    "scatter_add", "synchronize", "free_pool", "device_info",
]


def _requested_device() -> str:
    raw = os.environ.get("AION_DEVICE", "cpu").strip().lower()
    if raw in ("", "cpu", "host", "numpy"):
        return "cpu"
    if raw in ("cuda", "gpu", "cupy"):
        return "cuda"
    raise ValueError(
        f"AION_DEVICE={raw!r} is not a device.  Use 'cpu' or 'cuda'."
    )


_REQUESTED = _requested_device()

if _REQUESTED == "cuda":
    try:
        import cupy as _cp  # type: ignore[import-not-found]
        import cupyx as _cpx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise ImportError(
            "AION_DEVICE=cuda requires CuPy.  Install the build matching the "
            "pod's CUDA runtime, e.g. `pip install cupy-cuda12x`.  "
            "Unset AION_DEVICE to run on the CPU."
        ) from exc
    # Fail here rather than thousands of steps later: importing CuPy succeeds
    # on a machine with no GPU, but the first kernel launch does not.
    _cp.zeros(1).sum()
    DEVICE = "cuda"
    xp = _cp
else:
    _cp = None
    _cpx = None
    DEVICE = "cpu"
    xp = _np

#: The host array library.  Always NumPy, on every device.  Use this for RNG,
#: token ids, index construction, and anything headed for disk.
numpy = _np


def is_gpu() -> bool:
    """True when tensors live on a CUDA device."""
    return _cp is not None


def asarray(data, dtype=None):
    """Place ``data`` on the compute device, converting dtype if given.

    Accepts host arrays, device arrays, and Python sequences alike.  This is
    the single crossing point from host memory into the graph.
    """
    return xp.asarray(data, dtype=dtype)


def as_index(ids):
    """Return ``ids`` as an integer index array on the compute device.

    Fancy-indexing a device array requires the index array to be on the same
    device — NumPy indices raise rather than being copied implicitly, which is
    the behaviour we want: every host-to-device copy is written down.
    """
    return xp.asarray(ids, dtype=_np.intp)


def to_host(a):
    """Return a NumPy view or copy of ``a``, which may live on the device.

    Every path to disk and every value crossing into Python scalars goes
    through here.  On the CPU it is a no-op ``asarray``.
    """
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return _np.asarray(a)


def scatter_add(target, indices, values) -> None:
    """In-place ``target[indices] += values`` with duplicate indices summed.

    ``numpy.add.at`` is a ufunc method rather than an array function, so it
    does not dispatch to CuPy; CuPy spells the same operation
    ``cupyx.scatter_add``.  Embedding gradients depend on the duplicate-index
    accumulation, so plain ``target[indices] += values`` is not a substitute.
    """
    if _cpx is not None:
        _cpx.scatter_add(target, indices, values)
    else:
        _np.add.at(target, indices, values)


def synchronize() -> None:
    """Block until queued device work has finished.  No-op on the CPU.

    Only timing code needs this: CUDA kernel launches are asynchronous, so a
    tokens/sec figure measured without synchronising times the launch, not the
    work.
    """
    if _cp is not None:
        _cp.cuda.runtime.deviceSynchronize()


def free_pool() -> None:
    """Release CuPy's cached device memory back to the driver.  No-op on CPU.

    CuPy pools freed blocks rather than returning them, which is what makes
    per-step allocation cheap.  Call this only between phases (after training,
    before evaluation), never inside the training loop.
    """
    if _cp is not None:
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()


def device_info() -> dict:
    """A JSON-serializable description of the active compute device.

    Recorded alongside run metrics so a result can be attributed to the
    hardware that produced it.
    """
    info: dict = {"device": DEVICE, "array_library": xp.__name__,
                  "array_library_version": xp.__version__}
    if _cp is None:
        return info
    props = _cp.cuda.runtime.getDeviceProperties(_cp.cuda.runtime.getDevice())
    name = props["name"]
    info["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
    free_bytes, total_bytes = _cp.cuda.runtime.memGetInfo()
    info["gpu_memory_total_gb"] = round(total_bytes / 1024**3, 2)
    info["gpu_memory_free_gb"] = round(free_bytes / 1024**3, 2)
    info["cuda_runtime_version"] = _cp.cuda.runtime.runtimeGetVersion()
    return info
