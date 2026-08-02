"""Step-based, crash-safe checkpointing for long training runs.

Motivation
----------
Epoch-boundary checkpoints are useless when a single CPU epoch takes hours: an
interruption before the first epoch completes loses the entire run.  This module
saves a *complete* checkpoint every N optimizer steps (and/or every N minutes),
so a run can always resume from close to where it stopped.

A checkpoint is a **self-contained bundle** — everything needed to resume
bit-identically:

    step_<N>/
        model.npz      all model parameters
        optim.npz      optimizer moment buffers (Adam m/v, SGD velocity)
        state.json     global_step, epoch, batch_in_epoch, optimizer scalars,
                       RNG states (model / data / data-epoch-start / sample),
                       scheduler descriptor, training history, run metadata,
                       and the full TrainingConfig snapshot

Directory layout under ``checkpoints/<run_id>/``::

    latest/            an exact copy of the newest checkpoint (for --resume)
    step_250/
    step_500/
    step_750/

Atomicity
---------
Nothing is ever written in place.  Each bundle is written to a temporary
directory, every file and the directory itself are ``fsync``-ed, and only then
is it atomically ``os.replace``-d into its final name.  A power failure can
therefore never leave a half-written ``step_<N>/`` or ``latest/``.  If ``latest/``
is ever missing or unreadable, :meth:`StepCheckpointManager.load_latest` falls
back to the highest-numbered intact ``step_<N>/``, so a crash mid-swap never
loses the run.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aion import backend

SCHEMA_VERSION = 1


def _param_key(i: int, p) -> str:
    return f"p{i}__{p.name or 'param'}"


# ── low-level durability helpers ──────────────────────────────────────────────

def _fsync_path(path: Path) -> None:
    """Best-effort fsync of a file or directory (directory fsync is a no-op on
    platforms that do not support it, e.g. Windows)."""
    try:
        if path.is_dir():
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            with open(path, "rb") as f:
                os.fsync(f.fileno())
    except (OSError, PermissionError):
        # Directory fsync is unsupported on some platforms; the atomic rename
        # still provides crash-consistency for the file contents.
        pass


def _fsync_tree(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_file():
            _fsync_path(child)
    _fsync_path(directory)


def _replace_dir(src: Path, dst: Path) -> None:
    """Atomically move the fully-written temp dir ``src`` onto ``dst``.

    ``os.replace`` is atomic when ``dst`` does not exist.  When ``dst`` already
    exists (Windows cannot atomically replace a non-empty directory), the old
    directory is renamed aside first; the only crash window leaves ``dst``
    momentarily absent, which the caller's fallback scan tolerates.
    """
    if dst.exists():
        old = dst.parent / f".{dst.name}.old.{uuid.uuid4().hex[:8]}"
        os.replace(dst, old)
        try:
            os.replace(src, dst)
        finally:
            shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(src, dst)
    _fsync_path(dst.parent)


# ── bundle read / write ───────────────────────────────────────────────────────

def _write_bundle(dst: Path, model, optimizer, state: dict) -> None:
    """Write a complete checkpoint bundle into fresh directory ``dst``."""
    dst.mkdir(parents=True, exist_ok=True)

    params = model.parameters()
    # Host copies, so a checkpoint is device-neutral: a run interrupted on a
    # GPU pod can be resumed on a CPU box and the reverse.
    weights = {_param_key(i, p): backend.to_host(p.data)
               for i, p in enumerate(params)}
    np.savez(str(dst / "model.npz"), **weights)

    state = dict(state)
    state["schema_version"] = SCHEMA_VERSION
    state["param_keys"] = [_param_key(i, p) for i, p in enumerate(params)]

    if optimizer is not None:
        moment_arrays = optimizer.moment_arrays()
        if moment_arrays:
            np.savez(str(dst / "optim.npz"), **moment_arrays)
        state["optimizer"] = optimizer.state_dict()

    (dst / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _fsync_tree(dst)


@dataclass
class LoadedCheckpoint:
    """A checkpoint read from disk, ready to restore into a model/optimizer."""
    state: dict
    model_weights: dict           # param_key -> ndarray
    optim_arrays: dict            # optimizer buffer key -> ndarray
    directory: str

    @property
    def global_step(self) -> int:
        return int(self.state.get("global_step", 0))

    @property
    def epoch(self) -> int:
        return int(self.state.get("epoch", 0))

    @property
    def batch_in_epoch(self) -> int:
        return int(self.state.get("batch_in_epoch", 0))


# ── manager ───────────────────────────────────────────────────────────────────

class StepCheckpointManager:
    """Create, list, load, and prune step-based checkpoints for one run.

    Parameters
    ----------
    checkpoints_root:
        The project's ``checkpoints/`` directory.
    run_id:
        The training run identifier; all checkpoints live under
        ``checkpoints_root / run_id /``.
    keep_last_n:
        Maximum number of historical ``step_<N>/`` directories to retain
        (``latest/`` is always kept and does not count).  ``<= 0`` keeps all.
    """

    def __init__(self, checkpoints_root: Path, run_id: str, keep_last_n: int = 5) -> None:
        self.run_dir = Path(checkpoints_root) / run_id
        self.run_id = run_id
        self.keep_last_n = keep_last_n

    # ── save ──────────────────────────────────────────────────────────────────

    def save(self, model, optimizer, state: dict) -> Path:
        """Atomically write a checkpoint for ``state['global_step']``.

        Writes ``step_<N>/`` first (the durable, historical record), then
        refreshes ``latest/``.  Returns the path to ``latest/``.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        step = int(state["global_step"])

        step_dir = self.run_dir / f"step_{step}"
        tmp_step = self.run_dir / f".tmp_step_{step}_{uuid.uuid4().hex[:8]}"
        _write_bundle(tmp_step, model, optimizer, state)
        _replace_dir(tmp_step, step_dir)

        latest = self.run_dir / "latest"
        tmp_latest = self.run_dir / f".tmp_latest_{uuid.uuid4().hex[:8]}"
        _write_bundle(tmp_latest, model, optimizer, state)
        _replace_dir(tmp_latest, latest)

        self.prune()
        return latest

    # ── load ──────────────────────────────────────────────────────────────────

    def has_checkpoint(self) -> bool:
        return self.load_latest() is not None

    def load_latest(self) -> LoadedCheckpoint | None:
        """Return the newest checkpoint, or ``None`` if there is none.

        Prefers ``latest/``; if it is missing or unreadable (e.g. a crash during
        the swap), falls back to the highest-numbered intact ``step_<N>/``.
        """
        loaded = self._read_bundle(self.run_dir / "latest")
        if loaded is not None:
            return loaded
        for step_dir in self._step_dirs(descending=True):
            loaded = self._read_bundle(step_dir)
            if loaded is not None:
                return loaded
        return None

    def _read_bundle(self, directory: Path) -> LoadedCheckpoint | None:
        try:
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            with np.load(str(directory / "model.npz")) as wz:
                weights = {k: wz[k] for k in wz.files}
            optim_path = directory / "optim.npz"
            if optim_path.is_file():
                with np.load(str(optim_path)) as oz:
                    optim_arrays = {k: oz[k] for k in oz.files}
            else:
                optim_arrays = {}
            return LoadedCheckpoint(
                state=state, model_weights=weights,
                optim_arrays=optim_arrays, directory=str(directory),
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    # ── restore ───────────────────────────────────────────────────────────────

    @staticmethod
    def restore_model(model, loaded: LoadedCheckpoint) -> None:
        param_keys = loaded.state.get("param_keys", [])
        for i, p in enumerate(model.parameters()):
            key = param_keys[i] if i < len(param_keys) else _param_key(i, p)
            if key in loaded.model_weights:
                p.data = backend.asarray(
                    loaded.model_weights[key], dtype=p.data.dtype)

    @staticmethod
    def restore_optimizer(optimizer, loaded: LoadedCheckpoint) -> None:
        scalars = loaded.state.get("optimizer")
        if optimizer is not None and scalars is not None:
            optimizer.load_state_dict(scalars, loaded.optim_arrays)

    # ── housekeeping ──────────────────────────────────────────────────────────

    def _step_dirs(self, descending: bool = False) -> list[Path]:
        dirs = []
        if self.run_dir.is_dir():
            for d in self.run_dir.glob("step_*"):
                if d.is_dir():
                    try:
                        int(d.name.split("_")[1])
                        dirs.append(d)
                    except (IndexError, ValueError):
                        pass
        dirs.sort(key=lambda p: int(p.name.split("_")[1]), reverse=descending)
        return dirs

    def prune(self) -> None:
        """Delete the oldest ``step_<N>/`` beyond ``keep_last_n`` (keeps latest)."""
        if self.keep_last_n <= 0:
            return
        step_dirs = self._step_dirs()  # ascending
        excess = len(step_dirs) - self.keep_last_n
        for d in step_dirs[:max(0, excess)]:
            shutil.rmtree(d, ignore_errors=True)

    def list_steps(self) -> list[int]:
        return [int(d.name.split("_")[1]) for d in self._step_dirs()]
