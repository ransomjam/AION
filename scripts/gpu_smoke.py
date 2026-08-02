"""gpu_smoke.py — prove the GPU path is correct before paying for hours of it.

Run this on the pod, immediately after installing CuPy and before starting any
real training.  It takes under a minute and answers the only two questions that
matter at that moment:

1. **Does the engine actually run on the device?**  Importing CuPy succeeds on
   machines with no usable GPU; launching a kernel does not.
2. **Does it compute the same answers as the CPU?**  A backend that runs fast
   and wrong is worse than no backend, and the failure would only surface hours
   later as a loss curve that goes nowhere.

Usage
-----
    python scripts/gpu_smoke.py                 # check the current device
    python scripts/gpu_smoke.py --compare       # run both devices and diff them
    python scripts/gpu_smoke.py --benchmark     # measure steps/sec too

``--compare`` re-executes this file as a subprocess with the opposite
``AION_DEVICE``, because the device is fixed at import time and a single
process cannot hold both.

What "agreement" means here
---------------------------
CPU and GPU results are not expected to be bit-identical: cuBLAS and OpenBLAS
sum in different orders, and float32 addition is not associative.  What must
hold is that the difference stays at rounding scale.  A loss agreeing to ~1e-4
after twenty optimiser steps is a correct implementation; a loss that diverges
is a bug, and this script names which one it saw.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from aion import backend
from aion.gpt.config import GPTConfig
from aion.gpt.loss import CausalLanguageModelLoss
from aion.gpt.model import GPTModel
from aion.nn.optim import Adam

# Small enough to run in seconds, large enough to exercise every op the real
# model uses: embedding lookup, attention, layer norm, GELU, tied output head.
SMOKE_CONFIG = dict(vocab_size=512, d_model=64, n_heads=4, n_layers=2,
                    d_ff=256, max_seq_len=32, dropout=0.0)
STEPS = 60
BATCH, SEQ = 4, 32


def run_training_steps(steps: int = STEPS) -> dict:
    """Overfit one fixed batch and report the loss at every step.

    Repeating a single batch is the sharpest cheap test of a training stack.
    The model has enough capacity to memorise four short sequences, so a
    correct forward, backward, and optimiser step drive the loss from about
    ln(vocab) towards zero within tens of steps.  Anything broken — a gradient
    that does not reach a parameter, a scatter that overwrites instead of
    accumulating, an optimiser updating a stale buffer — shows up immediately
    as a loss that sits flat, because random data cannot mask it.

    Everything is seeded identically on every device, and the batch is drawn on
    the host, so a CPU/GPU difference can only come from the arithmetic.
    """
    cfg = GPTConfig(**SMOKE_CONFIG)
    model = GPTModel(cfg, rng=np.random.default_rng(1234))
    optimizer = Adam(model.parameters(), lr=3e-3)
    loss_fn = CausalLanguageModelLoss()

    tokens = np.random.default_rng(99).integers(
        0, cfg.vocab_size, size=(BATCH, SEQ + 1), dtype=np.int32)
    x, y = tokens[:, :-1], tokens[:, 1:]

    losses = []
    started = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    backend.synchronize()
    elapsed = time.monotonic() - started

    return {
        "device": backend.DEVICE,
        "losses": losses,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "steps": steps,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed else 0.0,
        "param_count": model.param_count(),
    }


def benchmark(batch: int, seq: int, steps: int = 30) -> dict:
    """Time the realistic shape, to size the batch before committing to a run."""
    cfg = GPTConfig(vocab_size=8192, d_model=512, n_heads=8, n_layers=6,
                    d_ff=2048, max_seq_len=seq, dropout=0.0)
    model = GPTModel(cfg, rng=np.random.default_rng(1))
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = CausalLanguageModelLoss()
    rng = np.random.default_rng(2)

    # One untimed step: the first allocates pools and compiles kernels, and
    # counting it would understate steady-state throughput.
    for index in range(steps + 1):
        if index == 1:
            backend.synchronize()
            started = time.monotonic()
        tokens = rng.integers(0, cfg.vocab_size, size=(batch, seq + 1), dtype=np.int32)
        optimizer.zero_grad()
        logits, _ = model(tokens[:, :-1])
        loss = loss_fn(logits, tokens[:, 1:])
        loss.backward()
        optimizer.step()
    backend.synchronize()
    elapsed = time.monotonic() - started

    tokens_per_step = batch * seq
    return {
        "device": backend.DEVICE,
        "param_count": model.param_count(),
        "batch": batch,
        "seq": seq,
        "steps": steps,
        "seconds": round(elapsed, 2),
        "steps_per_second": round(steps / elapsed, 3),
        "tokens_per_second": round(steps * tokens_per_step / elapsed, 1),
    }


def _spawn(device: str, args: list[str]) -> dict | None:
    """Re-run this script under a different device and read back its JSON."""
    env = {**os.environ, "AION_DEVICE": device}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--json", *args],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        print(f"  {device}: FAILED\n{proc.stderr.strip()[:1500]}")
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  {device}: unreadable output\n{proc.stdout[-800:]}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the compute backend.")
    parser.add_argument("--compare", action="store_true",
                        help="run on CPU and CUDA and compare the results")
    parser.add_argument("--benchmark", action="store_true",
                        help="also measure throughput at the training batch shape")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable results (used by --compare)")
    args = parser.parse_args()

    if args.json:
        result = (benchmark(args.batch, args.seq) if args.benchmark
                  else run_training_steps())
        print(json.dumps(result))
        return

    if not args.compare:
        info = backend.device_info()
        print("Compute device")
        for key, value in info.items():
            print(f"  {key:<24} {value}")

        print(f"\nOverfitting one batch for {STEPS} steps...")
        result = run_training_steps()
        print(f"  parameters:      {result['param_count']:,}")
        print(f"  first loss:      {result['first_loss']:.6f}")
        print(f"  final loss:      {result['final_loss']:.6f}")
        print(f"  steps/second:    {result['steps_per_second']:.1f}")

        if not np.isfinite(result["final_loss"]):
            print("\nFAIL: the loss is not finite. The backend is producing NaN or inf.")
            sys.exit(1)
        # A model with this much capacity must memorise four sequences easily.
        # Halving the loss is a low bar that only a broken gradient path misses.
        if result["final_loss"] > result["first_loss"] * 0.5:
            print(f"\nFAIL: the loss only fell from {result['first_loss']:.3f} to "
                  f"{result['final_loss']:.3f}.\n"
                  f"      A correct backend memorises a single batch far faster "
                  f"than this. Do not train.")
            sys.exit(1)
        print("\nOK: the model memorised a single batch. Forward, backward, and "
              "optimiser all work on this device.")

        if args.benchmark:
            print(f"\nThroughput at batch={args.batch}, seq={args.seq}...")
            bench = benchmark(args.batch, args.seq)
            print(f"  parameters:      {bench['param_count']:,}")
            print(f"  steps/second:    {bench['steps_per_second']}")
            print(f"  tokens/second:   {bench['tokens_per_second']:,.0f}")
            print(f"\n  At this rate, 100M training tokens takes "
                  f"{100_000_000 / max(1, bench['tokens_per_second']) / 3600:.1f} hours.")
        return

    # ── comparison ────────────────────────────────────────────────────────────
    extra = ["--benchmark", "--batch", str(args.batch), "--seq", str(args.seq)] \
        if args.benchmark else []
    print("Running the same computation on both devices...\n")
    cpu = _spawn("cpu", extra)
    gpu = _spawn("cuda", extra)

    if cpu is None or gpu is None:
        print("\nComparison could not complete.")
        sys.exit(1)

    line = "-" * 62
    if args.benchmark:
        speedup = gpu["tokens_per_second"] / max(1e-9, cpu["tokens_per_second"])
        print(f"{line}\n  Throughput at batch={args.batch}, seq={args.seq}\n{line}")
        print(f"  cpu    {cpu['tokens_per_second']:>12,.0f} tokens/sec")
        print(f"  cuda   {gpu['tokens_per_second']:>12,.0f} tokens/sec")
        print(f"  speedup {speedup:>11.1f}x")
        print(line)
        return

    cpu_losses = np.array(cpu["losses"])
    gpu_losses = np.array(gpu["losses"])
    max_diff = float(np.max(np.abs(cpu_losses - gpu_losses)))
    final_diff = abs(cpu["final_loss"] - gpu["final_loss"])

    print(f"{line}\n  CPU / CUDA agreement over {STEPS} optimiser steps\n{line}")
    print(f"  cpu  final loss   {cpu['final_loss']:.6f}   "
          f"({cpu['steps_per_second']:.1f} steps/s)")
    print(f"  cuda final loss   {gpu['final_loss']:.6f}   "
          f"({gpu['steps_per_second']:.1f} steps/s)")
    print(f"  largest per-step difference   {max_diff:.2e}")
    print(f"  final difference              {final_diff:.2e}")
    print(line)

    # Tolerance for float32 accumulated over 20 steps of Adam.  Loose enough
    # that reduction order does not trip it, tight enough that a genuinely
    # wrong gradient cannot pass.
    if max_diff > 1e-3:
        print("\nFAIL: the devices disagree beyond rounding error.")
        print("      Do not start a training run. Something in the GPU path is wrong.")
        sys.exit(1)
    print("\nOK: the devices agree to within float32 rounding. Safe to train.")


if __name__ == "__main__":
    main()
