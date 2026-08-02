"""train_aion_cyber.py — train AION-Cyber-0.1 on the mixed corpus.

Trains a GPT over ``aion-corpus`` (general language) plus ``aion-cyber``
(cybersecurity reference material and threat reasoning), on whichever device
``AION_DEVICE`` selects.

Usage
-----
    python scripts/train_aion_cyber.py                     # CPU, small batch
    AION_DEVICE=cuda python scripts/train_aion_cyber.py --preset gpu
    AION_DEVICE=cuda python scripts/train_aion_cyber.py --preset gpu --resume

Presets, not guesses
--------------------
The CPU and GPU presets differ in more than speed.  On a GPU the per-step
Python overhead of this autograd engine is amortised over a much larger batch,
so the *same wall-clock hour* buys a bigger model, a longer context, and more
tokens per step.  Running the CPU configuration on a rented GPU would leave
most of the card idle and most of the money unspent.

Both presets keep ``seed`` and ``train_split`` fixed so the two are comparable
as experiments rather than only as timings.

Checkpointing
-------------
Step-based checkpointing is on by default.  A rented pod can be reclaimed,
disconnected, or run out of credit at any moment, so nothing here assumes the
process survives to the end: a complete bundle is written every N steps and
every N minutes, and ``--resume`` continues from the exact interrupted step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from aion import backend
from aion.datasets.store import DatasetStore
from aion.tokenizers.store import TokenizerStore
from aion.training.config import TrainingConfig
from aion.training.project import TrainingProject
from aion.training.resume import ResumeError
from aion.workspace.store import ProjectStore

PROJECT_NAME = "aion-01"
MODEL_NAME = "AION-Cyber-0.1"
DEFAULT_DATASETS = ["aion-corpus", "aion-cyber"]

# ── presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    # Unchanged from AION-0.1 so a CPU run stays comparable to the existing one.
    "cpu": {
        "gpt": {"d_model": 256, "n_heads": 8, "n_layers": 4, "d_ff": 1024},
        "context_length": 256,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "epochs": 5,
        "checkpoint_every_n_steps": 250,
        "checkpoint_every_minutes": 15.0,
    },
    # ~14M parameters, batch 32, seq 512.  Sized against two hard limits, not
    # ambition:
    #
    #   Memory.  Measured activation cost is ~400 MB per batch element for this
    #   architecture, so batch 32 needs ~13 GB and leaves real headroom on a
    #   24 GB RTX 4090.  (An earlier version of this preset asked for batch 64
    #   at d_model=512/6 layers, which needs 30.1 GB and simply OOMs there.)
    #
    #   Data.  The corpus is ~16M tokens.  At 14M parameters, 10 epochs is
    #   ~11 tokens seen per parameter — the right order for a data-constrained
    #   run.  A larger model would not be better here, it would just memorise
    #   the corpus faster; the honest ceiling is set by the data, not the card.
    "gpu": {
        "gpt": {"d_model": 384, "n_heads": 6, "n_layers": 6, "d_ff": 1536},
        "context_length": 512,
        "batch_size": 32,
        "learning_rate": 6e-4,
        "epochs": 10,
        "checkpoint_every_n_steps": 500,
        "checkpoint_every_minutes": 10.0,
    },
}

# Prompts held fixed across every checkpoint and every preset, so samples from
# different runs can be read side by side.  Half general, half security: a model
# that has learned the reasoning format should continue the last three in it.
SAMPLE_PROMPTS = [
    "The history of",
    "Science is",
    "Vulnerability report: CVE-",
    "Message analysis\nChannel: SMS\nSender: MTN MoMo Alert\nMessage:\n",
    "Indicators observed:\n-",
]


def _progress(fraction: float, message: str) -> None:
    bar_len = 40
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    safe = message.encode("ascii", errors="replace").decode("ascii")
    print(f"\r  [{bar}] {fraction * 100:5.1f}%  {safe:<60}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AION-Cyber-0.1.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="cpu or gpu; defaults to matching AION_DEVICE")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="continue the most recent run of this model")
    args = parser.parse_args()

    preset_name = args.preset or ("gpu" if backend.is_gpu() else "cpu")
    preset = PRESETS[preset_name]

    info = backend.device_info()
    print("Compute device")
    for key, value in info.items():
        print(f"  {key:<24} {value}")
    if backend.is_gpu() and preset_name == "cpu":
        print("  NOTE: running the CPU preset on a GPU. Most of the card will idle.")
    if not backend.is_gpu() and preset_name == "gpu":
        print("  WARNING: the GPU preset on a CPU will be extremely slow.")
    print()

    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    print(f"Project: {project.name} ({project.id})")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    ts = TokenizerStore(project.dir("tokenizers"))
    tokenizers = ts.list()
    if not tokenizers:
        print("ERROR: no tokenizer in this project. Run scripts/train_tokenizer.py first.")
        sys.exit(1)
    default_id = project.manifest.get("defaults", {}).get("tokenizer")
    tok = ts.open_manifest(default_id) if default_id else tokenizers[0]
    print(f"Tokenizer: {tok['name']} ({tok['id']}), vocab {tok['vocab_size']:,}")

    trained_on = (tok.get("params") or {}).get("dataset_ids")
    if trained_on and sorted(trained_on) != sorted(args.datasets):
        print(f"  WARNING: this tokenizer was trained on {trained_on}, but training "
              f"is over {args.datasets}.\n"
              f"           Text from the extra dataset will encode poorly. "
              f"Retrain with:\n"
              f"           python scripts/train_tokenizer.py --datasets "
              f"{' '.join(args.datasets)} --retrain")

    # ── datasets ──────────────────────────────────────────────────────────────
    ds_store = DatasetStore(project.data_dir())
    dataset_ids: list[str] = []
    sizes: list[tuple[str, int]] = []
    for name in args.datasets:
        try:
            dataset = ds_store.open(name)
        except Exception:
            print(f"ERROR: dataset {name!r} not found. "
                  f"Run scripts/build_cyber_corpus.py first.")
            sys.exit(1)
        chars = sum(len(text) for _, text in dataset.stream())
        sizes.append((dataset.id, chars))
        dataset_ids.append(dataset.id)
        print(f"Dataset: {dataset.id} — {dataset.meta['document_count']:,} documents, "
              f"{chars:,} characters")
    total_chars = sum(chars for _, chars in sizes) or 1
    print("Domain balance (by characters):")
    for name, chars in sizes:
        print(f"  {name:<16} {chars / total_chars * 100:5.1f}%")

    # ── config ────────────────────────────────────────────────────────────────
    gpt_config = {
        **preset["gpt"],
        "dropout": 0.0,
        "activation": "gelu",
        "pre_norm": True,
        "tie_weights": True,
    }
    config = TrainingConfig(
        model_name=MODEL_NAME,
        gpt_config=gpt_config,
        dataset_ids=dataset_ids,
        tokenizer_id=tok["id"],
        context_length=args.context_length or preset["context_length"],
        batch_size=args.batch_size or preset["batch_size"],
        train_split=0.9,
        optimizer="adam",
        learning_rate=preset["learning_rate"],
        grad_clip=1.0,
        epochs=args.epochs or preset["epochs"],
        scheduler="cosine",
        warmup_steps=200,
        min_lr=preset["learning_rate"] / 10,
        checkpoint_every_n_epochs=1,
        keep_last_n_checkpoints=3,
        checkpoint_every_n_steps=preset["checkpoint_every_n_steps"],
        checkpoint_every_minutes=preset["checkpoint_every_minutes"],
        keep_last_n_step_checkpoints=5,
        log_every_n_steps=50,
        eval_every_n_epochs=1,
        eval_batches=None,
        sample_every_n_epochs=1,
        sample_prompts=SAMPLE_PROMPTS,
        sample_max_new_tokens=96,
        seed=args.seed,
    )
    config.validate()

    line = "-" * 62
    print(f"""
{MODEL_NAME} — preset '{preset_name}' on {info['device']}
{line}
  Architecture:  d_model={gpt_config['d_model']}, n_heads={gpt_config['n_heads']}, \
n_layers={gpt_config['n_layers']}, d_ff={gpt_config['d_ff']}
  Context:       {config.context_length} tokens
  Batch:         {config.batch_size}  ({config.batch_size * config.context_length:,} tokens/step)
  Optimizer:     {config.optimizer}, lr={config.learning_rate}, cosine to {config.min_lr}
  Epochs:        {config.epochs}
  Seed:          {config.seed}
  Checkpoints:   every {config.checkpoint_every_n_steps} steps \
or {config.checkpoint_every_minutes:.0f} minutes
  Mode:          {'RESUME' if args.resume else 'fresh run'}
{line}
""")

    tp = TrainingProject(project, config)
    started = time.monotonic()
    print("Starting training...\n")
    try:
        run_result = tp.run(resume=args.resume, progress_fn=_progress)
    except ResumeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Checkpoint saved.\n"
              "Resume with:\n\n"
              f"    python scripts/train_aion_cyber.py --preset {preset_name} --resume\n")
        sys.exit(130)
    print()

    elapsed = time.monotonic() - started
    backend.synchronize()

    metrics = run_result.metrics
    print(f"""
Training complete in {elapsed / 3600:.2f}h ({elapsed:.0f}s)
{line}
  Run id:            {run_result.run_id}
  Model id:          {run_result.model_id}
  Corpus fingerprint:{run_result.corpus_fingerprint}
  Train tokens:      {run_result.corpus_stats.get('n_train_tokens', 0):,}
  Val tokens:        {run_result.corpus_stats.get('n_val_tokens', 0):,}
  Tokens processed:  {metrics.get('tokens_processed', 0):,}
{line}
  Epoch  Train loss  Val loss  Perplexity  Tok/s""")
    losses = metrics.get("loss_history", [])
    val_losses = metrics.get("val_loss_history", [])
    perplexities = metrics.get("val_perplexity_history", [])
    throughput = metrics.get("tokens_per_sec", [])
    for i, train_loss in enumerate(losses):
        def at(seq, default="   -  "):
            return f"{seq[i]:.4f}" if i < len(seq) else default
        tps = f"{throughput[i]:.0f}" if i < len(throughput) else "  -  "
        ppl = f"{perplexities[i]:.2f}" if i < len(perplexities) else "  -  "
        print(f"  {i + 1:5d}  {train_loss:10.4f}  {at(val_losses):>8}  {ppl:>10}  {tps:>6}")
    print(f"""{line}
  Final train loss:  {metrics.get('final_loss', 0):.4f}""")
    if metrics.get("final_val_loss") is not None:
        print(f"  Final val loss:    {metrics['final_val_loss']:.4f}")
        print(f"  Final perplexity:  {metrics['final_val_perplexity']:.2f}")
    print(f"""  Checkpoints:       {run_result.checkpoint_dir}
  Model card:        workspace/projects/{PROJECT_NAME}/models/{run_result.model_id}/model_card.md
{line}""")

    # Record the hardware alongside the run.  A tokens/sec figure with no device
    # attached is not a measurement anyone can act on.
    device_path = Path(run_result.log_path).parent / "device.json"
    device_path.write_text(
        json.dumps({**info, "preset": preset_name,
                    "elapsed_seconds": round(elapsed, 1)}, indent=2),
        encoding="utf-8")

    print(f"\nNext: package the run for transfer\n"
          f"    python scripts/package_model.py {run_result.model_id}\n")


if __name__ == "__main__":
    main()
