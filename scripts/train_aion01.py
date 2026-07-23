"""train_aion01.py — Train AION-0.1.

Runs a complete GPT training run using TrainingProject with the AION-0.1
configuration.  All artifacts are saved to the aion-01 project.

Usage
-----
    python scripts/train_aion01.py           # fresh run
    python scripts/train_aion01.py --resume  # resume from latest checkpoint

Configuration
-------------
Architecture:  d_model=256, n_heads=8, n_layers=4, d_ff=1024
Parameters:    ~8M
Context:       256 tokens
Vocabulary:    8,192 (from trained BPE tokenizer)
Batch size:    8
Optimizer:     Adam, lr=3e-4
Schedule:      Cosine decay, 100 warmup steps, min_lr=3e-5
Epochs:        5
Seed:          42

Expected runtime: 3–7 hours on CPU.

Output
------
Prints live training progress (epoch, loss, val_loss, perplexity, tokens/sec).
All artifacts saved to workspace/projects/aion-01/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion.tokenizers.store import TokenizerStore
from aion.workspace.store import ProjectStore
from aion.training.config import TrainingConfig
from aion.training.project import TrainingProject

PROJECT_NAME = "aion-01"
DATASET_NAME = "aion-corpus"

# ── AION-0.1 architecture ─────────────────────────────────────────────────────
GPT_CONFIG = {
    "d_model":    256,
    "n_heads":    8,
    "n_layers":   4,
    "d_ff":       1024,
    "dropout":    0.0,
    "activation": "gelu",
    "pre_norm":   True,
    "tie_weights": True,
    # vocab_size and max_seq_len are set from tokenizer and context_length
}

# ── Sample prompts — fixed for all checkpoints ────────────────────────────────
SAMPLE_PROMPTS = [
    "The history of",
    "In the beginning",
    "Science is",
    "Once upon a time",
    "The most important",
]


def _progress(fraction: float, message: str) -> None:
    bar_len = 40
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    safe_msg = message.encode("ascii", errors="replace").decode("ascii")
    print(f"\r  [{bar}] {fraction*100:5.1f}%  {safe_msg:<60}", end="", flush=True)


def main() -> None:
    resume = "--resume" in sys.argv

    # ── Open project ──────────────────────────────────────────────────────────
    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    print(f"Project: {project.name} ({project.id})")

    # ── Resolve tokenizer ─────────────────────────────────────────────────────
    ts = TokenizerStore(project.dir("tokenizers"))
    tokenizers = ts.list()
    if not tokenizers:
        print("ERROR: No tokenizer found. Run scripts/train_tokenizer.py first.")
        sys.exit(1)

    # Use the default tokenizer, or the most recently created one
    default_tok = project.manifest.get("defaults", {}).get("tokenizer")
    if default_tok:
        tok_manifest = ts.open_manifest(default_tok)
    else:
        tok_manifest = tokenizers[0]
    tok_id = tok_manifest["id"]
    print(f"Tokenizer: {tok_manifest['name']} ({tok_id})")
    print(f"  Vocab size: {tok_manifest['vocab_size']}")

    # ── Resolve dataset ───────────────────────────────────────────────────────
    from aion.datasets.store import DatasetStore
    ds_store = DatasetStore(project.data_dir())
    ds = ds_store.open(DATASET_NAME)
    print(f"Dataset: {ds.id} — {ds.meta['document_count']} documents")

    # ── Build TrainingConfig ──────────────────────────────────────────────────
    config = TrainingConfig(
        model_name="AION-0.1",
        gpt_config=GPT_CONFIG,
        dataset_ids=[ds.id],
        tokenizer_id=tok_id,
        context_length=256,
        batch_size=8,
        train_split=0.9,
        optimizer="adam",
        learning_rate=3e-4,
        grad_clip=1.0,
        epochs=5,
        scheduler="cosine",
        warmup_steps=100,
        min_lr=3e-5,
        checkpoint_every_n_epochs=1,
        keep_last_n_checkpoints=3,
        eval_every_n_epochs=1,
        eval_batches=None,
        sample_every_n_epochs=1,
        sample_prompts=SAMPLE_PROMPTS,
        sample_max_new_tokens=64,
        seed=42,
    )

    print(f"\nAION-0.1 Training Configuration")
    print(f"─────────────────────────────────────────")
    print(f"  Architecture:  d_model={GPT_CONFIG['d_model']}, n_heads={GPT_CONFIG['n_heads']}, "
          f"n_layers={GPT_CONFIG['n_layers']}, d_ff={GPT_CONFIG['d_ff']}")
    print(f"  Context:       {config.context_length} tokens")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Optimizer:     {config.optimizer}, lr={config.learning_rate}")
    print(f"  Schedule:      {config.scheduler}, warmup={config.warmup_steps}, "
          f"min_lr={config.min_lr}")
    print(f"  Epochs:        {config.epochs}")
    print(f"  Seed:          {config.seed}")
    if resume:
        print(f"  Mode:          RESUME from latest checkpoint")
    print(f"─────────────────────────────────────────\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    tp = TrainingProject(project, config)
    t0 = time.monotonic()

    print("Starting training...\n")
    run_result = tp.run(resume=resume, progress_fn=_progress)
    print()  # newline after final progress bar

    elapsed = time.monotonic() - t0

    # ── Results ───────────────────────────────────────────────────────────────
    m = run_result.metrics
    loss_history = m.get("loss_history", [])
    val_loss_history = m.get("val_loss_history", [])
    val_ppl_history = m.get("val_perplexity_history", [])
    tps_history = m.get("tokens_per_sec", [])

    print(f"\nTraining complete in {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print(f"─────────────────────────────────────────")
    print(f"  Run id:          {run_result.run_id}")
    print(f"  Model id:        {run_result.model_id}")
    print(f"  Corpus fp:       {run_result.corpus_fingerprint}")
    print(f"  Corpus docs:     {run_result.corpus_stats.get('n_documents')}")
    print(f"  Train tokens:    {run_result.corpus_stats.get('n_train_tokens'):,}")
    print(f"  Val tokens:      {run_result.corpus_stats.get('n_val_tokens'):,}")
    print(f"  Tokens processed:{m.get('tokens_processed', 0):,}")
    print(f"─────────────────────────────────────────")
    print(f"  Epoch  Train Loss  Val Loss   Perplexity  Tok/s")
    for i, tl in enumerate(loss_history):
        vl = val_loss_history[i] if i < len(val_loss_history) else None
        vp = val_ppl_history[i] if i < len(val_ppl_history) else None
        tps = tps_history[i] if i < len(tps_history) else None
        vl_str = f"{vl:.4f}" if vl is not None else "  —   "
        vp_str = f"{vp:.2f}" if vp is not None else "  —  "
        tps_str = f"{tps:.1f}" if tps is not None else "  —  "
        print(f"  {i+1:5d}  {tl:.4f}      {vl_str}     {vp_str}    {tps_str}")
    print(f"─────────────────────────────────────────")
    print(f"  Final train loss:  {m.get('final_loss', 0):.4f}")
    if m.get("final_val_loss") is not None:
        print(f"  Final val loss:    {m['final_val_loss']:.4f}")
        print(f"  Final perplexity:  {m['final_val_perplexity']:.2f}")
    print(f"  Checkpoints:       {run_result.checkpoint_dir}")
    print(f"  Log:               {run_result.log_path}")
    print(f"  Model card:        workspace/projects/{PROJECT_NAME}/models/{run_result.model_id}/model_card.md")
    print(f"─────────────────────────────────────────")
    print(f"\nNext step: python scripts/evaluate_aion01.py {run_result.model_id}")


if __name__ == "__main__":
    main()
