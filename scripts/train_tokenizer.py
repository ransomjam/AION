"""train_tokenizer.py — Train the AION-0.1 BPE tokenizer.

Trains an 8,192-token byte-level BPE tokenizer on the aion-corpus dataset
in the aion-01 project, then sets it as the project default.

Usage
-----
    python scripts/train_tokenizer.py

Expected runtime: 20–60 minutes on CPU depending on corpus size.

Output
------
Saves the tokenizer to workspace/projects/aion-01/tokenizers/ and prints
the tokenizer id.  Sets it as the project default tokenizer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion.datasets.store import DatasetStore
from aion.tokenizers.bpe import ByteLevelBPETokenizer
from aion.tokenizers.store import TokenizerStore
from aion.workspace.experiments import ExperimentStore
from aion.workspace.store import ProjectStore

PROJECT_NAME = "aion-01"
DATASET_NAME = "aion-corpus"
TOKENIZER_NAME = "aion-01-bpe-8k"
VOCAB_SIZE = 8_192


def _progress(fraction: float, message: str) -> None:
    bar_len = 30
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    safe_msg = message.encode("ascii", errors="replace").decode("ascii")
    print(f"\r  [{bar}] {fraction*100:5.1f}%  {safe_msg:<50}", end="", flush=True)


def main() -> None:
    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    print(f"Project: {project.name} ({project.id})")

    ds_store = DatasetStore(project.data_dir())
    ds = ds_store.open(DATASET_NAME)
    print(f"Dataset: {ds.id} — {ds.meta['document_count']} documents")

    ts = TokenizerStore(project.dir("tokenizers"))

    # Check if already trained
    existing = [m for m in ts.list() if m.get("name") == TOKENIZER_NAME]
    if existing:
        print(f"\nTokenizer '{TOKENIZER_NAME}' already exists: {existing[0]['id']}")
        print("Delete it first if you want to retrain.")
        return

    ds_fp = ds.fingerprint()
    print(f"Dataset fingerprint: {ds_fp}")
    print(f"\nTraining BPE tokenizer: vocab_size={VOCAB_SIZE}")
    print("This may take 20–60 minutes on CPU.\n")

    tokenizer = ByteLevelBPETokenizer()
    corpus = (text for _, text in ds.stream())

    t0 = time.monotonic()
    result = tokenizer.train(
        corpus,
        vocab_size=VOCAB_SIZE,
        progress_fn=_progress,
        dataset_fingerprint=ds_fp,
    )
    elapsed = time.monotonic() - t0
    print()  # newline after progress bar

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Final vocab size:    {tokenizer.vocab_size}")
    print(f"  Merges:              {result.metrics['merge_count']}")
    print(f"  Compression ratio:   {result.metrics['compression_ratio']:.3f}x")
    print(f"  Avg tokens/word:     {result.metrics['avg_tokens_per_word']:.3f}")

    # Save
    print("\nSaving tokenizer...")
    manifest = ts.save(
        tokenizer, result,
        name=TOKENIZER_NAME,
        description=f"Byte-level BPE, {VOCAB_SIZE} tokens, trained on aion-corpus",
        dataset_id=ds.id,
        dataset_fingerprint=ds_fp,
        params={"vocab_size": VOCAB_SIZE, "algorithm": tokenizer.algorithm},
    )
    tok_id = manifest["id"]
    print(f"  Saved: {tok_id}")

    # Record experiment
    exp_store = ExperimentStore(project.dir("experiments"))
    exp_store.record(
        experiment_type="train_tokenizer",
        dataset_id=ds.id,
        dataset_fingerprint=ds_fp,
        params={"vocab_size": VOCAB_SIZE, "algorithm": tokenizer.algorithm},
        artifact_id=tok_id,
        metrics=result.metrics,
    )

    # Set as project default
    for m in ts.list():
        if m.get("status") == "default" and m["id"] != tok_id:
            ts.set_status(m["id"], "experimental")
    ts.set_status(tok_id, "default")
    store.update(PROJECT_NAME, defaults={
        **project.manifest.get("defaults", {}),
        "tokenizer": tok_id,
    })
    print(f"  Set as project default tokenizer.")

    # Quick smoke test
    test_text = "The history of science and mathematics."
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    print(f"\nSmoke test:")
    print(f"  Input:   {test_text!r}")
    print(f"  Tokens:  {len(ids)}  ids={ids[:10]}{'...' if len(ids) > 10 else ''}")
    print(f"  Decoded: {decoded!r}")

    sep = "-" * 41
    print(f"""
Tokenizer training complete.
{sep}
  Tokenizer id:  {tok_id}
  Vocab size:    {tokenizer.vocab_size}
  Compression:   {result.metrics['compression_ratio']:.3f}x
{sep}
Next step: python scripts/train_aion01.py
""")


if __name__ == "__main__":
    main()
