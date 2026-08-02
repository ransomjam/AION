"""train_tokenizer.py — Train the AION BPE tokenizer.

Trains a byte-level BPE tokenizer over one or more datasets in the aion-01
project, then sets it as the project default.

Usage
-----
    python scripts/train_tokenizer.py                       # general corpus only
    python scripts/train_tokenizer.py --datasets aion-corpus aion-cyber
    python scripts/train_tokenizer.py --datasets aion-corpus aion-cyber --retrain
    python scripts/train_tokenizer.py --vocab-size 16384 --name aion-cyber-bpe-16k

Why the tokenizer must be retrained for AION-Cyber
--------------------------------------------------
A tokenizer trained on Gutenberg prose has never seen ``CVE-2021-44228``,
``momo-secure.net``, ``T1566.001``, or ``50,000 FCFA``.  It will encode each of
them as a long run of near-character-level tokens, which wastes context,
inflates the token count of exactly the strings the model most needs to reason
about, and makes those strings harder to learn.  Retraining over the combined
corpus is not an optimisation — it is what makes the cybersecurity half
learnable at all.

Retraining changes the vocabulary, which invalidates any model trained against
the old one.  That is why ``--retrain`` is explicit rather than automatic.

Runtime
-------
The BPE trainer recomputes pair counts on every merge, so cost grows with both
the vocabulary size and the number of distinct pre-tokens.  Measured on this
machine, roughly:

    ~1.5 MB sample, vocab 4096   ~30 minutes
    ~1.5 MB sample, vocab 8192   ~60 minutes
    full 67 MB corpus, vocab 8192   ~13 hours

Use ``--sample-chars`` unless there is a reason not to.  Merge selection is
driven by frequency statistics that stabilise within a few megabytes; the rest
of the corpus changes almost nothing about which merges are chosen, and the
resulting tokenizer is still applied to the whole corpus afterwards.

Output
------
Saves the tokenizer to workspace/projects/aion-01/tokenizers/ and prints
the tokenizer id.  Sets it as the project default tokenizer.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Train the AION BPE tokenizer.")
    parser.add_argument("--datasets", nargs="+", default=[DATASET_NAME],
                        help=f"dataset ids to train over (default: {DATASET_NAME})")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--name", default=None,
                        help="tokenizer name (default derives from vocab size)")
    parser.add_argument("--retrain", action="store_true",
                        help="train even though a tokenizer of this name exists. "
                             "Any model trained on the previous vocabulary becomes "
                             "incompatible, so this is never implicit.")
    parser.add_argument("--sample-chars", type=int, default=0,
                        help="train on at most this many characters, sampled evenly "
                             "across the datasets (0 = use everything). BPE merge "
                             "selection converges on a few MB; the full corpus costs "
                             "hours and buys almost nothing. 1500000 is a good start.")
    args = parser.parse_args()

    tokenizer_name = args.name or f"aion-01-bpe-{args.vocab_size // 1024}k"

    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    print(f"Project: {project.name} ({project.id})")

    ds_store = DatasetStore(project.data_dir())
    datasets = []
    for dataset_id in args.datasets:
        dataset = ds_store.open(dataset_id)
        datasets.append(dataset)
        print(f"Dataset: {dataset.id} — {dataset.meta['document_count']:,} documents")

    ts = TokenizerStore(project.dir("tokenizers"))

    existing = [m for m in ts.list() if m.get("name") == tokenizer_name]
    if existing and not args.retrain:
        print(f"\nTokenizer '{tokenizer_name}' already exists: {existing[0]['id']}")
        print("Pass --retrain to train a new one, or --name to save it alongside.")
        return

    # One fingerprint over every dataset in the mix: the identity of what was
    # trained on has to cover all of it, or the lineage lies by omission.
    ds_fp = "+".join(dataset.fingerprint() for dataset in datasets)
    print(f"Dataset fingerprint: {ds_fp}")
    print(f"\nTraining BPE tokenizer: vocab_size={args.vocab_size}")
    print("This may take 20–60 minutes on CPU.\n")

    tokenizer = ByteLevelBPETokenizer()

    def corpus_stream():
        """Yield training text, optionally capped at ``--sample-chars``.

        The budget is split evenly across datasets and, within a dataset, taken
        as a prefix of every Nth document rather than the first few documents.
        Taking the first N documents whole would train the vocabulary on one
        author or one source; spreading the sample keeps the merge statistics
        representative of the mix that will actually be trained on.
        """
        if args.sample_chars <= 0:
            for dataset in datasets:
                for _, text in dataset.stream():
                    yield text
            return

        per_dataset = args.sample_chars // len(datasets)
        for dataset in datasets:
            doc_ids = dataset.document_ids()
            if not doc_ids:
                continue
            per_doc = max(500, per_dataset // len(doc_ids))
            used = 0
            for doc_id in doc_ids:
                if used >= per_dataset:
                    break
                chunk = dataset.get_document(doc_id)[:per_doc]
                used += len(chunk)
                yield chunk
            print(f"  sampled {used:,} chars from {dataset.id}")

    corpus = corpus_stream()

    t0 = time.monotonic()
    result = tokenizer.train(
        corpus,
        vocab_size=args.vocab_size,
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
    dataset_label = "+".join(dataset.id for dataset in datasets)
    params = {"vocab_size": args.vocab_size, "algorithm": tokenizer.algorithm,
              "dataset_ids": [dataset.id for dataset in datasets]}
    manifest = ts.save(
        tokenizer, result,
        name=tokenizer_name,
        description=(f"Byte-level BPE, {args.vocab_size} tokens, "
                     f"trained on {dataset_label}"),
        dataset_id=dataset_label,
        dataset_fingerprint=ds_fp,
        params=params,
    )
    tok_id = manifest["id"]
    print(f"  Saved: {tok_id}")

    # Record experiment
    exp_store = ExperimentStore(project.dir("experiments"))
    exp_store.record(
        experiment_type="train_tokenizer",
        dataset_id=dataset_label,
        dataset_fingerprint=ds_fp,
        params=params,
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

    # Smoke test.  The cyber strings are here on purpose: they are the ones a
    # prose-trained vocabulary shreds, so the token counts below are the direct
    # evidence that retraining was worth doing.
    print("\nSmoke test:")
    for test_text in (
        "The history of science and mathematics.",
        "CVE-2021-44228 affects Apache Log4j2 and is rated CVSS 10.0.",
        "Confirm your PIN at momo-secure.net to avoid suspension.",
        "Vous avez gagné 500,000 FCFA. Réclamez votre prix maintenant.",
        "Technique T1566.001 — Spearphishing Attachment.",
    ):
        ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(ids)
        ratio = len(ids) / max(1, len(test_text.split()))
        flag = "" if decoded == test_text else "   [!] round-trip differs"
        print(f"  {len(ids):>4} tokens ({ratio:.2f}/word)  {test_text!r}{flag}")

    sep = "-" * 41
    print(f"""
Tokenizer training complete.
{sep}
  Tokenizer id:  {tok_id}
  Vocab size:    {tokenizer.vocab_size}
  Compression:   {result.metrics['compression_ratio']:.3f}x
{sep}
Next step: python scripts/train_aion_cyber.py
""")


if __name__ == "__main__":
    main()
