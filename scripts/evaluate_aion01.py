"""evaluate_aion01.py — Post-training evaluation report for AION-0.1.

Loads a trained model, runs evaluation on the full validation set, generates
samples from a set of prompts, and writes a structured evaluation report.

Usage
-----
    python scripts/evaluate_aion01.py <model_id>
    python scripts/evaluate_aion01.py          # uses the most recent model

Output
------
Prints a full evaluation report to stdout.
Writes workspace/projects/aion-01/evaluations/<model_id>/report.json
Writes workspace/projects/aion-01/evaluations/<model_id>/report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion.gpt.store import GPTStore
from aion.tokenizers.store import TokenizerStore
from aion.workspace.store import ProjectStore
from aion.training.evaluation import EvaluationRunner
from aion.training.sampler import SampleGenerator
from aion.util import now_iso

PROJECT_NAME = "aion-01"
DATASET_NAME = "aion-corpus"

EVAL_PROMPTS = [
    "The history of",
    "In the beginning",
    "Science is",
    "Once upon a time",
    "The most important",
    "It was a dark and stormy",
    "The theory of",
    "Human beings are",
    "The greatest discovery",
    "Language is",
]

EVAL_MAX_NEW_TOKENS = 128


def main() -> None:
    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    print(f"Project: {project.name} ({project.id})")

    gpt_store = GPTStore(project.dir("models"))
    ts = TokenizerStore(project.dir("tokenizers"))

    # ── Resolve model ─────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        models = gpt_store.list()
        if not models:
            print("ERROR: No models found in project.")
            sys.exit(1)
        model_id = models[0]["id"]
        print(f"Using most recent model: {model_id}")

    manifest = gpt_store.open_manifest(model_id)
    print(f"Model: {manifest['name']} ({model_id})")
    print(f"  Architecture: {manifest['architecture']}")
    print(f"  Parameters:   {manifest['param_count']:,}")
    print(f"  Created:      {manifest['created_at']}")

    # ── Load model and tokenizer ──────────────────────────────────────────────
    print("\nLoading model...")
    model = gpt_store.load(model_id)

    tok_id = manifest.get("tokenizer_id", "")
    if not tok_id:
        print("ERROR: Model manifest has no tokenizer_id.")
        sys.exit(1)
    tokenizer = ts.load(tok_id)
    tok_manifest = ts.open_manifest(tok_id)
    print(f"Tokenizer: {tok_manifest['name']} (vocab={tokenizer.vocab_size})")

    # ── Rebuild validation set ────────────────────────────────────────────────
    print("\nRebuilding validation corpus...")
    from aion.datasets.store import DatasetStore
    from aion.training.corpus import CorpusManager
    from aion.gpt.data import BatchSampler, TokenizedDataset

    ds_store = DatasetStore(project.data_dir())
    ds = ds_store.open(DATASET_NAME)
    tok_fp = tok_manifest.get("vocabulary_fingerprint", "")
    eos_id = tokenizer.vocab_size - 1

    # Retrieve training config from model manifest to use same split/context
    training_config = manifest.get("params", {}).get("training_config", {})
    context_length = training_config.get("context_length", 256)
    train_split = training_config.get("train_split", 0.9)
    batch_size = training_config.get("batch_size", 8)

    corpus_mgr = CorpusManager(project.data_dir(), project.cache_dir())
    corpus = corpus_mgr.build(
        [ds.id],
        tokenizer,
        tokenizer_id=tok_id,
        tokenizer_fingerprint=tok_fp,
        eos_id=eos_id,
        split=train_split,
    )
    print(f"  Train tokens: {len(corpus.train_tokens):,}")
    print(f"  Val tokens:   {len(corpus.val_tokens):,}")

    val_ds = None
    if len(corpus.val_tokens) >= context_length + 1:
        val_ds = TokenizedDataset(corpus.val_tokens, context_length)
        val_sampler = BatchSampler(val_ds, batch_size, shuffle=False)
    else:
        print("  WARNING: Validation set too small for evaluation.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_result = None
    if val_ds is not None:
        print("\nRunning validation evaluation...")
        runner = EvaluationRunner(model)
        t0 = time.monotonic()
        eval_result = runner.evaluate(val_sampler)
        print(f"  Val loss:    {eval_result.val_loss:.4f}")
        print(f"  Perplexity:  {eval_result.perplexity:.2f}")
        print(f"  Batches:     {eval_result.n_batches}")
        print(f"  Elapsed:     {eval_result.elapsed_s:.1f}s")

    # ── Sample generation ─────────────────────────────────────────────────────
    print(f"\nGenerating {len(EVAL_PROMPTS)} samples (greedy, max {EVAL_MAX_NEW_TOKENS} tokens)...")
    gen = SampleGenerator(
        model, tokenizer,
        max_new_tokens=EVAL_MAX_NEW_TOKENS,
        strategy="greedy",
        eos_token_id=eos_id,
    )
    samples = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"  [{i+1}/{len(EVAL_PROMPTS)}] {prompt!r}...", end=" ", flush=True)
        result = gen.generate(prompt)
        samples.append({
            "prompt": prompt,
            "generated": result.generated_text,
            "full_text": result.full_text,
            "tokens": result.tokens,
            "elapsed_s": result.elapsed_s,
        })
        print(f"{result.tokens} tokens in {result.elapsed_s:.2f}s")

    # ── Training metrics from store ───────────────────────────────────────────
    stats = gpt_store.statistics(model_id) or {}

    # ── Assemble report ───────────────────────────────────────────────────────
    report = {
        "schema_version": 1,
        "evaluated_at": now_iso(),
        "model_id": model_id,
        "model_name": manifest["name"],
        "architecture": manifest.get("config", {}),
        "param_count": manifest["param_count"],
        "tokenizer_id": tok_id,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "corpus": {
            "dataset_id": ds.id,
            "n_documents": corpus.stats.n_documents,
            "n_train_tokens": corpus.stats.n_train_tokens,
            "n_val_tokens": corpus.stats.n_val_tokens,
            "corpus_fingerprint": corpus.fingerprint.combined,
        },
        "training_metrics": {
            "final_loss": stats.get("final_loss"),
            "loss_history": stats.get("loss_history", []),
            "val_loss_history": stats.get("val_loss_history", []),
            "val_perplexity_history": stats.get("val_perplexity_history", []),
            "tokens_processed": stats.get("tokens_processed"),
            "training_time_s": stats.get("training_time_s"),
            "epochs": stats.get("epochs"),
        },
        "evaluation": {
            "val_loss": eval_result.val_loss if eval_result else None,
            "perplexity": eval_result.perplexity if eval_result else None,
            "n_batches": eval_result.n_batches if eval_result else None,
            "eval_elapsed_s": eval_result.elapsed_s if eval_result else None,
        },
        "samples": samples,
    }

    # ── Write report ──────────────────────────────────────────────────────────
    eval_dir = project.root / "evaluations" / model_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    report_json = eval_dir / "report.json"
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_md = eval_dir / "report.md"
    report_md.write_text(_render_markdown(report), encoding="utf-8")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"AION-0.1 Evaluation Report")
    print(f"{'─'*60}")
    print(f"Model:          {manifest['name']} ({model_id})")
    print(f"Parameters:     {manifest['param_count']:,}")
    if eval_result:
        print(f"Val loss:       {eval_result.val_loss:.4f}")
        print(f"Perplexity:     {eval_result.perplexity:.2f}")
    loss_hist = stats.get("loss_history", [])
    if loss_hist:
        print(f"Train loss:     {loss_hist[0]:.4f} → {loss_hist[-1]:.4f} "
              f"(epoch 1 → {len(loss_hist)})")
    print(f"\nGenerated samples:")
    for s in samples:
        print(f"\n  Prompt:    {s['prompt']!r}")
        preview = s["full_text"][:200].replace("\n", " ")
        print(f"  Output:    {preview!r}{'...' if len(s['full_text']) > 200 else ''}")
    print(f"\n{'─'*60}")
    print(f"Report written to:")
    print(f"  {report_json}")
    print(f"  {report_md}")


def _render_markdown(report: dict) -> str:
    m = report
    ev = m["evaluation"]
    tm = m["training_metrics"]
    lines = [
        f"# AION-0.1 Evaluation Report",
        f"",
        f"**Model:** {m['model_name']} (`{m['model_id']}`)",
        f"**Evaluated:** {m['evaluated_at']}",
        f"**Parameters:** {m['param_count']:,}",
        f"",
        f"## Evaluation Metrics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Val loss | {ev['val_loss']:.4f} |" if ev['val_loss'] else "| Val loss | — |",
        f"| Perplexity | {ev['perplexity']:.2f} |" if ev['perplexity'] else "| Perplexity | — |",
        f"| Final train loss | {tm['final_loss']:.4f} |" if tm['final_loss'] else "| Final train loss | — |",
        f"| Epochs | {tm['epochs']} |" if tm['epochs'] else "| Epochs | — |",
        f"| Tokens processed | {tm['tokens_processed']:,} |" if tm['tokens_processed'] else "| Tokens processed | — |",
        f"| Training time | {tm['training_time_s']:.0f}s |" if tm['training_time_s'] else "| Training time | — |",
        f"",
        f"## Loss History",
        f"",
        f"| Epoch | Train Loss | Val Loss | Perplexity |",
        f"|---|---|---|---|",
    ]
    loss_hist = tm.get("loss_history", [])
    val_hist = tm.get("val_loss_history", [])
    ppl_hist = tm.get("val_perplexity_history", [])
    for i, tl in enumerate(loss_hist):
        vl = f"{val_hist[i]:.4f}" if i < len(val_hist) else "—"
        vp = f"{ppl_hist[i]:.2f}" if i < len(ppl_hist) else "—"
        lines.append(f"| {i+1} | {tl:.4f} | {vl} | {vp} |")
    lines += [
        f"",
        f"## Generated Samples",
        f"",
        f"All samples generated with greedy decoding, max {EVAL_MAX_NEW_TOKENS} tokens.",
        f"",
    ]
    for s in m["samples"]:
        lines += [
            f"### Prompt: {s['prompt']!r}",
            f"",
            f"```",
            s["full_text"],
            f"```",
            f"",
            f"*{s['tokens']} tokens in {s['elapsed_s']:.2f}s*",
            f"",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
