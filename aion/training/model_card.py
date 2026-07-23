"""ModelCard — generate a human-readable and machine-readable model card.

ModelCard.generate() returns a ModelCardResult containing:
    markdown    str     human-readable Markdown artifact
    data        dict    canonical machine-readable representation

The structured data is the source of truth; Markdown is rendered from it.
Both are written to the model directory by TrainingProject after training.

Model card sections
-------------------
- Model identity (name, version, architecture string)
- Architecture (d_model, n_heads, n_layers, d_ff, max_seq_len, param_count)
- Tokenizer (id, algorithm, vocab_size, vocabulary_fingerprint)
- Training corpus (dataset ids, fingerprint, n_documents, n_tokens, split)
- Training configuration (optimizer, lr, scheduler, epochs, batch_size, seed)
- Evaluation metrics (final_loss, val_loss, perplexity)
- Generation settings (default strategy, max_new_tokens)
- Known limitations (boilerplate + any user-supplied notes)
- License
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aion.util import now_iso


@dataclass
class ModelCardResult:
    markdown: str
    data: dict


class ModelCard:
    """Generate a model card for a trained GPT model.

    Parameters
    ----------
    model_name:
        Human-readable model name (e.g. ``"AION-0.1"``).
    gpt_config:
        ``GPTConfig.to_dict()`` snapshot.
    param_count:
        Total scalar parameter count.
    tokenizer_manifest:
        ``TokenizerStore.open_manifest(tokenizer_id)`` dict.
    corpus_stats:
        ``CorpusStats.to_dict()`` from the corpus build.
    corpus_fingerprint:
        ``DatasetFingerprint.combined`` string.
    dataset_ids:
        List of dataset ids used in the corpus.
    training_config:
        ``TrainingConfig.to_dict()`` snapshot.
    metrics:
        Training metrics dict from ``GPTTrainingResult.metrics``.
    model_id:
        Artifact id from ``GPTStore.save``.
    description:
        Optional free-text description.
    limitations:
        Optional list of known limitation strings.
    license_name:
        License identifier (default ``"proprietary"``).
    """

    def __init__(
        self,
        *,
        model_name: str,
        gpt_config: dict,
        param_count: int,
        tokenizer_manifest: dict,
        corpus_stats: dict,
        corpus_fingerprint: str,
        dataset_ids: list[str],
        training_config: dict,
        metrics: dict,
        model_id: str = "",
        description: str = "",
        limitations: list[str] | None = None,
        license_name: str = "proprietary",
    ) -> None:
        self._name = model_name
        self._gpt_config = gpt_config
        self._param_count = param_count
        self._tok = tokenizer_manifest
        self._corpus_stats = corpus_stats
        self._corpus_fp = corpus_fingerprint
        self._dataset_ids = dataset_ids
        self._training_config = training_config
        self._metrics = metrics
        self._model_id = model_id
        self._description = description
        self._limitations = limitations or [
            "Trained on a small corpus; outputs may be incoherent or repetitive.",
            "No safety filtering or alignment has been applied.",
            "Not suitable for production use.",
        ]
        self._license = license_name

    def generate(self) -> ModelCardResult:
        """Build and return a ``ModelCardResult``."""
        data = self._build_data()
        markdown = self._render_markdown(data)
        return ModelCardResult(markdown=markdown, data=data)

    # ── data builder ──────────────────────────────────────────────────────────

    def _build_data(self) -> dict:
        cfg = self._gpt_config
        tok = self._tok
        tc = self._training_config
        m = self._metrics
        return {
            "schema_version": 1,
            "generated_at": now_iso(),
            "model": {
                "id": self._model_id,
                "name": self._name,
                "description": self._description,
                "architecture": "gpt-v1",
                "param_count": self._param_count,
            },
            "architecture": {
                "d_model": cfg.get("d_model"),
                "n_heads": cfg.get("n_heads"),
                "n_layers": cfg.get("n_layers"),
                "d_ff": cfg.get("d_ff"),
                "max_seq_len": cfg.get("max_seq_len"),
                "vocab_size": cfg.get("vocab_size"),
                "activation": cfg.get("activation"),
                "tie_weights": cfg.get("tie_weights"),
                "pre_norm": cfg.get("pre_norm"),
            },
            "tokenizer": {
                "id": tok.get("id", ""),
                "name": tok.get("name", ""),
                "algorithm": tok.get("algorithm", ""),
                "vocab_size": tok.get("vocab_size"),
                "vocabulary_fingerprint": tok.get("vocabulary_fingerprint", ""),
            },
            "corpus": {
                "dataset_ids": self._dataset_ids,
                "fingerprint": self._corpus_fp,
                "n_documents": self._corpus_stats.get("n_documents"),
                "n_train_tokens": self._corpus_stats.get("n_train_tokens"),
                "n_val_tokens": self._corpus_stats.get("n_val_tokens"),
                "n_total_tokens": self._corpus_stats.get("n_total_tokens"),
                "vocab_coverage": self._corpus_stats.get("vocab_coverage"),
                "train_split": tc.get("train_split"),
            },
            "training": {
                "optimizer": tc.get("optimizer"),
                "learning_rate": tc.get("learning_rate"),
                "scheduler": tc.get("scheduler"),
                "warmup_steps": tc.get("warmup_steps"),
                "min_lr": tc.get("min_lr"),
                "epochs": tc.get("epochs"),
                "batch_size": tc.get("batch_size"),
                "context_length": tc.get("context_length"),
                "grad_clip": tc.get("grad_clip"),
                "seed": tc.get("seed"),
                "training_time_s": m.get("training_time_s"),
                "tokens_processed": m.get("tokens_processed"),
            },
            "evaluation": {
                "final_loss": m.get("final_loss"),
                "final_val_loss": m.get("final_val_loss"),
                "final_val_perplexity": m.get("final_val_perplexity"),
                "loss_history": m.get("loss_history", []),
                "val_loss_history": m.get("val_loss_history", []),
            },
            "generation": {
                "default_strategy": "greedy",
                "default_max_new_tokens": 128,
                "context_length": cfg.get("max_seq_len"),
            },
            "limitations": self._limitations,
            "license": self._license,
        }

    # ── Markdown renderer ─────────────────────────────────────────────────────

    def _render_markdown(self, d: dict) -> str:
        m = d["model"]
        arch = d["architecture"]
        tok = d["tokenizer"]
        corpus = d["corpus"]
        train = d["training"]
        ev = d["evaluation"]
        gen = d["generation"]

        def _fmt(v) -> str:
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        lines: list[str] = [
            f"# {m['name']}",
            "",
            m["description"] if m["description"] else "_No description provided._",
            "",
            f"**Model id:** `{m['id']}`  ",
            f"**Architecture:** {m['architecture']}  ",
            f"**Parameters:** {m['param_count']:,}  ",
            f"**Generated:** {d['generated_at']}",
            "",
            "## Architecture",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| d_model | {_fmt(arch['d_model'])} |",
            f"| n_heads | {_fmt(arch['n_heads'])} |",
            f"| n_layers | {_fmt(arch['n_layers'])} |",
            f"| d_ff | {_fmt(arch['d_ff'])} |",
            f"| max_seq_len | {_fmt(arch['max_seq_len'])} |",
            f"| vocab_size | {_fmt(arch['vocab_size'])} |",
            f"| activation | {_fmt(arch['activation'])} |",
            f"| tie_weights | {_fmt(arch['tie_weights'])} |",
            f"| pre_norm | {_fmt(arch['pre_norm'])} |",
            "",
            "## Tokenizer",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| id | `{tok['id']}` |",
            f"| name | {tok['name']} |",
            f"| algorithm | {tok['algorithm']} |",
            f"| vocab_size | {_fmt(tok['vocab_size'])} |",
            f"| vocabulary_fingerprint | `{tok['vocabulary_fingerprint'][:16]}` |",
            "",
            "## Training Corpus",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| datasets | {', '.join(f'`{d}`' for d in corpus['dataset_ids'])} |",
            f"| corpus_fingerprint | `{corpus['fingerprint'][:16]}` |",
            f"| documents | {_fmt(corpus['n_documents'])} |",
            f"| train_tokens | {_fmt(corpus['n_train_tokens'])} |",
            f"| val_tokens | {_fmt(corpus['n_val_tokens'])} |",
            f"| total_tokens | {_fmt(corpus['n_total_tokens'])} |",
            f"| vocab_coverage | {_fmt(corpus['vocab_coverage'])} |",
            f"| train_split | {_fmt(corpus['train_split'])} |",
            "",
            "## Training Configuration",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| optimizer | {_fmt(train['optimizer'])} |",
            f"| learning_rate | {_fmt(train['learning_rate'])} |",
            f"| scheduler | {_fmt(train['scheduler'])} |",
            f"| warmup_steps | {_fmt(train['warmup_steps'])} |",
            f"| min_lr | {_fmt(train['min_lr'])} |",
            f"| epochs | {_fmt(train['epochs'])} |",
            f"| batch_size | {_fmt(train['batch_size'])} |",
            f"| context_length | {_fmt(train['context_length'])} |",
            f"| grad_clip | {_fmt(train['grad_clip'])} |",
            f"| seed | {_fmt(train['seed'])} |",
            f"| training_time_s | {_fmt(train['training_time_s'])} |",
            f"| tokens_processed | {_fmt(train['tokens_processed'])} |",
            "",
            "## Evaluation Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| final_loss | {_fmt(ev['final_loss'])} |",
            f"| final_val_loss | {_fmt(ev['final_val_loss'])} |",
            f"| final_val_perplexity | {_fmt(ev['final_val_perplexity'])} |",
            "",
            "## Generation Settings",
            "",
            f"- Default strategy: **{gen['default_strategy']}**",
            f"- Default max new tokens: **{gen['default_max_new_tokens']}**",
            f"- Context length: **{_fmt(gen['context_length'])}**",
            "",
            "## Known Limitations",
            "",
        ]
        for lim in d["limitations"]:
            lines.append(f"- {lim}")
        lines += [
            "",
            "## License",
            "",
            d["license"],
            "",
        ]
        return "\n".join(lines)
