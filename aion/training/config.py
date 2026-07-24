"""TrainingConfig — serializable hyperparameter container for a training run.

Distinct from GPTConfig (architecture) and GenerationConfig (inference).
This is the single artifact needed to reproduce a run from scratch.
Written to logs/<run_id>/config.json at the start of every run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainingConfig:
    # ── identity ──────────────────────────────────────────────────────────────
    model_name: str = "aion-0.1"

    # ── model architecture (GPTConfig.to_dict() snapshot) ─────────────────────
    gpt_config: dict = field(default_factory=dict)

    # ── data ──────────────────────────────────────────────────────────────────
    dataset_ids: list[str] = field(default_factory=list)
    tokenizer_id: str = ""
    context_length: int = 512
    batch_size: int = 8
    train_split: float = 0.9

    # ── optimisation ──────────────────────────────────────────────────────────
    optimizer: str = "adam"          # "adam" | "sgd"
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    epochs: int = 10

    # ── learning rate schedule ────────────────────────────────────────────────
    scheduler: str = "cosine"        # "cosine" | "linear_warmup" | "constant"
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # ── checkpointing ─────────────────────────────────────────────────────────
    checkpoint_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3

    # ── step-based checkpointing (crash-safe, for long CPU runs) ──────────────
    # When ``checkpoint_every_n_steps`` or ``checkpoint_every_minutes`` is set,
    # the trainer writes complete, atomic checkpoints mid-epoch (model +
    # optimizer + scheduler + RNG + history + step/epoch) so a run can resume
    # from close to where it stopped instead of only at epoch boundaries.
    checkpoint_every_n_steps: int = 0        # 0 = disabled (epoch checkpoints only)
    checkpoint_every_minutes: float = 0.0    # 0 = disabled; wall-clock autosave
    keep_last_n_step_checkpoints: int = 5     # historical step_<N>/ dirs to keep
    log_every_n_steps: int = 50               # progress/ETA logging cadence

    # ── evaluation ────────────────────────────────────────────────────────────
    eval_every_n_epochs: int = 1
    eval_batches: int | None = None  # None = full validation set

    # ── sample generation ─────────────────────────────────────────────────────
    sample_every_n_epochs: int = 1
    sample_prompts: list[str] = field(default_factory=list)
    sample_max_new_tokens: int = 64

    # ── reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {self.context_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {self.optimizer!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.scheduler not in ("cosine", "linear_warmup", "constant"):
            raise ValueError(
                f"scheduler must be 'cosine', 'linear_warmup', or 'constant', "
                f"got {self.scheduler!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")
        if self.checkpoint_every_n_epochs <= 0:
            raise ValueError(
                f"checkpoint_every_n_epochs must be > 0, got {self.checkpoint_every_n_epochs}"
            )
        if self.keep_last_n_checkpoints <= 0:
            raise ValueError(
                f"keep_last_n_checkpoints must be > 0, got {self.keep_last_n_checkpoints}"
            )
        if self.checkpoint_every_n_steps < 0:
            raise ValueError(
                f"checkpoint_every_n_steps must be >= 0, got {self.checkpoint_every_n_steps}"
            )
        if self.checkpoint_every_minutes < 0:
            raise ValueError(
                f"checkpoint_every_minutes must be >= 0, got {self.checkpoint_every_minutes}"
            )
        if self.log_every_n_steps <= 0:
            raise ValueError(
                f"log_every_n_steps must be > 0, got {self.log_every_n_steps}"
            )
        if self.eval_every_n_epochs <= 0:
            raise ValueError(
                f"eval_every_n_epochs must be > 0, got {self.eval_every_n_epochs}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
