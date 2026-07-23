"""aion.training — Foundation Model Training Lab.

Orchestrates reproducible GPT training runs on top of the existing platform.

Public surface
--------------
TrainingConfig          serializable hyperparameter container for a run
DatasetFingerprint      stable identity for a corpus build (cache key)
CorpusManager           assemble + cache a packed corpus from project datasets
LearningRateScheduler   abstract scheduler protocol
ConstantScheduler       constant LR
LinearWarmupScheduler   linear warmup then constant
CosineDecayScheduler    linear warmup then cosine decay to min_lr
SchedulerCallback       TrainingCallback that applies a scheduler each step
CheckpointManager       checkpoint lifecycle: save, prune, best tracking
CheckpointCallback      TrainingCallback that saves checkpoints each N epochs
ResumeTraining          restore a run from its latest checkpoint
MetricsCollector        accumulate + persist per-step and per-epoch metrics
EvaluationRunner        compute validation loss and perplexity on a held-out set
SampleGenerator         generate text samples from a model during training
TrainingProject         top-level orchestrator: wires all components for one run
TrainingDashboard       pure formatting layer over MetricsCollector state
ModelCard               generate Markdown + structured data for a trained model
ModelCardResult         (markdown, data) pair returned by ModelCard.generate()
"""

from .checkpoint import CheckpointCallback, CheckpointManager
from .config import TrainingConfig
from .corpus import CorpusManager, CorpusResult, CorpusStats
from .dashboard import TrainingDashboard
from .evaluation import EvaluationRunner
from .fingerprint import DatasetFingerprint
from .metrics import MetricsCollector, TrainingMetrics
from .model_card import ModelCard, ModelCardResult
from .project import TrainingProject
from .resume import ResumeState, ResumeTraining
from .sampler import SampleGenerator
from .scheduler import (
    ConstantScheduler,
    CosineDecayScheduler,
    LearningRateScheduler,
    LinearWarmupScheduler,
    SchedulerCallback,
    build_scheduler,
)

__all__ = [
    "TrainingConfig",
    "DatasetFingerprint",
    "CorpusManager",
    "CorpusResult",
    "CorpusStats",
    "LearningRateScheduler",
    "ConstantScheduler",
    "LinearWarmupScheduler",
    "CosineDecayScheduler",
    "SchedulerCallback",
    "build_scheduler",
    "CheckpointManager",
    "CheckpointCallback",
    "ResumeTraining",
    "ResumeState",
    "MetricsCollector",
    "TrainingMetrics",
    "EvaluationRunner",
    "SampleGenerator",
    "TrainingProject",
    "TrainingDashboard",
    "ModelCard",
    "ModelCardResult",
]
