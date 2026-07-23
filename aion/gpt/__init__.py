"""aion.gpt — GPT-style language model package.

Public surface
--------------
GPTConfig               hyperparameter container
GPTModel                decoder-only language model
LanguageModelHead       vocabulary projection layer
CausalLanguageModelLoss next-token prediction loss
TokenizedDataset        packed sequence dataset
BatchSampler            batch iterator over TokenizedDataset
GPTTrainer              language-model training loop
GPTTrainingResult       training outcome dataclass
GPTCheckpoint           mid-training checkpoint save/load
GPTStore                model persistence store
GPTNotFound             raised when a model id is not found
record_gpt_experiment   record a training run in ExperimentStore
"""

from .checkpoint import GPTCheckpoint
from .config import GPTConfig
from .data import BatchSampler, TokenizedDataset
from .experiment import record_gpt_experiment
from .loss import CausalLanguageModelLoss
from .model import GPTModel, LanguageModelHead
from .store import GPTNotFound, GPTStore
from .trainer import GPTTrainer, GPTTrainingResult, TrainingCallback, TrainingCallbackList

__all__ = [
    "GPTConfig",
    "GPTModel",
    "LanguageModelHead",
    "CausalLanguageModelLoss",
    "TokenizedDataset",
    "BatchSampler",
    "GPTTrainer",
    "GPTTrainingResult",
    "TrainingCallback",
    "TrainingCallbackList",
    "GPTCheckpoint",
    "GPTStore",
    "GPTNotFound",
    "record_gpt_experiment",
]
