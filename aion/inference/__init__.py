"""aion.inference — inference and generation engine for decoder-only models.

Public surface
--------------
GenerationConfig            serializable generation configuration
Generator                   main inference engine
LogitsProcessor             base class for logits processors
LogitsProcessorList         sequential processor pipeline
TemperatureProcessor        divide logits by temperature
TopKProcessor               keep only top-k logits
TopPProcessor               nucleus (top-p) filtering
RepetitionPenaltyProcessor  penalise repeated tokens
Sampler                     base class for samplers
GreedySampler               deterministic argmax
StochasticSampler           multinomial draw
StoppingCriteria            base class for stopping criteria
StoppingCriteriaList        fires when any criterion fires
MaxNewTokensCriteria        stop after N tokens
EosTokenCriteria            stop on EOS token
StopSequenceCriteria        stop on stop string suffix
TokenStep                   per-step generation data
GenerationMetrics           aggregate generation statistics
GenerationResult            complete generation output
StreamingCallback           base streaming callback (no-op)
CollectingCallback          callback that collects tokens
GenerationVisualizer        JSON-serializable visualization data
record_inference_experiment record a generation run in ExperimentStore
"""

from .config import GenerationConfig
from .experiment import record_inference_experiment
from .generator import Generator
from .processors import (
    LogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyProcessor,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
)
from .result import GenerationMetrics, GenerationResult, TokenStep
from .sampler import GreedySampler, Sampler, StochasticSampler
from .stopping import (
    EosTokenCriteria,
    MaxNewTokensCriteria,
    StopSequenceCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)
from .streaming import CollectingCallback, StreamingCallback
from .visualizer import GenerationVisualizer

__all__ = [
    "GenerationConfig",
    "Generator",
    "LogitsProcessor",
    "LogitsProcessorList",
    "TemperatureProcessor",
    "TopKProcessor",
    "TopPProcessor",
    "RepetitionPenaltyProcessor",
    "Sampler",
    "GreedySampler",
    "StochasticSampler",
    "StoppingCriteria",
    "StoppingCriteriaList",
    "MaxNewTokensCriteria",
    "EosTokenCriteria",
    "StopSequenceCriteria",
    "TokenStep",
    "GenerationMetrics",
    "GenerationResult",
    "StreamingCallback",
    "CollectingCallback",
    "GenerationVisualizer",
    "record_inference_experiment",
]
