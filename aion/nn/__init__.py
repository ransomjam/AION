"""aion.nn — AION's neural network framework.

Built on a minimal tape-based autograd engine.  Every value that flows through
a model is a ``Tensor``; every trainable weight is a ``Parameter``; every
component is a ``Module``.

Public surface
--------------
Tensor, tensor          — the central value type
Parameter               — trainable leaf tensor
Module                  — base class for all layers and models
Sequential              — ordered module container
Linear                  — affine layer
EmbeddingLayer          — differentiable token embedding table
ReLU, Tanh, Sigmoid     — stateless activation modules
CrossEntropyLoss        — numerically stable classification loss
MSELoss                 — mean squared error
BCELoss                 — binary cross-entropy
SGD, Adam               — optimizers
Trainer, TrainingResult — training loop
ModelStore, ModelNotFound — model persistence
"""

from .tensor import Tensor, tensor
from .parameter import Parameter
from .module import Module
from .layers import EmbeddingLayer, Linear, ReLU, Sigmoid, Tanh
from .sequential import Sequential
from .loss import BCELoss, CrossEntropyLoss, MSELoss
from .optim import Adam, SGD
from .trainer import Trainer, TrainingResult
from .store import ModelNotFound, ModelStore

__all__ = [
    "Tensor", "tensor",
    "Parameter",
    "Module",
    "Sequential",
    "Linear", "EmbeddingLayer",
    "ReLU", "Tanh", "Sigmoid",
    "CrossEntropyLoss", "MSELoss", "BCELoss",
    "SGD", "Adam",
    "Trainer", "TrainingResult",
    "ModelStore", "ModelNotFound",
]
