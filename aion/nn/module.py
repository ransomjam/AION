"""Module — the base class for every layer and model.

Every trainable component in AION subclasses ``Module``.  The contract is
minimal and forward-only: subclasses implement ``forward()``; backward is
handled entirely by the autograd engine via ``loss.backward()``.

Parameter registration
----------------------
``Module.__setattr__`` intercepts assignments of ``Parameter`` and ``Module``
instances and registers them in ``_parameters`` and ``_modules`` dicts
respectively.  This means:

    class Linear(Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.W = Parameter(...)   # auto-registered — no boilerplate
            self.b = Parameter(...)   # auto-registered

``parameters()`` performs a recursive walk over ``_parameters`` and all
submodule ``_modules``, returning a flat list of every ``Parameter`` in the
tree.  Duplicate parameters (shared weights) appear once.

``zero_grad()`` calls ``param.zero_grad()`` on every parameter, setting
``.grad = None`` to signal that no gradient has been computed yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .parameter import Parameter
from .tensor import Tensor


class Module(ABC):
    """Abstract base for all AION neural network components."""

    def __init__(self) -> None:
        # Use object.__setattr__ to bypass our own override during __init__.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # ── attribute interception ────────────────────────────────────────────────

    def __setattr__(self, name: str, value) -> None:
        # Remove from registries if replacing an existing entry.
        params = object.__getattribute__(self, "_parameters")
        modules = object.__getattribute__(self, "_modules")
        params.pop(name, None)
        modules.pop(name, None)

        if isinstance(value, Parameter):
            params[name] = value
        elif isinstance(value, Module):
            modules[name] = value

        object.__setattr__(self, name, value)

    # ── parameter access ──────────────────────────────────────────────────────

    def parameters(self) -> list[Parameter]:
        """Return all parameters in this module and its submodules.

        Traversal is depth-first, registration order within each module.
        Shared parameters (same object) appear exactly once.
        """
        seen: set[int] = set()
        result: list[Parameter] = []

        def _collect(mod: Module) -> None:
            for p in mod._parameters.values():
                if id(p) not in seen:
                    seen.add(id(p))
                    result.append(p)
            for child in mod._modules.values():
                _collect(child)

        _collect(self)
        return result

    def zero_grad(self) -> None:
        """Set all parameter gradients to ``None``."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Set this module and all submodules to training mode.

        In training mode, ``Dropout`` applies random masks.  In evaluation
        mode it is an identity.  Returns ``self`` for chaining.
        """
        object.__setattr__(self, "training", mode)
        for child in object.__getattribute__(self, "_modules").values():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        """Set this module and all submodules to evaluation mode."""
        return self.train(False)

    # ── forward contract ──────────────────────────────────────────────────────

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Compute the output.  Subclasses implement this; do not call directly."""

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    # ── introspection ─────────────────────────────────────────────────────────

    def param_count(self) -> int:
        """Total number of scalar values across all parameters."""
        return builtins_sum(p.data.size for p in self.parameters())

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, child in self._modules.items():
            lines.append(f"  ({name}): {child}")
        lines.append(")")
        return "\n".join(lines) if len(lines) > 2 else f"{type(self).__name__}()"


# Alias to avoid shadowing Python's built-in ``sum`` inside the class body.
import builtins
builtins_sum = builtins.sum
