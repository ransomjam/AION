# Engineering component document template

Every capability in `aion/` ships with a `README.md` following this template.
Copy it, fill every section, delete this block. Keep it tight — this is a
reference for engineers extending or depending on the component, not a tutorial.

---

# <Component name>

## Role in the platform
What model-building capability this provides, and what depends on it. One or two
sentences. State the concrete future work it unlocks.

## Contract / API
The public surface: functions/classes, their inputs and outputs, and the
guarantees they make. What callers can rely on and what they must not.

## Invariants
The properties that must hold (e.g. determinism, mandatory tokens, immutability
after build). Each should have a test. Note anything that silently corrupts
downstream state if violated.

## Design notes
Non-obvious choices and why, deviations from the naive approach, and where this
diverges from prior art (Camtinel or elsewhere) with the reason.

## Performance & scaling
Time/space complexity in terms of the inputs that grow (corpus size, vocab size,
sequence length). Known bottlenecks and how the component is expected to scale or
be replaced when it stops scaling.

## Benchmarks
Numbers, when we have them, with the exact command to reproduce. "None yet" is a
valid answer — but never fabricate one.

## Tests
What the suite covers and how to run it.

## References
Primary sources for the algorithm or format, when relevant.
