# Mission & Vision

## What AION is

AION is an independent AI research and engineering platform. Its objective is to
**build AI** — to design, train, evaluate, and continuously improve our own
family of models — not to explain it. The mindset is that of an engineer inside
a frontier lab building internal systems: everything exists to move model
development forward.

The single decision criterion, applied to every module and every commit:

> **Does this move AION closer to training, evaluating, and improving our own
> models?** If the answer is no, it probably should not exist.

## Where this came from

AION is the successor to **Camtinel**, an offline Android threat-detection
system (preserved at the git tag `camtinel-final`). Building it produced four
lessons AION is founded on, all of them engineering lessons:

1. **A well-designed architecture is reusable.** Camtinel's evidence-fusion
   *math* was domain-neutral even though its detectors were not — worth
   extracting deliberately when a use arises, not assuming for free (see below).
2. **Systems are assembled from independent components**, not one monolith.
3. **Data is the leverage point.** Curation, cleaning, and versioning of data
   moved quality more than features did.
4. **Research infrastructure is first-class.** Reproducibility, evaluation, and
   versioning are load-bearing, not nice-to-haves.

### One verified caveat on lesson 1

We checked lesson 1 against Camtinel's actual code rather than assuming it. The
fusion *mathematics* (noisy-OR over independent confidences, hypothesis scoring
against a signal set) is genuinely reusable, **but the interfaces around it were
security-coupled** (hardcoded signal names, a fixed threat-category set, baked-in
advice). So "extract a reusable inference/fusion kernel" is a scoped engineering
task we take on when a model actually needs it — not a settled capability.

## What we are building toward

A platform that, over time, can: collect and version data, clean and prepare it,
train tokenizers and models, checkpoint and register them, evaluate and benchmark
them, run inference, fine-tune, and iterate — each piece added when the next
stage of model work requires it. The chat interface, if it appears, is a
continuous data-collection engine, not a product.

## How we judge quality (honesty, not modesty)

We hold a strict honesty invariant: **we measure, we do not claim.** No quality
number exists without the dataset version and the run that produced it;
provenance is tracked; synthetic data is never reported as genuine.

This is a discipline, not a ceiling. We are building real models and expect their
scale and quality to grow with data, compute, and iteration. We simply refuse to
assert results we have not measured — which is exactly how a serious lab keeps
its own numbers trustworthy.

## How we decide what to build

Before implementing anything, answer:

- Why do we need this, and what future capability does it unlock?
- How will it be reused, and will future models depend on it?
- Can it scale, and can it be benchmarked?
- Can it be replaced or evolved later without rewriting the system?

If those don't have good answers, reconsider the design. And we build
infrastructure only when it enables the next stage of research — never because a
layout looks complete.

The capability roadmap is in [roadmap.md](roadmap.md).
