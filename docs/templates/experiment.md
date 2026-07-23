# Experiment document template

An experiment answers a research question — it is not a feature. Copy this into
`experiments/<id>-<slug>/README.md`, fill it in, delete this block. No
experiment is ever deleted; a superseded one is marked superseded, not removed.

---

# Experiment <id>: <title>

- **Date:**
- **Author:**
- **Status:** planned | running | complete | superseded

## Research question
The single question this experiment answers. If you cannot state it in one
sentence, it is more than one experiment.

## Hypothesis
What you expect to happen, and why, *before* running it.

## Setup
- Dataset (name + version + fingerprint; "none" is a valid answer for algorithmic
  experiments).
- Algorithm / configuration.
- Hyperparameters.
- Random seed(s). Reproducibility is mandatory: the same seed must give the same
  result.

## How to reproduce
The exact command(s). Someone else must be able to run this and get your numbers.

## Results
Metrics and plots. Report them honestly, including the ones that disappoint.

## Observations & lessons learned
What actually happened, why, and what it changes about our understanding. A
negative result with a clear lesson is a successful experiment.
