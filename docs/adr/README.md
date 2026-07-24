# Architecture Decision Records

Every significant, hard-to-reverse decision is recorded here as a short,
numbered, immutable document. Superseded decisions are marked superseded and
kept — we never rewrite history, we append to it. This is how AION keeps its
promise that every architectural choice can be justified years later.

## Format

One file per decision: `NNNN-short-slug.md`, with sections **Status**,
**Context**, **Decision**, **Alternatives considered**, **Consequences**.
Status is one of: proposed | accepted | superseded (by NNNN) | deprecated.

## Index

- [0001 — Pivot from Camtinel to Project AION](0001-pivot-from-camtinel-to-aion.md)
- [0002 — Reorient from educational lab to AI engineering platform](0002-reorient-to-engineering-platform.md)
- [0003 — The application is AION's operating environment](0003-application-as-operating-environment.md)
- [0004 — Workspace and Projects as the platform's unit](0004-workspace-and-projects.md)
- [0005 — Project cache and the Job abstraction](0005-cache-and-jobs.md)
- [0006 — Tokenizer Framework and Byte-Level BPE](0006-tokenizer-framework-and-bpe.md)
- [0007 — Embedding Framework and CBOW](0007-embedding-framework-and-cbow.md)
- [0008 — Neural Network Framework and Tape-Based Autograd](0008-neural-network-framework-and-autograd.md)
- [0009 — Attention Framework](0009-attention-framework.md)
- [0010 — Transformer Framework](0010-transformer-framework.md)
- [0011 — GPT Language Model](0011-gpt-language-model.md)
- [0012 — Inference & Generation Engine](0012-inference-generation-engine.md)
- [0013 — Step-based, crash-safe checkpointing](0013-step-based-crash-safe-checkpointing.md)
