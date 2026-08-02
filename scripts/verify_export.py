"""verify_export.py — read an exported bundle with no AION imports and check it.

This is the reference reader.  It deliberately imports nothing from ``aion``:
it opens ``aion_model.bin`` with ``struct`` and ``numpy``, rebuilds the forward
pass from ``aion_model.json``, and checks the result against ``golden.json``.

Two jobs:

1. **Prove the bundle is self-sufficient.**  If this script needs AION, the
   bundle is not portable and the Android port will discover that the hard way.
2. **Be the thing a Kotlin reader is ported from.**  Every step below has a
   direct equivalent on Android, and ``golden.json`` gives that port a
   known-correct answer to check itself against, which is the difference
   between "it runs" and "it is right".

Usage
-----
    python scripts/verify_export.py build/camtinel-<model_id>
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"AION"


# ── bundle reader ─────────────────────────────────────────────────────────────

def load_bundle(directory: Path):
    meta = json.loads((directory / "aion_model.json").read_text(encoding="utf-8"))
    raw = (directory / "aion_model.bin").read_bytes()

    if raw[:4] != MAGIC:
        raise ValueError(f"bad magic: {raw[:4]!r}, expected {MAGIC!r}")
    version, count = struct.unpack("<II", raw[4:12])
    if version != meta["format_version"]:
        raise ValueError(f"binary says version {version}, JSON says {meta['format_version']}")
    if count != meta["tensor_count"]:
        raise ValueError(f"binary says {count} tensors, JSON lists {meta['tensor_count']}")

    # An ORDERED list, not a name->tensor map.  Order is the contract; names are
    # for humans.  See the note in the exporter about repeated parameter names.
    tensors = []
    for entry in meta["tensors"]:
        start = entry["byte_offset"]
        end = start + entry["byte_length"]
        tensors.append(np.frombuffer(raw[start:end], dtype="<f4").reshape(entry["shape"]))
    if len(tensors) != count:
        raise ValueError(f"read {len(tensors)} tensors, header declared {count}")
    return meta, tensors, [e["name"] for e in meta["tensors"]]


# ── tokenizer (byte-level BPE) ────────────────────────────────────────────────

class BundleTokenizer:
    def __init__(self, payload: dict) -> None:
        self.vocab = payload["vocab"]
        self.byte_encoder = {int(k): v for k, v in payload["byte_encoder"].items()}
        self.ranks = {(m[0], m[1]): i for i, m in enumerate(payload["merges"])}
        self.pattern = re.compile(payload["pretokenizer_regex"], re.UNICODE)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretoken in self.pattern.findall(text):
            symbols = [self.byte_encoder[b] for b in pretoken.encode("utf-8")]
            # Repeatedly apply the lowest-ranked merge that still applies.
            while len(symbols) > 1:
                pairs = [(self.ranks.get((symbols[i], symbols[i + 1]), None), i)
                         for i in range(len(symbols) - 1)]
                candidates = [(r, i) for r, i in pairs if r is not None]
                if not candidates:
                    break
                _, position = min(candidates)
                symbols[position:position + 2] = [symbols[position] + symbols[position + 1]]
            for symbol in symbols:
                ids.append(self.vocab[symbol])
        return ids


# ── forward pass ──────────────────────────────────────────────────────────────

def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * ((x - mean) / np.sqrt(var + eps)) + beta


def forward(meta, tensors, order, ids):
    cfg = meta["config"]
    d, H, L = cfg["d_model"], cfg["n_heads"], cfg["n_layers"]
    d_head = d // H
    seq = len(ids)
    T = tensors          # ordered exactly as written

    # Parameter order matches model.parameters(): embedding, positional table,
    # then per layer (ln1 g/b, W_Q, W_K, W_V, W_O, ln2 g/b, W_1, b_1, W_2, b_2),
    # then the final LayerNorm.  Consumed positionally, exactly as a Kotlin
    # reader would.
    cursor = 0
    def take():
        nonlocal cursor
        value = T[cursor]
        cursor += 1
        return value

    embedding = take()
    positional = take()
    x = embedding[np.array(ids)] + positional[:seq]

    causal = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)

    for _ in range(L):
        g1, b1 = take(), take()
        WQ, WK, WV, WO = take(), take(), take(), take()
        h = layer_norm(x, g1, b1)
        Q = (h @ WQ).reshape(seq, H, d_head).transpose(1, 0, 2)
        K = (h @ WK).reshape(seq, H, d_head).transpose(1, 0, 2)
        V = (h @ WV).reshape(seq, H, d_head).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head) + causal
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        # transpose BEFORE reshape - see FORMAT.md
        context = (weights @ V).transpose(1, 0, 2).reshape(seq, d)
        x = x + context @ WO

        g2, b2 = take(), take()
        W1, bias1, W2, bias2 = take(), take(), take(), take()
        h = layer_norm(x, g2, b2)
        x = x + gelu(h @ W1 + bias1) @ W2 + bias2

    gf, bf = take(), take()
    x = layer_norm(x, gf, bf)
    return x @ embedding.T if cfg.get("tie_weights") else x @ take()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/verify_export.py <bundle-directory>")
        sys.exit(2)
    directory = Path(sys.argv[1])
    if not (directory / "aion_model.bin").is_file():
        print(f"ERROR: no aion_model.bin in {directory}")
        sys.exit(1)

    meta, tensors, order = load_bundle(directory)
    cfg = meta["config"]
    print(f"Bundle: {directory}")
    print(f"  architecture {meta['architecture']}, {meta['param_count']:,} parameters")
    print(f"  d_model {cfg['d_model']}, heads {cfg['n_heads']}, layers {cfg['n_layers']}, "
          f"vocab {cfg['vocab_size']:,}")
    print(f"  tensors read: {len(tensors)}")

    golden = json.loads((directory / "golden.json").read_text(encoding="utf-8"))

    # 1. tokenizer round-trip against the recorded ids
    payload = json.loads((directory / "aion_tokenizer.json").read_text(encoding="utf-8"))
    ok_tokens = None
    if "vocab" in payload and "merges" in payload:
        ids = BundleTokenizer(payload).encode(golden["prompt"])
        ok_tokens = ids == golden["token_ids"]
        print(f"\n  tokenizer: encoded {len(ids)} ids, matches golden: {ok_tokens}")
        if not ok_tokens:
            print(f"    expected {golden['token_ids'][:12]}...")
            print(f"    got      {ids[:12]}...")
    else:
        print("\n  tokenizer: vocab/merges missing from the bundle, skipping")

    # 2. forward pass against the recorded logits
    logits = forward(meta, tensors, order, golden["token_ids"])
    expected = np.array(golden["logits_last"], dtype=np.float32)
    delta = float(np.abs(logits[-1] - expected).max())
    ok_logits = delta < 1e-3
    print(f"  forward pass: max |Δ| vs golden logits = {delta:.3e}  -> {'OK' if ok_logits else 'MISMATCH'}")
    print(f"  argmax token: {int(logits[-1].argmax())} "
          f"(golden {golden['argmax_token_id']}, {golden['argmax_token']!r})")

    if ok_logits and ok_tokens is not False:
        print("\nBundle verified. It reads and computes correctly with no AION imports.")
        return
    print("\nFAIL: the bundle does not reproduce its own golden values.")
    sys.exit(1)


if __name__ == "__main__":
    main()
