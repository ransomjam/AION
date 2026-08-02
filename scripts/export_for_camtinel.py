"""export_for_camtinel.py — write a trained model as a portable, self-describing bundle.

Produces a directory that can be read WITHOUT importing AION: a flat
little-endian float32 weight file, a JSON manifest describing every tensor's
offset and shape, and the tokenizer's vocabulary and merges.  The format
deliberately mirrors the convention Camtinel already uses for
``scam_classifier.bin`` + ``tfidf_vocab.json`` — a length-prefixed binary blob
beside a JSON sidecar — so it is familiar to whoever writes the reader.

Usage
-----
    python scripts/export_for_camtinel.py                 # newest model
    python scripts/export_for_camtinel.py <model_id>
    python scripts/export_for_camtinel.py <model_id> --out build/aion-export
    python scripts/export_for_camtinel.py --list

Output
------
    <out>/aion_model.bin       every parameter, concatenated, float32 LE
    <out>/aion_model.json      architecture + tensor table (name, offset, shape)
    <out>/aion_tokenizer.json  vocabulary + merge list
    <out>/golden.json          a fixed prompt with its exact logits
    <out>/FORMAT.md            the byte-level spec

What this does NOT do
---------------------
It does not make the model runnable on Android.  Camtinel today has no neural
network runtime — it ships Compose, Room, and Gson, and ``MlScorer`` reads a
2 KB logistic-regression blob by hand.  Running a transformer there needs an
inference implementation that does not exist yet, and writing one is a Camtinel
change, not an AION change.

What this bundle does is make that work possible and checkable: the format is
fully specified, and ``golden.json`` pins the exact logits the model produces
for a fixed prompt, so a new reader in any language can be verified against a
known-correct answer instead of "it compiles and returns numbers".
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion import backend
from aion.gpt.store import GPTStore
from aion.tokenizers.store import TokenizerStore
from aion.workspace.store import ProjectStore

PROJECT_NAME = "aion-01"
MAGIC = b"AION"
FORMAT_VERSION = 1
GOLDEN_PROMPT = "Message analysis\nChannel: SMS\nSender: MTN MoMo Alert\nMessage:\n"


FORMAT_SPEC = """# AION model bundle, format version 1

Four files. Everything needed to run the model; nothing that requires AION.

## aion_model.bin

    offset 0   4 bytes   magic "AION" (ASCII)
    offset 4   4 bytes   format version, uint32 little-endian
    offset 8   4 bytes   tensor count, uint32 little-endian
    offset 12  ...       parameter data, float32 little-endian, concatenated
                         in the order given by `tensors` in aion_model.json

Every numeric value in this file is little-endian. Tensor data is contiguous
and C-ordered (last axis varies fastest). `byte_offset` in the JSON is absolute
from the start of the file, so a reader can seek directly to any tensor without
walking the ones before it.

## aion_model.json

    format_version   this spec's version
    architecture     "gpt-v1"
    config           d_model, n_heads, n_layers, d_ff, vocab_size,
                     max_seq_len, activation, pre_norm, tie_weights
    param_count      total scalar count across all tensors
    tensors          ordered list of {name, shape, dtype, byte_offset, byte_length}

`tie_weights: true` means the output projection reuses the token-embedding
matrix; the bundle stores that tensor once and the reader must not expect a
separate copy.

## aion_tokenizer.json

    vocab            token string -> id
    merges           ordered list of ["left","right"] pairs; index = merge rank
    specials         {pad, unk, bos, eos} -> id
    byte_encoder     byte value -> the printable character standing for it

Encoding is byte-level BPE. Split text into pre-tokens with

    " ?\\w+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+"

(note the optional LEADING SPACE: " history" is one pre-token, not two), map
each pre-token to UTF-8 bytes, map each byte through `byte_encoder`, then
repeatedly apply the lowest-ranked applicable merge until none applies.

## golden.json

    prompt           a fixed input string
    token_ids        that prompt encoded
    logits_last      the model's logits for the final position, float32
    logits_sha256    SHA-256 of those logits as little-endian float32 bytes

A reader is correct when it reproduces `logits_last` from `prompt` to within
float32 rounding. Use this before trusting any port.

## Forward pass

Pre-norm decoder. For each of `n_layers` blocks:

    h = x + Attention(LayerNorm(x))
    y = h + FFN(LayerNorm(h))

Attention is causal multi-head: split `d_model` into `n_heads` heads of
`d_model / n_heads`, scale scores by `1/sqrt(d_head)`, add a mask of -1e9 above
the diagonal, softmax, then weight the values.

**When merging heads back, transpose before reshaping.** Going from
`[batch, heads, seq, d_head]` to `[batch, seq, d_model]` requires
`transpose(0, 2, 1, 3)` first. Reshaping directly reinterprets the head and
position axes as each other, which silently leaks future positions past the
causal mask. That exact bug shipped in AION and cost a full training run; the
symptom is a model that scores well and generates nonsense.

FFN is `Linear -> GELU (tanh approximation) -> Linear`. Positional encoding is
a learned table added to the token embedding.
"""


def _sha256_of(array: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model as a portable bundle.")
    parser.add_argument("model_id", nargs="?", default=None)
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    gpt_store = GPTStore(project.dir("models"))

    models = gpt_store.list()
    if not models:
        print("ERROR: no trained models in this project.")
        sys.exit(1)
    if args.list:
        for manifest in models:
            print(f"{manifest['id']:<34} {manifest.get('name',''):<20} "
                  f"{manifest.get('created_at','')}")
        return

    model_id = args.model_id or models[0]["id"]
    if not gpt_store.exists(model_id):
        print(f"ERROR: no model {model_id!r}. Use --list.")
        sys.exit(1)

    manifest = gpt_store.open_manifest(model_id)
    model = gpt_store.load(model_id)
    cfg = model.cfg

    tokenizer_id = manifest.get("tokenizer_id") or \
        (project.manifest.get("defaults", {}) or {}).get("tokenizer")
    if not tokenizer_id:
        print("ERROR: this model records no tokenizer id; the export would be unusable.")
        sys.exit(1)
    tokenizer = TokenizerStore(project.dir("tokenizers")).load(tokenizer_id)

    out = Path(args.out) if args.out else ROOT / "build" / f"camtinel-{model_id}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_id}")
    print(f"  parameters:  {model.param_count():,}")
    print(f"  tokenizer:   {tokenizer_id} (vocab {tokenizer.vocab_size:,})")

    # ── aion_model.bin ────────────────────────────────────────────────────────
    # Tied weights appear once in model.parameters(), so writing that list
    # directly is already de-duplicated.
    params = model.parameters()
    header = MAGIC + struct.pack("<II", FORMAT_VERSION, len(params))
    tensors: list[dict] = []
    offset = len(header)
    blobs: list[bytes] = []
    for index, p in enumerate(params):
        data = np.ascontiguousarray(backend.to_host(p.data), dtype="<f4")
        raw = data.tobytes()
        tensors.append({
            # Index-prefixed, matching the checkpoint convention.  Parameter
            # names repeat across layers (every LayerNorm has a "gamma"), so a
            # bare name is not a key — a reader that builds a name->tensor map
            # would silently keep one layer's weights and drop the rest.
            "name": f"p{index}__{p.name or 'param'}",
            "shape": list(data.shape),
            "dtype": "float32",
            "byte_offset": offset,
            "byte_length": len(raw),
        })
        blobs.append(raw)
        offset += len(raw)

    bin_path = out / "aion_model.bin"
    with bin_path.open("wb") as handle:
        handle.write(header)
        for raw in blobs:
            handle.write(raw)

    # ── aion_model.json ───────────────────────────────────────────────────────
    (out / "aion_model.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "architecture": manifest.get("architecture", "gpt-v1"),
        "model_id": model_id,
        "model_name": manifest.get("name"),
        "config": cfg.to_dict(),
        "param_count": model.param_count(),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }, indent=2), encoding="utf-8")

    # ── aion_tokenizer.json ───────────────────────────────────────────────────
    # The store writes vocabulary.json / merges.json with a small wrapper around
    # the payload; flatten it here so the bundle is one level deep and a reader
    # does not have to know the store's layout.
    tok_dir = Path(project.dir("tokenizers")) / tokenizer_id / "model"
    tok_payload: dict = {"vocab_size": tokenizer.vocab_size}
    vocab_path = tok_dir / "vocabulary.json"
    merges_path = tok_dir / "merges.json"
    if vocab_path.is_file():
        stored = json.loads(vocab_path.read_text(encoding="utf-8"))
        tokens = stored.get("tokens", stored)
        # ``tokens`` is an ordered list of token strings; the id IS the index.
        tok_payload["vocab"] = ({t: i for i, t in enumerate(tokens)}
                                if isinstance(tokens, list) else tokens)
    if merges_path.is_file():
        stored = json.loads(merges_path.read_text(encoding="utf-8"))
        tok_payload["merges"] = stored.get("merges", stored)
    # The byte encoder is derived, not stored: rebuild it from the same function
    # the tokenizer uses so the bundle never disagrees with the trained model.
    from aion.tokenizers.bpe import _build_byte_encoder
    tok_payload["byte_encoder"] = {str(k): v for k, v in _build_byte_encoder().items()}
    tok_payload["specials"] = {
        "pad": tokenizer.pad_id, "unk": tokenizer.unk_id,
        "bos": tokenizer.bos_id, "eos": tokenizer.eos_id,
    }
    tok_payload["pretokenizer_regex"] = r" ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"
    (out / "aion_tokenizer.json").write_text(
        json.dumps(tok_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if "vocab" not in tok_payload:
        print("  WARNING: tokenizer vocab file not found; the bundle is incomplete.")

    # ── golden.json ───────────────────────────────────────────────────────────
    # A known-correct answer, so a reader in another language can be verified
    # rather than merely made to run.
    model.eval()
    ids = tokenizer.encode(GOLDEN_PROMPT)[: cfg.max_seq_len]
    logits, _ = model(np.array([ids], dtype=np.int32))
    last = backend.to_host(logits.data)[0, -1].astype(np.float32)
    (out / "golden.json").write_text(json.dumps({
        "prompt": GOLDEN_PROMPT,
        "token_ids": [int(i) for i in ids],
        "logits_last": [float(v) for v in last],
        "logits_sha256": _sha256_of(last),
        "argmax_token_id": int(last.argmax()),
        "argmax_token": tokenizer.decode([int(last.argmax())]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "FORMAT.md").write_text(FORMAT_SPEC, encoding="utf-8")

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    line = "-" * 62
    print(f"""
Export written.
{line}
  Directory:   {out}
  aion_model.bin       {bin_path.stat().st_size / 1e6:>8.1f} MB  ({len(tensors)} tensors)
  aion_model.json      {(out / 'aion_model.json').stat().st_size / 1e3:>8.1f} KB
  aion_tokenizer.json  {(out / 'aion_tokenizer.json').stat().st_size / 1e3:>8.1f} KB
  golden.json          {(out / 'golden.json').stat().st_size / 1e3:>8.1f} KB
  FORMAT.md            the byte-level spec
  Total                {total / 1e6:>8.1f} MB
{line}

Verify the bundle reads back correctly:

    python scripts/verify_export.py {out}
""")


if __name__ == "__main__":
    main()
