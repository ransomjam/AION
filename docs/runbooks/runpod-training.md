# Runbook — one complete training run: GitHub → RunPod → trained model → Camtinel

Every command is meant to be run as written. Values you must supply appear in
`<angle brackets>`.

Target for this run: **~12.4M parameters, ~10–25 minutes of GPU time, under $1.**
The size is set by the data (about 19M tokens), not by the card. This run exists
to prove the pipeline end to end; scale comes after it works once.

---

## Before you start

Four correctness bugs were fixed for this run. Two of them invalidate any
earlier checkpoint, so start from a clean slate:

| Fixed | Consequence |
|---|---|
| Attention head-merge leaked future tokens | every model trained before this is void |
| BPE never merged spaces into words | the tokenizer must be retrained |
| Resume under-recorded position in the epoch | second and later resumes replayed batches |
| Four `.data.flat[0]` sites and a per-layer mask copy | CuPy compatibility and 3.2 GB/step of PCIe traffic |

Delete stale artifacts before building:

```bash
rm -rf workspace/projects/aion-01/cache workspace/projects/aion-01/checkpoints
```

---

## Phase A — on your laptop (no GPU, no cost)

The corpus and the tokenizer are CPU work. Doing them before the pod exists is
the whole cost-control story: the tokenizer alone would otherwise burn an hour
of rental sitting in a Python loop.

### A1. Build the corpus

```bash
python scripts/build_cyber_corpus.py --max-cves 60000
```

25–40 minutes, almost all of it waiting on NVD's anonymous rate limit. It is
idempotent — if interrupted, run it again. For a faster first pass:

```bash
python scripts/build_cyber_corpus.py --max-cves 15000 --no-cheatsheets
```

### A2. Retrain the tokenizer

Mandatory. The old one never merged a space into the following word, so every
word cost two tokens and the bare space was 37.8% of the corpus.

```bash
python scripts/train_tokenizer.py --datasets aion-corpus aion-cyber --vocab-size 4096 --sample-chars 1500000 --retrain
```

**20–30 minutes.** Notes on the two flags:

- `--sample-chars 1500000` trains merge selection on an evenly-spread 1.5 MB
  sample. The trainer recomputes pair counts on every merge, so the full 67 MB
  corpus would take about 13 hours and choose almost exactly the same merges.
  The resulting tokenizer is still applied to the whole corpus afterwards.
- `--vocab-size 4096` rather than 8192. With ~19M tokens of training data, half
  of an 8192 vocabulary would be seen too rarely to learn a good embedding, and
  4096 trains in half the time.

The smoke test at the end prints characters-per-token for security strings.
Expect roughly **3.5**; the old tokenizer managed 2.33.

### A3. Push the code

```bash
git add -A && git commit -m "Causality, tokenizer, resume, and GPU fixes" && git push origin main
```

### A4. Pack the workspace

The workspace is data, not code, and is gitignored:

```bash
tar -czf aion-workspace.tar.gz workspace/projects/aion-01/data workspace/projects/aion-01/tokenizers
```

---

## Phase B — on RunPod

### B1. Create the pod

**Rent an RTX 4090** (~$0.34/hr). This engine computes in fp32, and the 4090's
fp32 throughput (82.6 TFLOPS) is higher than an H100's (67 TFLOPS) at an eighth
of the price — the H100's value is in tensor cores this code never touches.
Choose a **PyTorch / CUDA 12.x** template (nothing here uses PyTorch; the image
is a convenient CUDA environment) and at least **20 GB** of disk.

Connect, then confirm the card is real before anything else:

```bash
nvidia-smi
```

### B2. Clone and load data

```bash
cd /workspace
git clone https://github.com/ransomjam/AION.git
cd AION
```

Transfer the workspace. **On your laptop:**

```bash
runpodctl send aion-workspace.tar.gz
```

It prints a one-time code. **On the pod:**

```bash
runpodctl receive <CODE>
tar -xzf aion-workspace.tar.gz
```

### B3. Install dependencies

```bash
python --version
pip install --upgrade numpy
pip install cupy-cuda12x
```

Match the CuPy build to the CUDA major version `nvidia-smi` reports — use
`cupy-cuda11x` if it says 11.x. Wrong build, and the first kernel launch fails.

### B4. Verify before spending anything

```bash
export AION_DEVICE=cuda
python -m unittest discover -s aion -p "test_*.py"
python scripts/gpu_smoke.py
python scripts/gpu_smoke.py --compare
```

What you need to see:

- **769 tests pass on the GPU**, not just on your laptop
- `gpu_smoke.py` drives the loss from ~6.2 to under 3.0 — the model memorising
  one batch, which proves forward, backward, and optimiser all work on-device
- `--compare` reports CPU and CUDA agreeing to about **1e-4**

CPU and GPU are not bit-identical (different reduction orders in cuBLAS versus
OpenBLAS); rounding-scale agreement is the bar. **If `--compare` fails, stop.**
Training would produce arithmetic that is fast and wrong.

### B5. Train

Always inside `tmux`. An SSH drop kills a foreground process and the pod keeps
billing.

```bash
tmux new -s train
export AION_DEVICE=cuda
cd /workspace/AION
python scripts/train_aion_cyber.py --preset gpu 2>&1 | tee train.log
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t train`.

The `gpu` preset is `d_model=384, 6 layers, 6 heads, d_ff=1536, seq 512,
batch 32, 10 epochs` — about **12.4M parameters** and **~12 GB** of VRAM, which
leaves real headroom on a 24 GB card.

Watch from a second window:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```

If utilisation sits near 0%, `AION_DEVICE` is not set in that shell and you are
training on the pod's CPU. Stop and fix it.

### B6. Checkpoints

Written automatically every **500 steps and every 10 minutes** to:

```
workspace/projects/aion-01/checkpoints/run_<run_id>/
    latest/          model.npz, optim.npz, state.json   <- what --resume reads
    step_5000/       historical bundles, last 5 kept
```

Each bundle is written to a temp directory, fsync'd, and atomically renamed, so
a power loss cannot leave a half-written checkpoint. Checkpoints store host
arrays, so one written on the GPU pod loads on your laptop unchanged.

### B7. Resume after an interruption

```bash
export AION_DEVICE=cuda
python scripts/train_aion_cyber.py --preset gpu --resume
```

Continues from the exact interrupted step, mid-epoch, with optimizer moments,
scheduler position, and data order restored. Resuming repeatedly is now safe —
that was the fourth bug.

### B8. Export

```bash
python scripts/package_model.py --list
python scripts/package_model.py <MODEL_ID>
python scripts/export_for_camtinel.py <MODEL_ID>
python scripts/verify_export.py build/camtinel-<MODEL_ID>
```

Two different artifacts, both wanted:

| Command | Produces | For |
|---|---|---|
| `package_model.py` | `<MODEL_ID>.tar.gz` | archiving the run — weights, tokenizer, logs, per-file SHA-256 |
| `export_for_camtinel.py` | `build/camtinel-<MODEL_ID>/` | a portable bundle readable without AION |

`verify_export.py` re-reads the bundle with no AION imports and checks it
reproduces its own recorded logits. Run it on the pod before transferring —
finding a bad export after you have shut the pod down is an avoidable trip.

### B9. Transfer both back

On the pod:

```bash
tar -czf camtinel-export.tar.gz build/camtinel-<MODEL_ID>
runpodctl send <MODEL_ID>.tar.gz
runpodctl send camtinel-export.tar.gz
```

On your laptop, for each code:

```bash
cd C:\Users\user\Desktop\AI
runpodctl receive <CODE>
```

### B10. Stop the pod

Do this **now**, from the RunPod console, before evaluating anything. A running
pod bills whether or not it is computing.

---

## Phase C — on your laptop

### C1. Restore and check

```bash
python scripts/restore_model.py <MODEL_ID>.tar.gz
python scripts/evaluate_aion01.py <MODEL_ID>
```

`restore_model.py` verifies every checksum, installs the model and tokenizer,
then **generates text**. That last step matters: a checksum proves the bytes
arrived, not that the tokenizer matches the weights — and a vocabulary mismatch
is silent, producing fluent-looking nonsense rather than an error.

### C2. Read the result honestly

Report **bits per character**, not per-token perplexity. Perplexity depends on
the tokenizer, so it cannot be compared across tokenizers or against published
numbers. Bits per character is invariant:

```
bits_per_char = val_loss_nats / ln(2) / chars_per_token
```

`chars_per_token` is printed by `train_tokenizer.py` (expect ~3.5).

---

## Step 9: loading the model inside Camtinel

**This step cannot be completed tonight, and the blocker is in Camtinel, not
AION.**

Camtinel has no neural network runtime. Its dependencies are Compose, Room,
Gson, and Material — no TensorFlow Lite, no ONNX Runtime, no PyTorch Mobile.
`MlScorer.kt` reads a 2 KB logistic-regression blob with a hand-written
little-endian reader. There is nothing there that can execute a transformer.

What the export gives you is everything needed to close that gap deliberately:

```
build/camtinel-<MODEL_ID>/
    aion_model.bin        weights, float32 LE, with a tensor table
    aion_model.json       architecture + every tensor's offset and shape
    aion_tokenizer.json   vocabulary, merges, byte encoder, pre-token regex
    golden.json           a fixed prompt with its exact logits
    FORMAT.md             the byte-level specification
```

`golden.json` is the important one. It pins the exact logits the model produces
for a known prompt, so a reader written in Kotlin can be checked against a
correct answer instead of merely being made to compile.
`scripts/verify_export.py` is a working reference reader — it uses only
`struct` and `numpy`, imports nothing from AION, and reproduces those logits to
7.7e-07. Port that file, and `golden.json` tells you when the port is right.

Two routes from here, both real work and both Camtinel-side:

1. **Write a Kotlin reader** for this bundle — roughly a direct translation of
   `verify_export.py` (~150 lines: BPE encode, layer norm, attention, GELU).
   No new dependencies, full control, and it matches how Camtinel already ships
   models. A 12.4M fp32 model is ~50 MB of assets, which is large for an APK;
   quantising to int8 would bring it to ~12 MB and is a natural follow-up.
2. **Add ONNX Runtime or LiteRT** to Camtinel and write an ONNX exporter on the
   AION side. Less hand-written inference code, but a substantial new
   dependency in the app and a new export path in AION.

Either is a separate piece of work with its own plan. Say which you want and I
will scope it.

---

## What to expect

| | |
|---|---|
| First model size | ~12.4M parameters (`d_model` 384, 6 layers, 6 heads, `d_ff` 1536) |
| Context | 512 tokens |
| Batch | 32 (~12 GB VRAM, comfortable on a 24 GB 4090) |
| Corpus | ~67 MB, ~19M tokens |
| Training tokens | ~170M (10 epochs) |
| GPU | RTX 4090, ~$0.34/hr |
| Training time | **10–25 minutes** |
| Total pod time | budget **1 hour** including setup and transfers |
| **Expected cost** | **under $1** |
| Checkpoints | `workspace/projects/aion-01/checkpoints/run_<run_id>/latest/` |
| Run archive | `<MODEL_ID>.tar.gz` in the repo root |
| Camtinel bundle | `build/camtinel-<MODEL_ID>/` |

The estimate assumes 50% of the 4090's published fp32 peak and ~15 µs per CuPy
call. It has not been measured on real hardware — `gpu_smoke.py --benchmark`
will tell you the truth in about a minute, and it is worth running before you
walk away from the machine.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: AION_DEVICE=cuda requires CuPy` | CuPy not installed | `pip install cupy-cuda12x` |
| CuPy imports, first kernel fails | wheel/driver mismatch | check `nvidia-smi`, install `cupy-cuda11x` or `12x` to match |
| `--compare` reports disagreement | GPU path is wrong | do not train; send me the numbers it printed |
| GPU utilisation ~0% | `AION_DEVICE` unset in that shell | `export AION_DEVICE=cuda`, restart |
| Out of memory | batch too large | `--batch-size 16` and resume |
| Loss becomes NaN | learning rate too high | `--preset gpu` with a lower lr, resume from a step checkpoint |
| `restore_model.py` says corrupt | truncated transfer | re-send; do not use the archive |
| Vocabulary mismatch on restore | tokenizer retrained after training | re-package from the pod with the matching tokenizer |
| `verify_export.py` mismatch | bad export | re-run `export_for_camtinel.py`; do not ship the bundle |
