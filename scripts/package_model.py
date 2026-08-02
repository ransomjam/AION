"""package_model.py — bundle a trained model for transfer off the pod.

Run this on the RunPod machine when training finishes.  It collects everything
needed to *use* the model somewhere else, records a checksum for each file, and
writes a single ``.tar.gz`` ready to copy home.

Usage
-----
    python scripts/package_model.py                       # newest model
    python scripts/package_model.py <model_id>
    python scripts/package_model.py <model_id> --with-checkpoints
    python scripts/package_model.py --list

What goes in
------------
- ``models/<model_id>/``   weights, architecture, statistics, model card
- ``tokenizers/<id>/``     the tokenizer the model was trained against
- ``logs/<run_id>/``       metrics, the config that produced it, the device record

The tokenizer is included deliberately.  Weights without the exact vocabulary
that produced them are unusable — token id 4,182 means nothing on its own — and
the most common way a transferred model turns into garbage is arriving next to
a tokenizer that was retrained in the meantime.

What stays behind
-----------------
Checkpoints are excluded by default.  They contain optimiser moment buffers,
roughly tripling the archive, and they exist to resume training, not to run the
model.  Pass ``--with-checkpoints`` only when the intention is to continue the
run on other hardware.

The corpus stays behind too: it is reproducible from
``scripts/build_cyber_corpus.py`` and its fingerprint is recorded in the model
card, so shipping tens of megabytes of source text would add nothing that the
fingerprint does not already establish.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion.gpt.store import GPTStore
from aion.workspace.store import ProjectStore

PROJECT_NAME = "aion-01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def collect(root: Path, base: Path) -> list[tuple[Path, str]]:
    """Every file under ``root``, paired with its path relative to ``base``."""
    if not root.exists():
        return []
    return [(path, str(path.relative_to(base)).replace("\\", "/"))
            for path in sorted(root.rglob("*")) if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a trained model for transfer.")
    parser.add_argument("model_id", nargs="?", default=None,
                        help="model to package (default: the most recent)")
    parser.add_argument("--with-checkpoints", action="store_true",
                        help="include optimiser state so training can be continued elsewhere")
    parser.add_argument("--out", default=None, help="output path for the archive")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()

    store = ProjectStore(ROOT / "workspace")
    project = store.open(PROJECT_NAME)
    project_dir = Path(project.dir("models")).parent
    gpt_store = GPTStore(project.dir("models"))

    models = gpt_store.list()
    if not models:
        print("ERROR: no trained models in this project. Nothing to package.")
        sys.exit(1)

    if args.list:
        print(f"{'model id':<34} {'name':<20} created")
        for manifest in models:
            print(f"{manifest['id']:<34} {manifest.get('name', ''):<20} "
                  f"{manifest.get('created_at', '')}")
        return

    model_id = args.model_id or models[0]["id"]
    if not gpt_store.exists(model_id):
        print(f"ERROR: no model {model_id!r}. Run with --list to see what is available.")
        sys.exit(1)

    manifest = gpt_store.open_manifest(model_id)
    print(f"Packaging {model_id}")
    print(f"  name:        {manifest.get('name')}")
    print(f"  parameters:  {manifest.get('param_count', 0):,}")

    # ── resolve the tokenizer and run this model belongs to ───────────────────
    tokenizer_id = manifest.get("tokenizer_id") or \
        (project.manifest.get("defaults", {}) or {}).get("tokenizer")
    params = manifest.get("params") or {}
    run_id = params.get("run_id")
    if not run_id:
        # Models saved before run_id was recorded in the manifest: fall back to
        # the run registry, which maps a model name to its most recent run.
        registry_path = project_dir / "checkpoints" / "active_runs.json"
        if registry_path.is_file():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                run_id = registry.get(manifest.get("name", ""))
            except (json.JSONDecodeError, OSError):
                run_id = None

    members: list[tuple[Path, str]] = []
    members += collect(project_dir / "models" / model_id, project_dir)

    if tokenizer_id:
        tokenizer_files = collect(project_dir / "tokenizers" / tokenizer_id, project_dir)
        if tokenizer_files:
            print(f"  tokenizer:   {tokenizer_id}")
            members += tokenizer_files
        else:
            print(f"  WARNING: tokenizer {tokenizer_id} not found on disk. The archive "
                  f"will not be usable on its own.")
    else:
        print("  WARNING: this model records no tokenizer id. Include one manually.")

    if run_id:
        members += collect(project_dir / "logs" / run_id, project_dir)
        if args.with_checkpoints:
            checkpoints = collect(project_dir / "checkpoints" / run_id, project_dir)
            print(f"  checkpoints: {len(checkpoints)} files "
                  f"(optimiser state included, archive will be large)")
            members += checkpoints

    if not members:
        print("ERROR: nothing was collected. The model directory appears to be empty.")
        sys.exit(1)

    # ── checksums ─────────────────────────────────────────────────────────────
    # Written into the archive so the receiving side can verify the transfer
    # rather than trusting that a multi-hundred-megabyte copy over the public
    # internet arrived intact.
    print(f"\nChecksumming {len(members)} files...")
    checksums = {arcname: sha256_file(path) for path, arcname in members}
    total_bytes = sum(path.stat().st_size for path, _ in members)

    transfer_manifest = {
        "schema_version": 1,
        "packaged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": PROJECT_NAME,
        "model_id": model_id,
        "model_name": manifest.get("name"),
        "param_count": manifest.get("param_count"),
        "tokenizer_id": tokenizer_id,
        "run_id": run_id,
        "includes_checkpoints": args.with_checkpoints,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "dataset_id": manifest.get("dataset_id"),
        "tokenizer_fingerprint": manifest.get("tokenizer_fingerprint"),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "file_count": len(members),
        "total_bytes": total_bytes,
        "sha256": checksums,
    }

    out_path = Path(args.out) if args.out else ROOT / f"{model_id}.tar.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {out_path.name} ({human(total_bytes)} uncompressed)...")
    with tarfile.open(out_path, "w:gz") as archive:
        for path, arcname in members:
            archive.add(path, arcname=f"{model_id}/{arcname}")
        payload = json.dumps(transfer_manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name=f"{model_id}/TRANSFER.json")
        info.size = len(payload)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(payload))

    archive_size = out_path.stat().st_size
    line = "-" * 62
    print(f"""
Package ready.
{line}
  Archive:     {out_path}
  Size:        {human(archive_size)}  (from {human(total_bytes)})
  Files:       {len(members)}
  Model:       {model_id}
  Tokenizer:   {tokenizer_id}
{line}

Copy it to your machine, then restore it there:

    python scripts/restore_model.py {out_path.name}
""")


if __name__ == "__main__":
    main()
