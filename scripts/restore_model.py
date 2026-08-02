"""restore_model.py — install a model trained elsewhere into this workspace.

Run this locally on the archive produced by ``package_model.py``.  It verifies
every file against the checksums recorded at packaging time, installs the model
and its tokenizer into the local project, and then generates text to prove the
result is actually usable.

Usage
-----
    python scripts/restore_model.py AION-Cyber-0.1-a1b2c3d4.tar.gz
    python scripts/restore_model.py <archive> --verify-only
    python scripts/restore_model.py <archive> --force     # overwrite an existing id

Why it generates before declaring success
-----------------------------------------
A checksum proves the bytes survived the journey.  It does not prove the model
loads, that the tokenizer matches the vocabulary the weights were trained
against, or that the architecture on disk reconstructs.  Those are the failures
that actually happen when a model moves between machines, and they are silent:
a mismatched tokenizer produces fluent-looking nonsense rather than an error.
Generating a few tokens is the cheapest check that catches all three, so it is
not optional here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion.gpt.store import GPTStore
from aion.tokenizers.store import TokenizerStore
from aion.training.sampler import SampleGenerator
from aion.workspace.store import ProjectStore

PROJECT_NAME = "aion-01"

VERIFY_PROMPTS = [
    "The history of",
    "Vulnerability report: CVE-",
    "Message analysis\nChannel: SMS\nSender: MTN MoMo Alert\nMessage:\n",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract, refusing any member that would escape ``destination``.

    An archive is untrusted input even when we produced it: a path like
    ``../../.ssh/authorized_keys`` inside a tarball is a real and old attack,
    and this script's whole purpose is to unpack something that arrived from a
    machine we do not control.
    """
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise ValueError(f"archive member escapes the destination: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"archive contains a link, which is not expected: {member.name!r}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a model packaged on another machine.")
    parser.add_argument("archive", help="the .tar.gz produced by package_model.py")
    parser.add_argument("--verify-only", action="store_true",
                        help="check the archive and report, installing nothing")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing model or tokenizer of the same id")
    parser.add_argument("--no-generate", action="store_true",
                        help="skip the generation check (not recommended)")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.is_file():
        print(f"ERROR: no such archive: {archive_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="aion_restore_") as tmpdir:
        staging = Path(tmpdir)
        print(f"Extracting {archive_path.name}...")
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                safe_extract(archive, staging)
        except (tarfile.TarError, ValueError) as exc:
            print(f"ERROR: could not extract the archive: {exc}")
            sys.exit(1)

        roots = [p for p in staging.iterdir() if p.is_dir()]
        if len(roots) != 1:
            print(f"ERROR: expected one top-level directory in the archive, found {len(roots)}.")
            sys.exit(1)
        payload = roots[0]

        manifest_path = payload / "TRANSFER.json"
        if not manifest_path.is_file():
            print("ERROR: TRANSFER.json is missing. This archive was not produced by "
                  "package_model.py.")
            sys.exit(1)
        transfer = json.loads(manifest_path.read_text(encoding="utf-8"))

        model_id = transfer["model_id"]
        tokenizer_id = transfer.get("tokenizer_id")
        line = "-" * 62
        print(f"""{line}
  Model:       {model_id}
  Name:        {transfer.get('model_name')}
  Parameters:  {transfer.get('param_count', 0):,}
  Tokenizer:   {tokenizer_id}
  Packaged:    {transfer.get('packaged_at')}
  Files:       {transfer.get('file_count')}
{line}""")

        # ── verify ────────────────────────────────────────────────────────────
        print("\nVerifying checksums...")
        mismatched, missing = [], []
        for arcname, expected in transfer["sha256"].items():
            candidate = payload / arcname
            if not candidate.is_file():
                missing.append(arcname)
            elif sha256_file(candidate) != expected:
                mismatched.append(arcname)

        if missing or mismatched:
            for name in missing:
                print(f"  MISSING   {name}")
            for name in mismatched:
                print(f"  CORRUPT   {name}")
            print(f"\nFAIL: {len(missing)} missing, {len(mismatched)} corrupt. "
                  f"The transfer did not arrive intact; copy the archive again.")
            sys.exit(1)
        print(f"  all {len(transfer['sha256'])} files match.")

        if args.verify_only:
            print("\n--verify-only: nothing was installed.")
            return

        # ── install ───────────────────────────────────────────────────────────
        store = ProjectStore(ROOT / "workspace")
        try:
            project = store.open(PROJECT_NAME)
        except Exception:
            project = store.create(PROJECT_NAME,
                                   description="AION-0.1 foundation model project")
        project_dir = Path(project.dir("models")).parent

        print("\nInstalling...")
        for relative in ("models", "tokenizers", "logs"):
            source_root = payload / relative
            if not source_root.is_dir():
                continue
            for source in source_root.iterdir():
                target = project_dir / relative / source.name
                if target.exists():
                    if not args.force:
                        print(f"  SKIP  {relative}/{source.name} already exists "
                              f"(pass --force to replace it)")
                        continue
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
                print(f"  ok    {relative}/{source.name}")

    # ── prove it works ────────────────────────────────────────────────────────
    if args.no_generate:
        print("\nInstalled. Generation check skipped.")
        return

    print("\nLoading the model and generating...")
    gpt_store = GPTStore(project.dir("models"))
    if not gpt_store.exists(model_id):
        print(f"ERROR: {model_id} is not readable after installation.")
        sys.exit(1)

    try:
        model = gpt_store.load(model_id)
    except Exception as exc:
        print(f"FAIL: the model did not reconstruct: {exc}")
        sys.exit(1)

    tokenizer_store = TokenizerStore(project.dir("tokenizers"))
    if not tokenizer_id:
        print("WARNING: no tokenizer id recorded; cannot run the generation check.")
        return
    try:
        tokenizer = tokenizer_store.load(tokenizer_id)
    except Exception as exc:
        print(f"FAIL: the tokenizer did not load: {exc}")
        sys.exit(1)

    if tokenizer.vocab_size != model.cfg.vocab_size:
        print(f"FAIL: vocabulary mismatch — tokenizer has {tokenizer.vocab_size:,} "
              f"tokens, the model expects {model.cfg.vocab_size:,}.\n"
              f"      This model cannot be used with this tokenizer.")
        sys.exit(1)

    # Generation appends to the context, so prompt + new tokens must fit inside
    # the model's window.  Budget for it rather than discovering the overflow as
    # an IndexError partway through the check.
    max_new = min(48, max(8, model.cfg.max_seq_len // 4))
    generator = SampleGenerator(model, tokenizer, max_new_tokens=max_new,
                                strategy="greedy")
    line = "-" * 62
    print(line)
    failures = 0
    for prompt in VERIFY_PROMPTS:
        budget = model.cfg.max_seq_len - max_new
        ids = tokenizer.encode(prompt)
        if len(ids) > budget:
            prompt = tokenizer.decode(ids[-budget:])
        try:
            result = generator.generate(prompt)
        except Exception as exc:
            failures += 1
            print(f"  prompt: {prompt.replace(chr(10), ' / ')}")
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print(line)
            continue
        print(f"  prompt: {prompt.replace(chr(10), ' / ')}")
        print(f"  output: {result.generated_text.strip()[:300]}")
        print(line)

    if failures:
        print(f"\nFAIL: {failures} of {len(VERIFY_PROMPTS)} prompts did not generate. "
              f"The model is installed but not usable.")
        sys.exit(1)

    print(f"""
Restored and verified.
  Model:      {model_id}
  Location:   workspace/projects/{PROJECT_NAME}/models/{model_id}/
  Parameters: {model.param_count():,}

Evaluate it:

    python scripts/evaluate_aion01.py {model_id}
""")


if __name__ == "__main__":
    main()
