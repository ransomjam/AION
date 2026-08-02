"""build_cyber_corpus.py — assemble the AION-Cyber corpus.

Fetches public cybersecurity sources, generates structured threat-reasoning
examples, cleans and deduplicates everything, and imports the result as the
``aion-cyber`` dataset in the ``aion-01`` project.  Writes a manifest recording
where every document came from and under what licence.

Usage
-----
    python scripts/build_cyber_corpus.py                  # full build
    python scripts/build_cyber_corpus.py --max-cves 20000 # smaller, faster
    python scripts/build_cyber_corpus.py --offline        # generated text only
    python scripts/build_cyber_corpus.py --report-only    # inspect, import nothing

The build is idempotent: documents already present (by SHA-256 of the cleaned
text) are skipped, so an interrupted run is resumed by running it again.

Why a separate dataset
----------------------
AION-Cyber is imported as its own dataset rather than appended to
``aion-corpus``.  ``TrainingConfig.dataset_ids`` takes a list, so a training run
composes them at corpus-build time, and keeping them separate means the general
and cybersecurity halves can be re-fetched, re-weighted, or dropped
independently.  Merging them would make the 70/30 balance a one-way door.

The domain balance
------------------
The target is roughly 70% general language and 30% cybersecurity.  That ratio
is not arbitrary: a model trained purely on advisories writes like an advisory
and loses the fluency needed to *explain* a threat to someone who is not a
security professional, which is the whole point of putting it in front of a
user.  The existing ``aion-corpus`` supplies the general half.  This script
reports the ratio it actually achieved rather than assuming it, and
``--report-only`` prints it before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aion.datasets.store import DatasetStore
from aion.workspace.store import ProjectStore

import cyber_reasoning
import cyber_sources
from cyber_sources import CyberDocument, SourceResult

PROJECT_NAME = "aion-01"
GENERAL_DATASET = "aion-corpus"
CYBER_DATASET = "aion-cyber"

MIN_DOC_CHARS = 120          # below this a document carries no reasoning
MAX_DOC_CHARS = 60_000       # a handful of cheat sheets are book-length
DEFAULT_MAX_CVES = 60_000
DEFAULT_REASONING = 12_000
TARGET_CYBER_FRACTION = 0.30


# ── cleaning ──────────────────────────────────────────────────────────────────

_WHITESPACE_RUNS = re.compile(r"[ \t]{2,}")
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean(text: str) -> str | None:
    """Normalise a document, or reject it by returning ``None``.

    Deliberately conservative.  URLs, brand names, and amounts are *preserved*:
    they are the indicators a threat model has to learn to read, and stripping
    them for tidiness would remove the signal along with the noise.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = _WHITESPACE_RUNS.sub(" ", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()

    if len(text) < MIN_DOC_CHARS:
        return None
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS].rsplit("\n", 1)[0]
    return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, width: int = 9) -> set[str]:
    """Word shingles used for near-duplicate detection."""
    words = text.lower().split()
    if len(words) < width:
        return {" ".join(words)}
    return {" ".join(words[i:i + width]) for i in range(0, len(words) - width + 1, 3)}


class Deduplicator:
    """Exact and near-duplicate rejection.

    Exact matching alone is not enough for fetched material.  Thousands of CVE
    entries differ only in a version number, and a corpus that repeats
    near-identical text teaches the model to memorise it — visible as a
    training loss that falls much further than the model's actual understanding
    justifies.  Near-dup detection is the cheapest guard against that.

    It is applied only to genuine documents.  See ``import_documents`` for why
    the generated reasoning examples are exempt.
    """

    def __init__(self, near_threshold: float = 0.8) -> None:
        self.hashes: set[str] = set()
        self.near_threshold = near_threshold
        self._buckets: dict[str, list[set[str]]] = {}
        self.exact_rejected = 0
        self.near_rejected = 0

    def seen_exact(self, text: str) -> bool:
        digest = sha256(text)
        if digest in self.hashes:
            self.exact_rejected += 1
            return True
        self.hashes.add(digest)
        return False

    def seen_near(self, text: str, bucket: str) -> bool:
        """Near-duplicate check within one source bucket.

        Bucketing by source keeps the comparison set small; a CVE is never a
        near-duplicate of an OWASP cheat sheet, so comparing them would only
        cost time.
        """
        current = shingles(text)
        if not current:
            return False
        existing = self._buckets.setdefault(bucket, [])
        for other in existing[-400:]:      # bounded window: recent neighbours
            overlap = len(current & other)
            if overlap / max(1, min(len(current), len(other))) >= self.near_threshold:
                self.near_rejected += 1
                return True
        existing.append(current)
        return False


# ── build ─────────────────────────────────────────────────────────────────────

def collect_genuine(max_cves: int, include_cheatsheets: bool) -> list[SourceResult]:
    results: list[SourceResult] = []
    for make_source in cyber_sources.all_sources(max_cves, include_cheatsheets):
        started = time.monotonic()
        result = make_source()
        elapsed = time.monotonic() - started
        status = "ok" if result.ok else "NO DOCUMENTS"
        print(f"  {result.name:<14} {len(result.documents):>7,} documents  "
              f"{elapsed:>6.1f}s  {status}")
        for error in result.errors:
            print(f"    [error] {error}")
        results.append(result)
    return results


def collect_generated(count: int, seed: int) -> SourceResult:
    """Structured reasoning examples, all labelled synthetic."""
    result = SourceResult(name="reasoning")
    examples = list(cyber_reasoning.curated_examples(seed=seed))
    examples += list(cyber_reasoning.generate(count, seed=seed))
    for example in examples:
        result.documents.append(CyberDocument(
            text=example.render(),
            source=("Camtinel curated scenario" if example.seed_of
                    else "AION reasoning generator"),
            licence="Own work",
            url=example.seed_of or example.family,
            category="threat-reasoning",
            provenance="synthetic",
        ))
    print(f"  {result.name:<14} {len(result.documents):>7,} documents  "
          f"(curated seeds + generated variations)")
    return result


def import_documents(dataset, documents: list[CyberDocument], dedup: Deduplicator
                     ) -> tuple[int, list[dict]]:
    """Clean, deduplicate, and append.  Returns (imported, manifest entries)."""
    imported = 0
    manifest: list[dict] = []
    rejected_short = 0
    for document in documents:
        text = clean(document.text)
        if text is None:
            rejected_short += 1
            continue
        if dedup.seen_exact(text):
            continue
        # Near-duplicate rejection applies to fetched reference material, where
        # two CVE entries differing only by a version number are genuine
        # redundancy.  It must NOT apply to the generated reasoning examples:
        # they deliberately repeat the analytical frame while varying the brand,
        # amount, channel, and language, and that repetition-with-variation is
        # exactly the invariance the model is meant to extract.  Filtering it
        # out would discard the curriculum and keep only the noise.
        if document.provenance == "genuine" and dedup.seen_near(text, document.category):
            continue
        doc_id = dataset.add_document(text)
        imported += 1
        manifest.append({
            "document_id": doc_id,
            "source": document.source,
            "licence": document.licence,
            "url": document.url,
            "category": document.category,
            "provenance": document.provenance,
            "characters": len(text),
        })
    if rejected_short:
        print(f"    ({rejected_short:,} rejected as too short to carry reasoning)")
    return imported, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AION-Cyber corpus.")
    parser.add_argument("--max-cves", type=int, default=DEFAULT_MAX_CVES,
                        help=f"CVE records to fetch from NVD (default {DEFAULT_MAX_CVES:,}). "
                             "NVD allows about 2,000 per request with a 6s pause between them.")
    parser.add_argument("--reasoning", type=int, default=DEFAULT_REASONING,
                        help=f"generated reasoning examples (default {DEFAULT_REASONING:,})")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for generation; the same seed rebuilds the same examples")
    parser.add_argument("--offline", action="store_true",
                        help="skip all network sources; generate reasoning examples only")
    parser.add_argument("--no-cheatsheets", action="store_true",
                        help="skip the OWASP Cheat Sheet Series (120 files, slower)")
    parser.add_argument("--report-only", action="store_true",
                        help="report what would be built and import nothing")
    args = parser.parse_args()

    store = ProjectStore(ROOT / "workspace")
    try:
        project = store.open(PROJECT_NAME)
    except Exception:
        project = store.create(PROJECT_NAME, description="AION-0.1 foundation model project")
    print(f"Project: {project.name} ({project.id})\n")

    # ── collect ───────────────────────────────────────────────────────────────
    print("Collecting sources")
    results: list[SourceResult] = []
    if args.offline:
        print("  (offline: network sources skipped)")
    else:
        results += collect_genuine(args.max_cves, not args.no_cheatsheets)
    results.append(collect_generated(args.reasoning, args.seed))

    documents = [d for result in results for d in result.documents]
    if not documents:
        print("\nERROR: no documents collected. Nothing to import.")
        sys.exit(1)

    genuine = sum(1 for d in documents if d.provenance == "genuine")
    synthetic = len(documents) - genuine
    print(f"\n  collected {len(documents):,} documents "
          f"({genuine:,} genuine, {synthetic:,} synthetic)")

    if args.report_only:
        by_category: dict[str, int] = {}
        for document in documents:
            by_category[document.category] = by_category.get(document.category, 0) + 1
        print("\nWould import, by category:")
        for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
            print(f"  {category:<22} {count:>8,}")
        print("\n--report-only: nothing was written.")
        return

    # ── import ────────────────────────────────────────────────────────────────
    ds_store = DatasetStore(project.data_dir())
    try:
        dataset = ds_store.open(CYBER_DATASET)
        print(f"\nDataset: {dataset.id} ({dataset.meta['document_count']:,} existing documents)")
    except Exception:
        dataset = ds_store.create(
            CYBER_DATASET,
            description="AION-Cyber: public cybersecurity corpus plus structured threat reasoning",
            source="NVD, CISA KEV, MITRE ATT&CK, OWASP, AION reasoning generator",
            language="en+fr",
        )
        print(f"\nDataset: {dataset.id} (created)")

    dedup = Deduplicator()
    print("Hashing existing documents...")
    for _, text in dataset.stream():
        dedup.seen_exact(text)
    print(f"  {len(dedup.hashes):,} already present")

    print("\nImporting...")
    imported, manifest = import_documents(dataset, documents, dedup)

    # ── manifest ──────────────────────────────────────────────────────────────
    # Written next to the dataset, not inside it: dataset.json holds counters
    # only, and provenance is a separate, appendable record.
    manifest_path = Path(dataset.root) / "provenance.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── report ────────────────────────────────────────────────────────────────
    cyber_chars = sum(len(text) for _, text in dataset.stream())
    try:
        general = ds_store.open(GENERAL_DATASET)
        general_chars = sum(len(text) for _, text in general.stream())
        general_docs = general.meta["document_count"]
    except Exception:
        general_chars, general_docs = 0, 0

    total_chars = cyber_chars + general_chars
    cyber_fraction = cyber_chars / total_chars if total_chars else 1.0

    licences: dict[str, int] = {}
    for entry in manifest:
        licences[entry["licence"]] = licences.get(entry["licence"], 0) + 1

    line = "-" * 62
    print(f"""
Corpus build complete.
{line}
  Imported this run:   {imported:,} documents
  Exact duplicates:    {dedup.exact_rejected:,} skipped
  Near duplicates:     {dedup.near_rejected:,} skipped
  aion-cyber total:    {dataset.meta['document_count']:,} documents, {cyber_chars:,} characters
  aion-corpus total:   {general_docs:,} documents, {general_chars:,} characters
{line}
  Domain balance (by characters)
    general:           {(1 - cyber_fraction) * 100:5.1f}%   (target {(1 - TARGET_CYBER_FRACTION) * 100:.0f}%)
    cybersecurity:     {cyber_fraction * 100:5.1f}%   (target {TARGET_CYBER_FRACTION * 100:.0f}%)
{line}
  Licences represented""")
    for licence, count in sorted(licences.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>7,}  {licence}")
    print(f"""{line}
  Provenance manifest: {manifest_path}
{line}""")

    if cyber_fraction < TARGET_CYBER_FRACTION - 0.05:
        needed = int((TARGET_CYBER_FRACTION * general_chars
                      / (1 - TARGET_CYBER_FRACTION)) - cyber_chars)
        print(f"NOTE: the cybersecurity share is below target by roughly "
              f"{needed:,} characters.\n"
              f"      Raise --max-cves or --reasoning and run again; the build "
              f"is idempotent.\n")
    elif cyber_fraction > TARGET_CYBER_FRACTION + 0.05:
        print("NOTE: the cybersecurity share is above target. The general corpus "
              "carries fluency;\n      consider extending aion-corpus rather than "
              "trimming this one.\n")

    print("Next step: python scripts/train_tokenizer.py --retrain\n"
          "  The vocabulary changes materially with this corpus (CVE ids, domain\n"
          "  names, FCFA amounts), so the tokenizer must be retrained before training.\n")


if __name__ == "__main__":
    main()
