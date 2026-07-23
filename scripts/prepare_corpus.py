"""prepare_corpus.py — Download, clean, deduplicate, and import the AION-0.1 corpus.

Sources
-------
1. Project Gutenberg — public domain English prose (fetched via HTTPS).
2. Wikipedia Simple English — clean factual text (fetched via Wikipedia API).

Both sources are public domain / CC-BY-SA.  No login or API key required.

Usage
-----
    python scripts/prepare_corpus.py

The script is idempotent: if the dataset already exists in the project it
appends only documents not already present (by SHA-256 fingerprint).

Output
------
Creates (or appends to) dataset "aion-corpus" in the "aion-01" project.
Prints a summary of documents imported and tokens estimated.

Cleaning pipeline (applied to every document before import)
-----------------------------------------------------------
1. UTF-8 decode with error replacement.
2. Strip Gutenberg header/footer (*** START OF / *** END OF markers).
3. Strip Wikipedia markup artifacts (lines starting with [[Category:, ==References==, etc.).
4. Collapse runs of 3+ blank lines to 2 blank lines.
5. Strip leading/trailing whitespace.
6. Discard documents shorter than 200 characters.
7. Discard documents where non-ASCII ratio > 0.25 (encoding errors / non-English).
8. Exact deduplication by SHA-256 of cleaned text.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

# ── Add repo root to path so aion is importable ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aion.datasets.store import DatasetStore
from aion.workspace.store import ProjectStore

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_NAME = "aion-01"
DATASET_NAME = "aion-corpus"

# Target: a few million training tokens.  At ~4 chars/token average for BPE-8k
# on English prose, ~50k chars/document for Gutenberg books gives ~12.5k
# tokens/book.  A few hundred documents therefore yields several million tokens.
# We cap at TARGET_DOCUMENTS to keep runtime manageable.

TARGET_DOCUMENTS = 300          # total documents across all sources
MIN_DOC_CHARS = 200             # discard shorter documents
MAX_NON_ASCII_RATIO = 0.25      # discard documents with too many non-ASCII chars

# Gutenberg book ids — public domain English prose, varied genres
GUTENBERG_IDS = [
    # Fiction
    1342,   # Pride and Prejudice — Austen
    11,     # Alice's Adventures in Wonderland — Carroll
    1661,   # The Adventures of Sherlock Holmes — Doyle
    98,     # A Tale of Two Cities — Dickens
    1400,   # Great Expectations — Dickens
    2701,   # Moby Dick — Melville
    84,     # Frankenstein — Shelley
    345,    # Dracula — Stoker
    1260,   # Jane Eyre — Brontë
    174,    # The Picture of Dorian Gray — Wilde
    76,     # Adventures of Huckleberry Finn — Twain
    74,     # The Adventures of Tom Sawyer — Twain
    1080,   # A Modest Proposal — Swift
    910,    # Two Years Before the Mast — Dana
    2554,   # Crime and Punishment — Dostoevsky
    2600,   # War and Peace — Tolstoy
    1399,   # Anna Karenina — Tolstoy
    5200,   # Metamorphosis — Kafka
    2814,   # Dubliners — Joyce
    4300,   # Ulysses — Joyce
    # Non-fiction / essays
    16,     # Peter Pan — Barrie
    1232,   # The Prince — Machiavelli
    2680,   # Meditations — Marcus Aurelius
    3207,   # Leviathan — Hobbes
    7370,   # Democracy in America — Tocqueville
    # Science / philosophy
    1228,   # The Origin of Species — Darwin
    5827,   # The Wealth of Nations — Smith
    4363,   # The Federalist Papers
    # More fiction for volume
    514,    # Little Women — Alcott
    1952,   # The Yellow Wallpaper — Gilman
    219,    # Heart of Darkness — Conrad
    35,     # The Time Machine — Wells
    36,     # The War of the Worlds — Wells
    5230,   # The Island of Doctor Moreau — Wells
    768,    # Wuthering Heights — Brontë
    1184,   # The Count of Monte Cristo — Dumas
    2097,   # The Three Musketeers — Dumas
    2500,   # Siddhartha — Hesse
    844,    # The Importance of Being Earnest — Wilde
    1727,   # The Odyssey — Homer (Butler translation)
    6130,   # The Iliad — Homer (Butler translation)
    3296,   # Narrative of the Life of Frederick Douglass
    45,     # Anne of Green Gables — Montgomery
    47,     # Anne of Avonlea — Montgomery
    23,     # Narrative of Sojourner Truth
    1064,   # The Call of the Wild — London
    215,    # The Sea-Wolf — London
    1257,   # The Jungle — Sinclair
    2148,   # Twenty Thousand Leagues Under the Sea — Verne
    103,    # Around the World in Eighty Days — Verne
]

# Wikipedia Simple English article titles — clean, factual, varied
WIKIPEDIA_TITLES = [
    "Science", "Mathematics", "Physics", "Chemistry", "Biology",
    "History", "Geography", "Philosophy", "Literature", "Music",
    "Art", "Technology", "Computer science", "Astronomy", "Medicine",
    "Economics", "Psychology", "Sociology", "Linguistics", "Logic",
    "Democracy", "Government", "Law", "Education", "Religion",
    "Climate", "Ecology", "Evolution", "Genetics", "Neuroscience",
    "Quantum mechanics", "Relativity", "Thermodynamics", "Electromagnetism",
    "Optics", "Acoustics", "Fluid dynamics", "Mechanics", "Statistics",
    "Calculus", "Algebra", "Geometry", "Number theory", "Topology",
    "Algorithm", "Data structure", "Programming language", "Operating system",
    "Internet", "Artificial intelligence", "Machine learning", "Robotics",
    "Ancient Rome", "Ancient Greece", "Ancient Egypt", "Middle Ages",
    "Renaissance", "Industrial Revolution", "World War I", "World War II",
    "Cold War", "French Revolution", "American Revolution", "Roman Empire",
    "British Empire", "Silk Road", "Black Death", "Scientific Revolution",
    "Enlightenment", "Reformation", "Crusades", "Byzantine Empire",
    "Mongol Empire", "Ottoman Empire", "Ming dynasty", "Mughal Empire",
    "Africa", "Asia", "Europe", "North America", "South America",
    "Australia", "Antarctica", "Ocean", "Mountain", "River", "Desert",
    "Forest", "Volcano", "Earthquake", "Hurricane", "Tsunami",
    "Solar System", "Milky Way", "Black hole", "Star", "Planet",
    "Moon", "Sun", "Mars", "Jupiter", "Saturn",
    "Atom", "Molecule", "Cell", "DNA", "Protein",
    "Photosynthesis", "Respiration", "Ecosystem", "Food chain",
    "Human body", "Brain", "Heart", "Immune system", "Nervous system",
    "Language", "Writing", "Alphabet", "Grammar", "Phonetics",
    "Novel", "Poetry", "Drama", "Mythology", "Folklore",
    "Shakespeare", "Aristotle", "Plato", "Socrates", "Kant",
    "Newton", "Einstein", "Darwin", "Curie", "Turing",
    "Piano", "Guitar", "Orchestra", "Opera", "Jazz",
    "Painting", "Sculpture", "Architecture", "Photography", "Film",
    "Trade", "Money", "Bank", "Market", "Inflation",
    "Monarchy", "Republic", "Constitution", "Parliament",
    "United Nations", "European Union", "NATO", "World Trade Organization",
    "Vaccine", "Antibiotic", "Surgery", "Epidemic", "Nutrition",
    "Electricity", "Magnetism", "Nuclear energy", "Solar energy", "Wind energy",
    "Train", "Automobile", "Airplane", "Ship", "Bicycle",
    "Printing press", "Steam engine", "Telephone", "Radio", "Television",
    "Computer", "Smartphone", "Satellite", "Rocket", "Nuclear weapon",
]


# ── Cleaning functions ────────────────────────────────────────────────────────

def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "This etext was prepared",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    # Find start
    start_pos = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip to end of that line
            end_of_line = text.find("\n", idx)
            if end_of_line != -1:
                start_pos = end_of_line + 1
            break
    # Find end
    end_pos = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_pos = idx
            break
    return text[start_pos:end_pos]


def _strip_wikipedia_markup(text: str) -> str:
    """Remove Wikipedia-specific markup artifacts from plain-text extracts."""
    lines = []
    skip_sections = {"references", "see also", "external links", "notes",
                     "further reading", "bibliography"}
    in_skip = False
    for line in text.splitlines():
        stripped = line.strip()
        # Section headers
        if stripped.startswith("==") and stripped.endswith("=="):
            section = stripped.strip("=").strip().lower()
            in_skip = section in skip_sections
            continue
        if in_skip:
            continue
        # Category / file lines
        if stripped.startswith(("[[Category:", "[[File:", "[[Image:", "{{", "}}")):
            continue
        # Bare URLs
        if re.match(r"^https?://\S+$", stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def _clean(text: str, source: str = "gutenberg") -> str | None:
    """Apply the full cleaning pipeline.  Returns None if document is rejected."""
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if source == "gutenberg":
        text = _strip_gutenberg_boilerplate(text)
    elif source == "wikipedia":
        text = _strip_wikipedia_markup(text)

    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Length filter
    if len(text) < MIN_DOC_CHARS:
        return None

    # Non-ASCII ratio filter
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > MAX_NON_ASCII_RATIO:
        return None

    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_url(url: str, timeout: int = 30) -> str | None:
    """Fetch a URL and return the body as a string, or None on error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AION-corpus-builder/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        # Try UTF-8 first, fall back to latin-1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="replace")
    except Exception as exc:
        print(f"  [warn] fetch failed: {url} — {exc}")
        return None


def fetch_gutenberg(book_id: int) -> str | None:
    """Fetch a Project Gutenberg plain-text book."""
    # Try the standard plain-text URL patterns
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]
    for url in urls:
        text = _fetch_url(url)
        if text and len(text) > 1000:
            return text
    return None


def fetch_wikipedia_simple(title: str) -> str | None:
    """Fetch a Simple English Wikipedia article as plain text via the API."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = (
        f"https://simple.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}&prop=extracts&explaintext=1"
        f"&format=json&redirects=1"
    )
    raw = _fetch_url(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "extract" in page and page.get("ns", 0) == 0:
                return page["extract"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Open / create project ─────────────────────────────────────────────────
    store = ProjectStore(ROOT / "workspace")
    try:
        project = store.open(PROJECT_NAME)
        print(f"Opened project: {project.name} ({project.id})")
    except Exception:
        project = store.create(PROJECT_NAME, description="AION-0.1 foundation model project")
        print(f"Created project: {project.name} ({project.id})")

    # ── Open / create dataset ─────────────────────────────────────────────────
    ds_store = DatasetStore(project.data_dir())
    try:
        ds = ds_store.open(DATASET_NAME)
        print(f"Opened dataset: {ds.id} ({ds.meta['document_count']} existing docs)")
    except Exception:
        ds = ds_store.create(DATASET_NAME, description="AION-0.1 training corpus",
                             source="Gutenberg + Simple Wikipedia", language="en")
        print(f"Created dataset: {ds.id}")

    # ── Build seen-hash set from existing documents ───────────────────────────
    seen: set[str] = set()
    print("Hashing existing documents...")
    for _, text in ds.stream():
        seen.add(_sha256(text))
    print(f"  {len(seen)} existing documents hashed.")

    imported = 0
    skipped_clean = 0
    skipped_dup = 0
    skipped_fetch = 0

    # ── Gutenberg ─────────────────────────────────────────────────────────────
    print(f"\nFetching {len(GUTENBERG_IDS)} Gutenberg books...")
    for i, book_id in enumerate(GUTENBERG_IDS):
        if imported >= TARGET_DOCUMENTS:
            break
        print(f"  [{i+1}/{len(GUTENBERG_IDS)}] book {book_id}...", end=" ", flush=True)
        raw = fetch_gutenberg(book_id)
        if not raw:
            print("fetch failed")
            skipped_fetch += 1
            time.sleep(0.5)
            continue
        cleaned = _clean(raw, source="gutenberg")
        if not cleaned:
            print("rejected by cleaner")
            skipped_clean += 1
            time.sleep(0.5)
            continue
        h = _sha256(cleaned)
        if h in seen:
            print("duplicate")
            skipped_dup += 1
            time.sleep(0.5)
            continue
        seen.add(h)
        ds.add_document(cleaned)
        imported += 1
        print(f"ok ({len(cleaned):,} chars)")
        time.sleep(0.5)  # be polite to Gutenberg servers

    # ── Wikipedia Simple English ──────────────────────────────────────────────
    print(f"\nFetching {len(WIKIPEDIA_TITLES)} Wikipedia Simple articles...")
    for i, title in enumerate(WIKIPEDIA_TITLES):
        if imported >= TARGET_DOCUMENTS:
            break
        print(f"  [{i+1}/{len(WIKIPEDIA_TITLES)}] {title!r}...", end=" ", flush=True)
        raw = fetch_wikipedia_simple(title)
        if not raw:
            print("fetch failed")
            skipped_fetch += 1
            time.sleep(0.3)
            continue
        cleaned = _clean(raw, source="wikipedia")
        if not cleaned:
            print("rejected by cleaner")
            skipped_clean += 1
            time.sleep(0.3)
            continue
        h = _sha256(cleaned)
        if h in seen:
            print("duplicate")
            skipped_dup += 1
            time.sleep(0.3)
            continue
        seen.add(h)
        ds.add_document(cleaned)
        imported += 1
        print(f"ok ({len(cleaned):,} chars)")
        time.sleep(0.3)

    # ── Fail early on an empty corpus ─────────────────────────────────────────
    # If every fetch failed (e.g. no network) the dataset would be empty and the
    # downstream tokenizer/training steps would produce garbage or crash later.
    if ds.meta["document_count"] == 0:
        print(
            "\nERROR: corpus is empty — no documents were imported "
            f"(fetch failures: {skipped_fetch}). "
            "Check network access to Gutenberg/Wikipedia and retry."
        )
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_chars = sum(len(text) for _, text in ds.stream())
    # Rough token estimate: BPE-8k on English prose ≈ 4 chars/token
    estimated_tokens = total_chars // 4

    sep = "-" * 41
    print(f"""
Corpus preparation complete.
{sep}
  Project:            {project.id}
  Dataset:            {ds.id}
  Documents imported: {imported}
  Skipped (fetch):    {skipped_fetch}
  Skipped (clean):    {skipped_clean}
  Skipped (dup):      {skipped_dup}
  Total documents:    {ds.meta['document_count']}
  Total characters:   {total_chars:,}
  Estimated tokens:   {estimated_tokens:,}  (at ~4 chars/token)
{sep}
Next step: python scripts/train_tokenizer.py
""")


if __name__ == "__main__":
    main()
