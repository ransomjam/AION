# Tokenization

*Platform input path. Follows the [component-doc template](../../docs/templates/component-doc.md).*

## Role in the platform

Owns the boundary every training run and every inference request crosses: raw
text → normalized text → tokens → integer ids. Two files:

- `text.py` — `normalize`, `strip_accents`, `tokenize`
- `vocabulary.py` — `Vocabulary`, the token↔id map with an out-of-vocabulary and
  special-token policy

**Unlocks:** subword tokenizer training (BPE) reuses `normalize`/`tokenize` as
its pre-tokenization step; dataset preparation and every model's embedding table
are defined against a `Vocabulary`. Getting this contract stable now avoids
re-indexing corpora and retraining embeddings later.

## Contract / API

```python
normalize(text, *, form="NFKC", casefold=True, accents=True) -> str
strip_accents(text) -> str
tokenize(text, *, normalize_first=True) -> list[str]

Vocabulary.build(corpus, *, min_freq=1, max_size=None, specials=DEFAULT_SPECIALS) -> Vocabulary
vocab.encode(text, *, add_bos=False, add_eos=False) -> list[int]
vocab.decode(ids, *, skip_specials=True) -> list[str]
vocab.token_to_id(tok) -> int        # <unk> id for unknown tokens
vocab.to_json()/from_json(), save()/load()
```

Specials occupy fixed low ids: `0:<pad> 1:<unk> 2:<bos> 3:<eos>`, so a model's
embedding table has stable reserved slots independent of the corpus.

## Invariants

- **Determinism.** `Vocabulary.build` orders tokens by `(-frequency, token)`;
  same corpus + options ⇒ byte-identical vocabulary. (`test_ordering_is_deterministic`)
- **One tokenizer, shared.** `Vocabulary` tokenizes via the same `tokenize`, so
  build-time and encode-time splitting can never diverge — the classic silent
  bug this structurally prevents.
- **`<unk>` is mandatory.** The constructor rejects a vocabulary that cannot
  encode unseen input. (`test_vocabulary_without_unk_is_rejected`)
- **Immutable after build.** A shifting id map corrupts every id a model has
  already learned; treat instances as read-only.
- **`normalize` is idempotent** under fixed options.

## Design notes

- **NFKC + casefold by default.** Collapses cosmetic Unicode variants (ligatures,
  full-width forms) and folds case Unicode-correctly (ß→ss). Accent stripping is
  a lossy, English-biased convenience — on by default, disable with
  `accents=False` when diacritics carry meaning.
- **Rule-based tokenizer by design.** Alphanumeric runs form word tokens; each
  punctuation/symbol is its own token; whitespace is discarded. It is the honest
  baseline that the learned subword tokenizer will replace — kept minimal so BPE
  can sit on top cleanly.
- **JSON stores the ordered token list** (index = id), not a dict, so ids stay
  explicit and version diffs stay readable.

## Performance & scaling

For a corpus of `N` total tokens, `V` distinct: `build` is `O(N)` to count +
`O(V log V)` to sort; `encode`/`decode` are `O(len(input))` hash lookups;
`normalize`/`tokenize` are `O(len(text))`. Pure Python and single-pass — fine at
current scale; the count/sort step is the first thing to revisit (streaming
counts, bounded heap) when corpora outgrow memory.

## Benchmarks

None yet — no corpus at this scale in the repo. Throughput numbers get added with
the first real dataset, alongside the reproducing command.

## Tests

`tests/test_text.py`, `tests/test_vocabulary.py` — normalization/idempotence,
NFKC/casefold/accents, tokenizer edge cases (Unicode scripts, punctuation, empty
input), `min_freq`/`max_size`, deterministic ordering, `<unk>`/`<bos>`/`<eos>`,
JSON round-trip. Run from the repo root:

```
python -m unittest discover -s aion -p "test_*.py" -v
```

## References

- Unicode Standard Annex #15, *Normalization Forms*.
- Sennrich, Haddow & Birch (2016), *Neural Machine Translation of Rare Words with
  Subword Units* — the subword tokenizer this component precedes.
