# Amazon ML Challenge 2026 — Business Entity Resolution

**Team Future bytes** · Karthikeya Gupta, Sreekar Kumar

Match every business in a reference list (Source 1, 1.7M records) to its noisy duplicates in two other
sources (~10M records), across India, the US and an **unseen country (France)** that never appears in
the training data. Metric: macro-averaged F0.5 per Source-1 business (precision-heavy).

| | Macro F0.5 |
|---|---|
| Validation (India + US, held-out 10% of training businesses) | **0.9879** |
| Leaderboard (test set incl. France) | **0.979130** |

The whole solution is **data-derived**: no external data, no dictionaries, no geocoding, no pretrained
models — every vocabulary mapping is mined from the provided training pairs. It runs on a laptop CPU
(12 threads, 12 GB RAM).

## How it works

```
Source 1 ──┐
Source 2 ──┼─► normalise ─► mined vocabulary maps ─► blocking ─► pruning ─► stage 1 ─► stage 2 ─► decision
Source 3 ──┘   (text)       (abbrev., states,       (hashed     (learned   (LightGBM  (LightGBM   (one owner,
                            native scripts)          keys)       filter)    pairs)     context)   thresholds)
```

1. **Normalisation + mined vocabulary.** Abbreviations (`rd → road`), state codes, legal forms and a
   native-script → Latin dictionary are learned from matched training pairs. Native-script names
   (Devanagari, Tamil, Bengali, …) are word-by-word renderings of the Latin name, so aligning words by
   position gives an exact dictionary.
2. **Blocking.** Seven hashed key families (rare words, name-word pairs, house number × street word,
   glued names, …) in a per-country inverted index; a learned ranker keeps the top 35 per business,
   plus *deep* candidates (ranks 36–120 that look alike) and *reverse* candidates (the best business for
   each Source-2/3 record). Candidate recall ≈ 98.1% at ~5–7 candidates per business after pruning.
3. **Stage 1** — LightGBM on ~70 language-agnostic features (string similarities, IDF-weighted overlaps,
   house-number agreement, competition between candidates), trained out-of-fold.
4. **Stage 2** — LightGBM that adds collective evidence from all stage-1 probabilities: who else claims
   this record, how many confident matches the business already has, and whether the candidate resembles
   the business's other records or its rejected look-alikes.
5. **Decision.** Each Source-2/3 record goes to at most one business; thresholds per country.

## What mattered most (lessons)

- **The test set was not like validation.** It contains many more records of businesses that are *not*
  in Source 1 — look-alike decoys at neighbouring addresses. The model had learned "a record nobody else
  wants is probably a match", which is true on training data and false on test. Two rules fixed it:
  stage 2 may not *raise* a pair stage 1 rejected, and uncontested records need a confident stage-1
  probability (leaderboard 0.970 → 0.977).
- **Unseen country = over-confident model.** France's "fairly confident" matches were mostly wrong; a
  strict threshold (0.995) found by comparing submissions that differed *only* in France's threshold
  added +0.0006. Self-training on confident French predictions helped too.
- **Recall lives in blocking.** Half of the remaining validation loss was true pairs that never became
  candidates; deep and reverse candidates recovered a large part of it.
- **Mined maps can be subtly wrong.** Co-occurrence mining mapped the Hindi word for *private* to
  *limited* (they always appear together); positional alignment fixed it (+0.0009 India).

| Version | Validation | Leaderboard |
|---|---|---|
| Single LightGBM, forward blocking | 0.9822 | 0.973 |
| + reverse blocking, two-stage model, decoy rules | 0.9856 | 0.977 |
| + deep candidates | 0.9873 | 0.977619 |
| + France threshold 0.995 | — | 0.978259 |
| + native-script alignment, all training data | **0.9879** | **0.979130** |

The full write-up — EDA, every experiment (including the ones that failed) and error analysis — is in
[Documentation_template.md](Documentation_template.md).

## Repository layout

```
code/business_entity_resolution/
├── src/            pipeline (entry point: run_all.py)
├── reference/      exact vocabulary maps used for the final submission
├── README.md       step-by-step reproduction guide
└── requirements.txt
Documentation_template.md   methodology document submitted to the challenge
```

## Running it

The competition data is not included (it belongs to the challenge organisers). Place it as
`Dataset/student_resource/dataset/{train,test}/*.tsv`, then:

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python run_all.py            # full pipeline -> output/matching_results.tsv + output/candidate_pairs.tsv
```

A full run takes several hours on a 12-thread CPU; see
[code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) for the individual
steps and exact reproduction.

Dependencies: polars, numpy, rapidfuzz, lightgbm (MIT/BSD licensed).
