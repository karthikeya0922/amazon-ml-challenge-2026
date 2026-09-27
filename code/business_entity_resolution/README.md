# Business Entity Resolution — reproduction guide

Everything is derived from the provided files only: no internet access, external APIs,
external datasets, geocoding, hand-written dictionaries or pretrained model weights.

## Layout expected

```
<project root>/
├── Dataset/student_resource/dataset/{train,test}/*.tsv   # provided data (unchanged)
├── code/business_entity_resolution/src/                 # this code
├── work/                                                # intermediate parquet files (created)
└── output/                                              # matching_results.tsv, candidate_pairs.tsv
```

## Environment

Python 3.11. `pip install -r requirements.txt`

## Run end-to-end (from `src/`)

One command: `python run_all.py` (resume with `--from <step>`). Live progress: `python progress.py train|test`.
Step by step:

```bash
python prep.py                          # 1. normalize all source files -> work/*_sourceN.parquet
python mine_tokens.py                   # 2. mine per-country vocabulary maps from matched train pairs
python block_ranker.py                  # 2b. learn how to rank the blocking union (non-validation queries)
CAND_ONLY=1 python pipeline.py train    # 3. forward + reverse blocking and competition context for train
python prune.py                         # 3b. learned candidate pruner (last blocking stage, ~5 candidates per S1)
REUSE_CAND=1 python pipeline.py train   # 3c. features of the pruned train candidates (every train S1)
python train.py                         # 4. stage-1 LightGBM, two folds, out-of-fold p1 for every train pair
python stage2.py build train            # 5. stage-2 features (competition + cluster evidence) from p1
python stage2.py train                  # 5b. stage-2 LightGBM, two folds; picks the decision rule on validation
python pipeline.py test                 # 6. blocking + pruning + features for every test Source-1 record
python predict.py stage1                # 7. stage-1 probabilities for test
python stage2.py build test             # 7b. stage-2 features for test
python predict.py final                 # 7c. first test prediction
python mine_pseudo.py                   # 8. maps for countries without labels, from confident predictions
python pipeline.py test unseen          # 8b. rebuild those countries with their maps (+ steps 7-7c again)
python adapt.py                         # 9. self-training of stage 1 for those countries (+ steps 7b-7c again);
                                        #    run twice (the second round uses the first round's predictions)
python make_package.py --team "<team_name>"   # 10. build <team_name>_submission.zip
```

(On Windows PowerShell set the environment switches with `$env:CAND_ONLY=1` etc.; `run_all.py` sets them itself.)

Validate the output format:

```bash
python ../../../Dataset/student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../Dataset/student_resource/dataset/test
```

## Pipeline summary

| Step | Module | What it does |
|---|---|---|
| Normalize | `normalize.py`, `prep.py` | Unicode/accents (Latin only), case, punctuation, null markers, web-domain names, dotted initialisms, zero-padded numbers, repeated words |
| Vocabulary | `mine_tokens.py` | Learns variant→canonical maps (abbreviations, state codes) from token differences in true pairs, per country, plus a word-by-word native-script→Latin dictionary from positionally aligned true pairs |
| Blocking | `blocking.py`, `block_ranker.py` | Per-country inverted index over 7 rare-key families (words, name-word pairs, number×street-word, name-word×number, name-word×address-word, glued names, address-word pairs); IDF-weighted; union ranked by a learned blocking ranker, top-35 per Source-1 record, plus deeper ranks 36–120 with a high cheap pair score (≤ 20 per record), plus reverse blocking (best Source-1 record per S2/S3 record) |
| Pruning | `prune.py` | Learned cheap filter on blocking-time signals (key scores, reverse rank, one fuzzy name+address score, competition ranks) → ~5–7 candidates per Source-1 record = `candidate_pairs.tsv` |
| Features | `features.py` | String similarities (Levenshtein, Jaro-Winkler, token set/sort, partial), Jaccard, IDF-weighted coverage, number agreement, competition context |
| Stage 1 | `train.py` | LightGBM binary classifier trained from scratch, two out-of-fold models |
| Stage 2 | `stage2.py` | LightGBM on stage-1 features + p1 + ownership competition + cluster evidence (similarity to the Source-1 record's confident matches vs rejected candidates), two out-of-fold models |
| Decision | `decide.py`, `predict.py` | One owner per S2/S3 record; stage 2 may not lift a pair stage 1 rejected; uncontested records need p1 >= 0.8; threshold 0.7 (0.995 for countries without training labels, set from the leaderboard) |
| Unseen country | `mine_pseudo.py`, `adapt.py` | Maps mined from confident test predictions; self-training of stage 1 on confident test predictions |

## Exact reproduction

`reference/token_maps.json` contains the vocabulary maps used for the submitted results. The miner can
break ties between equally frequent rules differently from run to run; to reproduce the submitted files
exactly, copy it to `work/token_maps.json` after step 8 (`mine_pseudo.py`) and continue from step 8b.
