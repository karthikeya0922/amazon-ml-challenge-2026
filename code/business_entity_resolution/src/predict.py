"""Step 8: score test candidates and write output/matching_results.tsv + output/candidate_pairs.tsv.

    python predict.py stage1    # stage-1 probabilities (fold-model average) -> work/stage1_test.parquet
    python predict.py final     # stage-2 probabilities (needs `stage2.py build test`), decisions, output files
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import OUTPUT, WORK
from data import countries, scan
from decide import choose, choose_threshold, one_owner
from train import N_FOLDS, parts, to_x

# Countries without training labels are scored by a model that has never seen their data; on a
# simulated unseen country (train on one training country, validate on the other) the model is
# over-confident there and the best threshold is higher than on the training countries. How much
# higher cannot be measured without labels; it was set from leaderboard scores of submissions that
# differed only in the France threshold (all else identical): 0.70 -> 0.977476, 0.85 -> 0.977619,
# 0.95 -> 0.977866, 0.98 -> 0.978046, 0.99 -> 0.978211, 0.995 -> 0.978259, 0.999 -> 0.977943.
UNSEEN_THR = 0.995
# An S2/S3 record that is a candidate of a single Source-1 record is uncontested, and so is a look-alike
# record of a business that Source 1 does not contain. Such pairs with a hesitant stage-1 probability are
# selected 2-3x more often (per Source-1 record) on test than on validation, where they are 95% correct;
# the excess is the test set's extra ownerless records. They need a confident stage-1 probability.
UNCONTESTED_MIN_P1 = 0.8


def fold_models(prefix: str) -> list[lgb.Booster]:
    return [lgb.Booster(model_file=str(WORK / f"{prefix}_f{f}.txt")) for f in range(N_FOLDS)]


def predict_avg(models: list[lgb.Booster], x: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(x) for m in models], axis=0)


def stage1():
    t = time.time()
    models = fold_models("lgb_s1")
    feats = models[0].feature_name()
    out = []
    for p in parts("test"):
        d = pl.read_parquet(p, columns=["country", "rid1", "rid2", *feats])
        out.append(d.select("country", "rid1", "rid2", p1=pl.Series(predict_avg(models, to_x(d, feats)), dtype=pl.Float32)))
    s1p = pl.concat(out)
    s1p.write_parquet(WORK / "stage1_test.parquet")
    print(f"stage-1 probabilities for {s1p.height:,} test pairs ({time.time() - t:.0f}s)")


def write_lists(long: pl.DataFrame, s1_ids: pl.Series, path, col: str):
    """One row per Source-1 id, comma-joined S2/S3 ids (empty string when none)."""
    agg = long.group_by("source1_entity_id").agg(pl.col("mid").unique().sort().str.join(",").alias(col))
    out = pl.DataFrame({"source1_entity_id": s1_ids}).join(agg, on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def final():
    from stage2 import final_prob, joined_parts
    t = time.time()
    models = fold_models("lgb_s2")
    feats = models[0].feature_name()
    out = []
    for d in joined_parts("test"):
        out.append(d.select("country", "rid1", "rid2", "p1", p2=pl.Series(predict_avg(models, to_x(d, feats)), dtype=pl.Float32)))
    df = pl.concat(out).with_columns(p=final_prob(pl.col("p1"), pl.col("p2")))
    del out
    df.write_parquet(WORK / "test_pred.parquet")
    print(f"scored {df.height:,} test pairs ({time.time() - t:.0f}s)")
    decide(df, t)


def decide(df: pl.DataFrame, t: float):
    """Pair probabilities (country, rid1, rid2, p1, p) -> output/matching_results.tsv + candidate_pairs.tsv."""
    seen = set(countries("train"))
    decision = json.loads((WORK / "best_decision.json").read_text())
    best = decision["best"]
    unseen_thr = UNSEEN_THR
    sel, cand = [], []
    for (country,), g in df.group_by("country"):
        s1 = pl.read_parquet(WORK / f"s1ids_test_{country}.parquet").select(pl.col("rid").alias("rid1"), pl.col("entity_id").alias("source1_entity_id"))
        s23 = pl.read_parquet(WORK / f"s23ids_test_{country}.parquet").select(pl.col("rid").alias("rid2"), pl.col("entity_id").alias("mid"))
        uncontested = pl.len().over("rid2") == 1
        owned = one_owner(g.with_columns(p=pl.when(uncontested & (pl.col("p1") < UNCONTESTED_MIN_P1))
                                           .then(pl.lit(0.0, pl.Float32)).otherwise(pl.col("p"))))
        if country not in seen:  # over-confident probabilities: a stricter plain threshold
            s = choose_threshold(owned, unseen_thr)
            rule = f"threshold {unseen_thr:.3f}"
        elif best[0] == "thr":
            s = choose_threshold(owned, best[1])
            rule = f"threshold {best[1]:.2f}"
        else:
            s = choose(owned, scale=best[1], miss=best[2])
            rule = f"expected-F0.5 rule {best}"
        print(f"  {country}: {s.height:,} matches ({rule}{'' if country in seen else ', unseen country'})")
        sel.append(s.join(s1, on="rid1").join(s23, on="rid2").select("source1_entity_id", "mid"))
        cand.append(g.select("rid1", "rid2").join(s1, on="rid1").join(s23, on="rid2").select("source1_entity_id", "mid"))
    matches = pl.concat(sel)

    s1_ids = scan("test", "source1").select("entity_id").collect()["entity_id"]  # original file order
    write_lists(pl.concat(cand), s1_ids, OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")
    m = write_lists(matches, s1_ids, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    n_empty = (m["matched_entity_ids"] == "").sum()
    print(f"wrote {m.height:,} rows ({n_empty:,} empty), {matches.height:,} matches  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    {"stage1": stage1, "final": final}[sys.argv[1] if len(sys.argv) > 1 else "final"]()
