"""Step 6: stage-1 pair classifier (LightGBM), trained out-of-fold so stage 2 can use its probabilities.

Non-validation training Source-1 entities are split into two folds by hash. One model is trained per
fold (on a sample of that fold's pairs, early-stopped on the validation split) and predicts the other
fold, so every training pair gets an out-of-fold probability p1. Validation and test pairs get the
average of the two fold models.

    python train.py
Writes work/lgb_s1_f{0,1}.txt, work/stage1_train.parquet (every train pair with p1) and prints the
stage-1 validation score.
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import SEED, WORK
from data import is_valid, truth_pairs
from decide import choose, choose_threshold, one_owner
from metric import score

NON_FEATURES = {"rid1", "rid2", "mid", "source1_entity_id", "label", "country", "pp", "p1", "fold"}
N_FOLDS = 2
TRAIN_FRAC = 10  # of 10: share of each fold's Source-1 entities whose pairs are used to fit that fold's model (all)
PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, verbose=-1, num_threads=10)


def fold_expr() -> pl.Expr:
    return (pl.col("source1_entity_id").hash(17) % N_FOLDS).cast(pl.Int8)


def parts(split: str) -> list:
    return sorted(WORK.glob(f"feat_{split}_*.parquet"))


def feature_cols(cols) -> list[str]:
    return [c for c in cols if c not in NON_FEATURES]


def load(split: str, filt: pl.Expr | None = None, cols: list[str] | None = None) -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(p) for p in parts(split)], how="vertical_relaxed")
    if filt is not None:
        lf = lf.filter(filt)
    if cols is not None:
        lf = lf.select(cols)
    return lf.collect()


def to_x(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def predictions_to_long(sel: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    ids = df.select("country", "rid1", "rid2", "source1_entity_id", "mid")
    return sel.join(ids, on=["country", "rid1", "rid2"]).select("source1_entity_id", "mid")


def evaluate(owned: pl.DataFrame, s1_ids: pl.Series, truth: pl.DataFrame, tag: str = "", prob: str = "p",
             thresholds=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> tuple:
    """Try the decision rules on validation pairs already reduced to one owner per S2/S3 record."""
    results = {}
    val = owned
    for thr in thresholds:
        sel = pl.concat([choose_threshold(g, thr, prob).with_columns(country=pl.lit(c))
                         for (c,), g in owned.group_by("country")])
        results[("thr", thr)] = score(predictions_to_long(sel, val), truth, s1_ids)["f05"]
    for scale in (0.9, 1.0):
        sel = pl.concat([choose(g, prob, miss=0.2, scale=scale).with_columns(country=pl.lit(c))
                         for (c,), g in owned.group_by("country")])
        results[("ef", scale, 0.2)] = score(predictions_to_long(sel, val), truth, s1_ids)["f05"]
    for k, v in sorted(results.items(), key=lambda x: -x[1])[:4]:
        print(f"  {tag} {k}: {v:.5f}")
    return max(results, key=results.get), results


def validation_population():
    s1_ids = pl.concat([pl.read_parquet(p) for p in WORK.glob("s1ids_train_*.parquet")]).filter(is_valid("entity_id"))
    truth = truth_pairs("train").filter(is_valid()).collect()
    return s1_ids, truth


def report(pairs: pl.DataFrame, prob: str, tag: str):
    """pairs: every train pair (all Source-1 records compete for ownership) with a `v` validation flag."""
    s1_ids, truth = validation_population()
    owned = pl.concat([one_owner(g, prob) for _, g in pairs.group_by("country")]).filter(pl.col("v"))
    best, results = evaluate(owned, s1_ids["entity_id"], truth, f"{tag} ALL", prob)
    for c in s1_ids["country"].unique().sort():
        ids_c = s1_ids.filter(pl.col("country") == c)["entity_id"]
        evaluate(owned.filter(pl.col("country") == c), ids_c,
                 truth.filter(pl.col("source1_entity_id").is_in(ids_c.implode())), f"{tag} {c}", prob)
    return best, results


def main():
    t = time.time()
    feats = feature_cols(pl.read_parquet_schema(parts("train")[0]).keys())
    val = load("train", is_valid())
    print(f"validation pairs {val.height:,}; {len(feats)} features ({time.time() - t:.0f}s)", flush=True)
    xva, yva = to_x(val, feats), val["label"].to_numpy()

    models = []
    for f in range(N_FOLDS):
        tr = load("train", ~is_valid() & (fold_expr() == f) & (pl.col("source1_entity_id").hash(3) % 10 < TRAIN_FRAC),
                  feats + ["label"])
        print(f"fold {f}: {tr.height:,} training pairs (pos {tr['label'].sum():,})", flush=True)
        dtr = lgb.Dataset(to_x(tr, feats), tr["label"].to_numpy(), feature_name=feats, free_raw_data=True)
        del tr
        m = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[lgb.Dataset(xva, yva, reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
        m.save_model(str(WORK / f"lgb_s1_f{f}.txt"), num_iteration=m.best_iteration)
        models.append(m)
        print(f"fold {f}: {m.best_iteration} rounds ({time.time() - t:.0f}s)", flush=True)

    # out-of-fold p1 for every training pair; fold-average for validation
    out = []
    for p in parts("train"):
        d = pl.read_parquet(p).with_columns(fold=fold_expr(), v=is_valid())
        x = to_x(d, feats)
        pf = np.stack([m.predict(x, num_iteration=m.best_iteration) for m in models], axis=1)
        fold = d["fold"].to_numpy()
        p1 = np.where(d["v"].to_numpy(), pf.mean(axis=1), pf[np.arange(len(d)), 1 - fold] if N_FOLDS == 2 else pf.mean(axis=1))
        out.append(d.select("country", "rid1", "rid2", "source1_entity_id", "mid", "label", "v",
                            p1=pl.Series(p1, dtype=pl.Float32)))
    s1p = pl.concat(out)
    s1p.write_parquet(WORK / "stage1_train.parquet")
    print(f"stage-1 probabilities for {s1p.height:,} train pairs ({time.time() - t:.0f}s)", flush=True)

    report(s1p, "p1", "stage1")
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, int(g)) for f, g in imp[:20]])


if __name__ == "__main__":
    main()
