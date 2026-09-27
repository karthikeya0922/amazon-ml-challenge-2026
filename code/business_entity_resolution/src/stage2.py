"""Step 7: stage-2 pair classifier over stage-1 probabilities (collective / cluster evidence).

Stage 1 scores every (Source-1, S2/S3) pair on its own. Stage 2 adds what only becomes visible once
all pairs are scored:
  * competition — the pair's p1 rank for its Source-1 record and for its S2/S3 record, the best
    competing owner's p1, the Source-1 record's expected number of matches (sum of p1);
  * cluster evidence — how similar the candidate is to the Source-1 record's other confident
    matches (records of the same business resemble each other) versus to its rejected candidates
    (decoy records of a sibling business form their own look-alike group).

    python stage2.py build train     # stage-2 features for every train pair (needs stage1_train.parquet)
    python stage2.py train           # fit the stage-2 model, pick the decision rule on validation
    python stage2.py build test      # stage-2 features for every test pair (needs stage1_test.parquet)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from blocking import load_country, load_maps
from config import SEED, WORK
from data import is_valid
from features import _cpdist
from train import feature_cols, parts, report, to_x

MEMBER_P = 0.5       # a candidate counts as a confident match of its Source-1 record at p1 >= MEMBER_P ...
NO_LIFT_P1 = 0.5     # stage 2 may lower but never raise the probability of a pair stage 1 rejected (p1 < NO_LIFT_P1)


def final_prob(p1: pl.Expr, p2: pl.Expr) -> pl.Expr:
    """Final pair probability: stage 2, except that it cannot lift a pair stage 1 rejected.

    Stage 2 learned from training data that an S2/S3 record no other Source-1 record wants, and that
    resembles the Source-1 record's other matches, is a match. The test set has many more S2/S3
    records whose business is absent from Source 1; such look-alike records are uncontested too, and
    stage 2 lifted 5-11x more stage-1-rejected pairs on test than on validation (leaderboard 0.970 vs
    0.973). Without lifting, validation F0.5 is 0.9864 instead of 0.9869.
    """
    return pl.when(p1 < NO_LIFT_P1).then(pl.min_horizontal(p1, p2)).otherwise(p2)
S2_TRAIN_FRAC = 5    # of 10: non-validation Source-1 entities whose pairs train the stage-2 model
CHUNK = 100_000      # Source-1 records per cluster-feature chunk
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED + 1, verbose=-1, num_threads=10)


def p_context(d: pl.DataFrame) -> pl.DataFrame:
    """d: (rid1, rid2, p1) for every pair of one country."""
    return d.with_columns(
        p1_rank1=pl.col("p1").rank("ordinal", descending=True).over("rid1").cast(pl.Int16),
        p1_gap1=(pl.col("p1") - pl.col("p1").max().over("rid1")).cast(pl.Float32),
        p1_sum1=pl.col("p1").sum().over("rid1").cast(pl.Float32),
        n_hi1=(pl.col("p1") >= MEMBER_P).sum().over("rid1").cast(pl.Int16),
        p1_rank2=pl.col("p1").rank("ordinal", descending=True).over("rid2").cast(pl.Int16),
        n_hi2=(pl.col("p1") >= MEMBER_P).sum().over("rid2").cast(pl.Int16),
        _max2=pl.col("p1").max().over("rid2"),
        _sec2=pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("rid2").fill_null(0.0),
    ).with_columns(
        # best competing owner's p1 (the runner-up when this pair is the S2/S3 record's best)
        p1_other2=pl.when(pl.col("p1_rank2") == 1).then(pl.col("_sec2")).otherwise(pl.col("_max2")).cast(pl.Float32),
    ).drop("_max2", "_sec2")


FIRST_NUM = r"\b(\d+)\b"


def cluster_feats(d: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """Similarity of each candidate to its Source-1 record's confident matches vs its other candidates.

    d: (rid1, rid2, p1, p1_rank2) for a block of Source-1 records.
    (Tried and dropped: counts of candidates sharing a house number different from the Source-1
    record's. In train such groups are rare and mostly true matches, in test 13x more frequent and
    mostly decoys, so the learned signal pointed the wrong way on test.)
    """
    txt = s23.select(pl.col("rid").alias("rid2"), "n_name", "n_addr", num=pl.col("n_addr").str.extract(FIRST_NUM, 1))
    x = d.select("rid1", "rid2", member=(pl.col("p1") >= MEMBER_P) & (pl.col("p1_rank2") == 1)).join(txt, on="rid2", how="left")
    pr = x.join(x, on="rid1", suffix="_y").filter(pl.col("rid2") != pl.col("rid2_y"))
    if pr.height == 0:
        return d.select("rid1", "rid2")
    # names: token-sort (not token-set) so that an extra word ("... international") lowers the similarity —
    # decoy records of a sibling business typically differ from the real ones by exactly such a word
    nsim = _cpdist(pr["n_name"].to_list(), pr["n_name_y"].to_list(), fuzz.token_sort_ratio)
    asim = _cpdist(pr["n_addr"].to_list(), pr["n_addr_y"].to_list(), fuzz.token_set_ratio)
    both_empty = (pr["n_addr"] == "") | (pr["n_addr_y"] == "")
    pr = pr.select(
        "rid1", "rid2", "member_y",
        ns=pl.Series(nsim),
        as_=pl.when(both_empty).then(None).otherwise(pl.Series(asim)),
        num_eq=(pl.col("num") == pl.col("num_y")).fill_null(False),
        dup=(pl.col("n_name") == pl.col("n_name_y")),
    ).with_columns(bs=(pl.col("ns") + pl.col("as_").fill_null(pl.col("ns"))) / 2)
    m, o = pl.col("member_y"), ~pl.col("member_y")
    agg = pr.group_by("rid1", "rid2").agg(
        m_cnt=m.sum().cast(pl.Int16),
        m_name_max=pl.col("ns").filter(m).max(),
        m_addr_max=pl.col("as_").filter(m).max(),
        m_both_max=pl.col("bs").filter(m).max(),
        m_both_mean=pl.col("bs").filter(m).mean(),
        m_num_eq=pl.col("num_eq").filter(m).any().cast(pl.Int8),
        o_name_max=pl.col("ns").filter(o).max(),
        o_addr_max=pl.col("as_").filter(o).max(),
        o_both_max=pl.col("bs").filter(o).max(),
        n_dup_name=pl.col("dup").sum().cast(pl.Int16),
    )
    return d.select("rid1", "rid2").join(agg, on=["rid1", "rid2"], how="left").with_columns(
        pl.col("m_cnt").fill_null(0), pl.col("n_dup_name").fill_null(0),
        m_vs_o=(pl.col("m_both_max").fill_null(0) - pl.col("o_both_max").fill_null(0)).cast(pl.Float32))


def build(split: str):
    t = time.time()
    maps = load_maps()
    p1 = pl.read_parquet(WORK / f"stage1_{split}.parquet", columns=["country", "rid1", "rid2", "p1"])
    for (country,), d in p1.group_by("country"):
        _, s23 = load_country(split, country, maps, cols=("entity_id", "n_name", "n_addr"))
        d = p_context(d.drop("country"))
        out = []
        n1 = d["rid1"].max() + 1
        for lo in range(0, n1, CHUNK):
            b = d.filter(pl.col("rid1").is_between(lo, lo + CHUNK - 1))
            out.append(b.join(cluster_feats(b, s23), on=["rid1", "rid2"], how="left"))
        f = pl.concat(out).drop("p1").with_columns(country=pl.lit(country))
        f.write_parquet(WORK / f"s2feat_{split}_{country}.parquet")
        print(f"[{country}] stage-2 features for {f.height:,} pairs ({time.time() - t:.0f}s)", flush=True)
        del s23, d, out, f


KEYS = ["country", "rid1", "rid2"]


def joined_parts(split: str, filt: pl.Expr | None = None):
    """Yield stage-1 features + p1 + stage-2 features, one feature part at a time."""
    s1p = pl.read_parquet(WORK / f"stage1_{split}.parquet", columns=KEYS + ["p1"])
    s2f = pl.concat([pl.read_parquet(p) for p in WORK.glob(f"s2feat_{split}_*.parquet")], how="vertical_relaxed")
    for p in parts(split):
        f = pl.read_parquet(p)
        if filt is not None:
            f = f.filter(filt)
        yield f.join(s1p, on=KEYS, how="left").join(s2f, on=KEYS, how="left")


def stage2_features(cols) -> list[str]:
    return [f for f in dict.fromkeys(list(feature_cols(cols)) + ["p1"]) if f != "v"]


def fit():
    """Two stage-2 models on the stage-1 folds (out-of-fold p2 for every train pair, like stage 1)."""
    from train import N_FOLDS, fold_expr
    t = time.time()
    sample = is_valid() | (pl.col("source1_entity_id").hash(5) % 10 < S2_TRAIN_FRAC)
    df = pl.concat(list(joined_parts("train", sample)), how="vertical_relaxed").with_columns(fold=fold_expr(), v=is_valid())
    feats = stage2_features(df.columns)
    va = df.filter(pl.col("v"))
    xva, yva = to_x(va, feats), va["label"].to_numpy()
    del va
    print(f"stage 2: {df.height:,} sampled pairs; {len(feats)} features ({time.time() - t:.0f}s)", flush=True)
    models = []
    for f in range(N_FOLDS):
        tr = df.filter(~pl.col("v") & (pl.col("fold") == f))
        dtr = lgb.Dataset(to_x(tr, feats), tr["label"].to_numpy(), feature_name=feats, free_raw_data=True)
        del tr
        m = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[lgb.Dataset(xva, yva, reference=dtr)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
        m.save_model(str(WORK / f"lgb_s2_f{f}.txt"), num_iteration=m.best_iteration)
        models.append(m)
        print(f"stage 2 fold {f}: {m.best_iteration} rounds ({time.time() - t:.0f}s)", flush=True)
    del df, xva

    # out-of-fold p2 for every train pair (fold-average on validation), then decide on validation
    out = []
    for d in joined_parts("train"):
        d = d.with_columns(fold=fold_expr(), v=is_valid())
        x = to_x(d, feats)
        pf = np.stack([m.predict(x, num_iteration=m.best_iteration) for m in models], axis=1)
        p2 = np.where(d["v"].to_numpy(), pf.mean(axis=1), pf[np.arange(len(d)), 1 - d["fold"].to_numpy()])
        out.append(d.select(*KEYS, "source1_entity_id", "mid", "label", "v", "p1", p2=pl.Series(p2, dtype=pl.Float32)))
    allp = pl.concat(out).with_columns(p=final_prob(pl.col("p1"), pl.col("p2")))
    allp.write_parquet(WORK / "stage2_train_pred.parquet")
    print(f"stage-2 probabilities for {allp.height:,} train pairs ({time.time() - t:.0f}s)", flush=True)
    report(allp, "p1", "stage1")
    report(allp, "p2", "stage2")
    # final rule: a plain threshold on the no-lift probability (the expected-F0.5 rule relies on
    # calibrated probabilities, which do not hold for the test set's ownerless look-alike records)
    _, results = report(allp, "p", "final")
    best_thr = max((k for k in results if k[0] == "thr"), key=results.get)
    (WORK / "best_decision.json").write_text(json.dumps({"best": list(best_thr), "best_thr": best_thr[1]}))
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, int(g)) for f, g in imp[:25]])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "train":
        fit()
