"""Step 3b: learned candidate pruner — the last blocking stage.

The multi-key blocking keeps up to K candidates per Source-1 record. This cheap second stage uses
only blocking-time signals (per-key scores, learned blocking probability, one fuzzy name+address
score per pair, and Source-1 competition ranks) to drop candidates that are almost certainly not
matches. What survives is the candidate set written to candidate_pairs.tsv and scored by the
matching model.

    python prune.py            # train on non-validation train S1, report size/recall trade-off
"""
import json

import lightgbm as lgb
import polars as pl

from blocking import BS_COLS
from config import SEED, WORK
from data import countries, is_valid, truth_pairs

PRUNE_FEATS = BS_COLS + ["bscore", "n_types", "bprob", "brank", "pair_q", "q_rank1", "q_rank2", "b_rank2",
                         "q_gap1", "q_gap2", "q_second2", "n_cand1", "n_cand2", "n_strong1", "b_rel1", "r_rank2"]
PRUNER_PATH = WORK / "pruner.txt"
PRUNE_THR = 0.005  # ~5 candidates per Source-1 record; keeps >99.98% of matcher-accepted true pairs


_BOOSTER = None


def apply(df: pl.DataFrame, thr: float = PRUNE_THR) -> pl.DataFrame:
    """Keep only candidates the pruner scores >= thr (adds column `pp`)."""
    global _BOOSTER
    if _BOOSTER is None:
        _BOOSTER = lgb.Booster(model_file=str(PRUNER_PATH))
    pp = _BOOSTER.predict(df.select(PRUNE_FEATS).to_numpy())
    return df.with_columns(pp=pl.Series(pp, dtype=pl.Float32)).filter(pl.col("pp") >= thr)


TRAIN_MOD = 20  # non-validation S1 with hash % 20 == 0 (~5%)


def labeled(country: str, which: str) -> pl.DataFrame:
    ids1 = pl.read_parquet(WORK / f"s1ids_train_{country}.parquet").select(pl.col("rid").alias("rid1"), pl.col("entity_id").alias("source1_entity_id"))
    sel = is_valid("source1_entity_id") if which == "valid" else (~is_valid("source1_entity_id") & (pl.col("source1_entity_id").hash(13) % TRAIN_MOD == 0))
    ids1 = ids1.filter(sel)
    ids2 = pl.read_parquet(WORK / f"s23ids_train_{country}.parquet").select(pl.col("rid").alias("rid2"), pl.col("entity_id").alias("mid"))
    cand = pl.scan_parquet(WORK / f"cand_train_{country}.parquet").select("rid1", "rid2", *PRUNE_FEATS) \
             .join(ids1.lazy(), on="rid1").collect().join(ids2, on="rid2")
    tr = truth_pairs("train").collect().join(ids1.select("source1_entity_id"), on="source1_entity_id")
    cand = cand.join(tr.with_columns(label=pl.lit(1, pl.Int8)), on=["source1_entity_id", "mid"], how="left") \
               .with_columns(pl.col("label").fill_null(0), country=pl.lit(country))
    return cand, tr.height, ids1.height


def main():
    tr_parts, va_parts, n_true_va, n_q_va = [], [], 0, 0
    for c in countries("train"):
        t, _, _ = labeled(c, "train")
        v, nt, nq = labeled(c, "valid")
        tr_parts.append(t), va_parts.append(v)
        n_true_va += nt
        n_q_va += nq
    tr, va = pl.concat(tr_parts), pl.concat(va_parts)
    print(f"pruner train pairs {tr.height:,} (pos {tr['label'].sum():,}); valid pairs {va.height:,}")

    params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, seed=SEED, verbose=-1, num_threads=10)
    model = lgb.train(params, lgb.Dataset(tr.select(PRUNE_FEATS).to_numpy(), tr["label"].to_numpy()), num_boost_round=400)
    model.save_model(str(PRUNER_PATH))
    va = va.with_columns(pp=pl.Series(model.predict(va.select(PRUNE_FEATS).to_numpy())))

    # true pairs the v2 matcher accepts (p >= 0.7): the pruner must keep these
    vpp = WORK / "val_pred.parquet"
    vp = (pl.read_parquet(vpp).filter((pl.col("label") == 1) & (pl.col("p") >= 0.7)).select("source1_entity_id", "mid")
          if vpp.exists() else va.filter(pl.col("label") == 1).select("source1_entity_id", "mid"))
    va = va.join(vp.with_columns(acc=pl.lit(1, pl.Int8)), on=["source1_entity_id", "mid"], how="left").with_columns(pl.col("acc").fill_null(0))
    n_acc = va["acc"].sum()
    print(f"K=35: recall {va['label'].sum() / n_true_va:.4f}, avg cand 35.0; matcher-accepted true pairs {n_acc:,}")
    res = []
    for thr in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        k = va.filter(pl.col("pp") >= thr)
        r = dict(thr=thr, avg_cand=k.height / n_q_va, recall=k["label"].sum() / n_true_va, kept_accepted=k["acc"].sum() / n_acc)
        res.append(r)
        print(f"  pp>={thr:<7} avg cand {r['avg_cand']:5.2f}   recall {r['recall']:.4f}   keeps {r['kept_accepted']:.5f} of matcher-accepted true pairs")
    (WORK / "prune_tradeoff.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
