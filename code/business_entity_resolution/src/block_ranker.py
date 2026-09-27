"""Step 2b: learn how to rank the blocking union (which candidates to keep per Source-1 record).

Trained on a small sample of NON-validation training Source-1 queries; reports recall@K on a
held-out half of that sample, comparing the learned ranking against the summed IDF score.

    python block_ranker.py
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import RANK_FEATS, RANKER_PATH, generate, load_country, load_maps
from config import SEED
from data import countries, is_valid, truth_pairs

QUERY_MOD = 40  # ~2.5% of training Source-1 records


def main():
    maps = load_maps()
    truth = truth_pairs("train").collect()
    parts, all_q = [], []
    for country in countries("train"):
        t = time.time()
        s1, s23 = load_country("train", country, maps, cols=("entity_id", "n_name", "n_addr"))
        mask = ~is_valid("entity_id") & (pl.col("entity_id").hash(5) % QUERY_MOD == 0)
        cand = generate(s1, s23, k=10**6, query_mask=mask, use_ranker=False, keep_all=True)
        cand = cand.join(s1.select(pl.col("rid").alias("rid1"), pl.col("entity_id").alias("source1_entity_id")), on="rid1") \
                   .join(s23.select(pl.col("rid").alias("rid2"), pl.col("entity_id").alias("mid")), on="rid2")
        qids = s1.filter(mask).select(pl.col("entity_id").alias("source1_entity_id"))
        tr = truth.join(qids, on="source1_entity_id")
        all_q.append(qids)
        cand = cand.join(tr.select("source1_entity_id", "mid", label=pl.lit(1, pl.Int8)), on=["source1_entity_id", "mid"], how="left") \
                   .with_columns(pl.col("label").fill_null(0))
        n_true = tr.height
        print(f"{country}: {cand.height:,} union pairs, {n_true:,} true pairs, union recall {cand['label'].sum() / n_true:.4f} ({time.time() - t:.0f}s)")
        parts.append(cand)
        del s1, s23
    df = pl.concat(parts, how="vertical_relaxed")

    half = pl.col("source1_entity_id").hash(9) % 2 == 0
    tr, te = df.filter(half), df.filter(~half)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, seed=SEED, verbose=-1, num_threads=10)
    model = lgb.train(params, lgb.Dataset(tr.select(RANK_FEATS).to_numpy(), tr["label"].to_numpy()), num_boost_round=300)
    te = te.with_columns(p=pl.Series(model.predict(te.select(RANK_FEATS).to_numpy())))

    true_te = truth.join(pl.concat(all_q).filter(~half), on="source1_entity_id").height
    for k in (20, 30, 40, 50, 60, 80):
        r_sum = te.filter(pl.col("brank_raw") <= k)["label"].sum() / true_te
        r_lrn = te.filter(pl.col("p").rank("ordinal", descending=True).over("source1_entity_id") <= k)["label"].sum() / true_te
        print(f"  top-{k}: summed-score recall {r_sum:.4f}   learned-ranker recall {r_lrn:.4f}")
    print(f"  union recall on held-out half: {te['label'].sum() / true_te:.4f}")

    model = lgb.train(params, lgb.Dataset(df.select(RANK_FEATS).to_numpy(), df["label"].to_numpy()), num_boost_round=300)
    model.save_model(str(RANKER_PATH))
    print(f"saved {RANKER_PATH}")


if __name__ == "__main__":
    main()
