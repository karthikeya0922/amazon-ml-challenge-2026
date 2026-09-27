"""Step 5: candidates + features per country, written to work/ as parquet parts.

    python pipeline.py train      # candidates + features for every train S1
    python pipeline.py test       # candidates + features for every test S1
    python pipeline.py test France   # re-run one country (e.g. after mine_pseudo.py)

Environment switches: CAND_ONLY=1 stops after candidates (prune.py trains on them), REUSE_CAND=1
rebuilds features from saved candidates, REUSE_FWD=1 rebuilds the candidate union from the saved
forward candidates (skips the forward blocking pass).
"""
import gc
import os
import sys
import time

import polars as pl

import prune
from blocking import K_REV, KEEP_COLS, NO_RANK, add_reverse, generate, generate_reverse, load_country, load_maps
from config import WORK
from data import countries, truth_pairs
from features import build_features, context_features, idf_table, pair_quality, s1_stats

K = 35                 # forward candidates kept per Source-1 record (+ reverse top-K_REV per S2/S3 record)
# ... plus forward ranks K+1..DEEP_K whose cheap pair score (name + address token-set + 100 * number
# Jaccard) is >= DEEP_MIN_Q. Validation sample: India recall 96.31% -> 97.22%, US 98.34% -> 98.61%, for
# ~13 more candidates per Source-1 record before pruning (plain top-80: 97.23% / 98.85% at 80).
DEEP_K = 120
DEEP_MIN_Q = 160.0
DEEP_MAX = 20  # at most this many deep candidates per Source-1 record (best pair score first): India/US
               # average ~13, but records built from a small shared vocabulary (France) pass far more


def deep_selector(s1: pl.DataFrame, s23: pl.DataFrame):
    def select(d: pl.DataFrame) -> pl.DataFrame:
        d = d.with_columns(pair_quality(d, s1, s23).alias("_q")).filter(pl.col("_q") >= DEEP_MIN_Q)
        return d.filter(pl.col("_q").rank("ordinal", descending=True).over("rid1") <= DEEP_MAX).drop("_q")
    return select
FEAT_CHUNK = 150_000   # Source-1 records per feature chunk (features are built on pruned candidates only)


def run_country(split: str, country: str, maps: dict):
    t0 = time.time()
    s1, s23 = load_country(split, country, maps, cols=("entity_id", "n_name", "n_addr", "addr_parts"))
    if os.environ.get("SMOKE"):  # quick end-to-end check on a slice of Source-1
        s1 = s1.head(int(os.environ["SMOKE"]))
    cand_path = WORK / f"cand_{split}_{country}.parquet"
    rev_path = WORK / f"rev_{split}_{country}.parquet"
    if os.environ.get("REUSE_CAND") and cand_path.exists():  # features-only rebuild: keep blocking + context
        cand = pl.read_parquet(cand_path)
        print(f"[{country}] reusing {cand.height:,} saved candidates  ({time.time() - t0:.0f}s)")
    else:
        if os.environ.get("REUSE_FWD") and cand_path.exists():
            fwd = pl.read_parquet(cand_path, columns=KEEP_COLS).filter(pl.col("brank") <= DEEP_K)
            print(f"[{country}] reusing {fwd.height:,} forward candidates  ({time.time() - t0:.0f}s)")
        else:
            fwd = generate(s1, s23, k=K, deep_k=DEEP_K, deep_select=deep_selector(s1, s23))
        if (os.environ.get("REUSE_FWD") or os.environ.get("REUSE_REV")) and rev_path.exists():
            rev = pl.read_parquet(rev_path)
        else:
            rev = generate_reverse(s1, s23, k=K_REV)
            rev.write_parquet(rev_path)
        cand = add_reverse(fwd, rev, K_REV).sort("rid1")
        del fwd, rev
        print(f"[{country}] {s1.height:,} S1, {s23.height:,} S2/S3, {cand.height:,} candidates "
              f"({(cand['brank'] == NO_RANK).sum():,} reverse-only, {cand['brank'].is_between(K + 1, DEEP_K).sum():,} deep)  ({time.time() - t0:.0f}s)")

        # cheap pair quality for every candidate (chunked by rid1; cand is sorted by rid1, so the chunks'
        # scores concatenate in row order), then competition context computed on a slim copy
        qs = []
        bounds = list(range(0, s1.height, 200_000)) + [s1.height]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            qs.append(pair_quality(cand.filter(pl.col("rid1").is_between(lo, hi - 1)).select("rid1", "rid2"), s1, s23))
        cand = cand.with_columns(pl.concat(qs).cast(pl.Float32).alias("pair_q"))
        del qs
        ctx = context_features(cand.select("rid1", "rid2", "bscore", "pair_q"))
        cand = cand.hstack(ctx.drop("rid1", "rid2", "bscore", "pair_q"))
        del ctx
        gc.collect()
        tmp = cand_path.with_suffix(".tmp")
        cand.write_parquet(tmp)
        tmp.replace(cand_path)  # never leave a half-written candidate file behind
        print(f"[{country}] context done ({time.time() - t0:.0f}s)")
    if os.environ.get("CAND_ONLY"):  # the pruner (prune.py) is trained on these before features are built
        return

    # the learned pruner is the last blocking stage: features are only built for what survives it
    cand = prune.apply(cand)
    print(f"[{country}] {cand.height:,} candidates after pruning ({cand.height / s1.height:.2f} per S1)")
    name_idf, addr_idf = idf_table(s1, s23, "n_name"), idf_table(s1, s23, "n_addr")
    stats = s1_stats(s1)
    truth = None
    if split == "train":
        truth = truth_pairs("train").collect().join(
            s1.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id").with_columns(label=pl.lit(1, pl.Int8))

    rids = s1["rid"]  # every Source-1 record: stage 2 needs stage-1 probabilities of all competing S1 records
    for part, lo in enumerate(range(0, rids.len(), FEAT_CHUNK)):
        sel = rids[lo:lo + FEAT_CHUNK]
        c = cand.filter(pl.col("rid1").is_in(sel.implode()))
        f = build_features(c.select("rid1", "rid2", "bscore", "brank"), s1, s23, name_idf, addr_idf, stats)
        f = f.join(c.drop("bscore", "brank"), on=["rid1", "rid2"], how="left")
        f = f.join(s1.select(pl.col("rid").alias("rid1"), pl.col("entity_id").alias("source1_entity_id")), on="rid1")
        if truth is not None:
            f = f.join(truth, on=["source1_entity_id", "mid"], how="left").with_columns(pl.col("label").fill_null(0))
        f.with_columns(country=pl.lit(country)).write_parquet(WORK / f"feat_{split}_{country}_{part:03d}.parquet")
        del f, c
        gc.collect()
    # Source-1 ids of this country (needed to score singletons / write every row)
    s1.select("rid", "entity_id").with_columns(country=pl.lit(country)).write_parquet(WORK / f"s1ids_{split}_{country}.parquet")
    s23.select("rid", "entity_id").write_parquet(WORK / f"s23ids_{split}_{country}.parquet")
    print(f"[{country}] features done ({time.time() - t0:.0f}s)")


def main(split: str, only: list[str] | None = None):
    maps = load_maps()
    if only == ["unseen"]:  # countries without training labels (re-run after mine_pseudo.py)
        only = sorted(set(countries(split)) - set(countries("train")))
    todo = only or countries(split)
    for country in todo:
        for old in WORK.glob(f"feat_{split}_{country}_*.parquet"):
            old.unlink()
    for country in todo:
        run_country(split, country, maps)
        gc.collect()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train", sys.argv[2:] or None)
