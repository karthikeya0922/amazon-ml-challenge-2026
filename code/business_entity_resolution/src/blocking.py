"""Step 3: candidate generation (blocking) with an IDF-weighted inverted index.

Per country (country always agrees within true pairs), every record is turned into a set of
tokens: name words (n|), address words (a|) and address numbers (d|). A Source-1 record and an
S2/S3 record become candidates when they share rare tokens; the candidate score is the sum of
IDF weights of the shared tokens. Tokens whose document frequency exceeds MAX_DF are ignored
(they carry little identity information and would explode the join). The top-K S2/S3 records per
Source-1 record are kept.
"""
import json
import time

import numpy as np
import polars as pl

from config import WORK
from normalize import apply_part_map, apply_token_map
from prep import prep_path

CHUNK = 25_000
KEY_SLICE = 250_000


def load_maps() -> dict:
    p = WORK / "token_maps.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_country(split: str, country: str, maps: dict,
                 cols=("entity_id", "business_name", "business_address", "n_name", "n_addr", "addr_parts")
                 ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Normalized + vocabulary-mapped Source-1 and S2/S3 records of one country."""
    m = maps.get(country, {})

    def prep(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.filter(pl.col("country") == country)
            .with_columns(apply_part_map("addr_parts", m.get("part_map", {})))
            .with_columns(n_addr=pl.col("addr_parts").list.join(" "))
            .with_columns(apply_token_map("n_addr", m.get("addr_map", {})),
                          apply_token_map("n_name", m.get("name_map", {})))
            .select(*cols)
        )

    s1 = prep(pl.scan_parquet(prep_path(split, "source1"))).collect()
    s23 = prep(pl.concat([pl.scan_parquet(prep_path(split, s)) for s in ("source2", "source3")])).collect()
    return s1.with_row_index("rid"), s23.with_row_index("rid")


def _toks(col: str, n: int, numeric: bool | None, min_len: int = 1) -> pl.Expr:
    """First n distinct tokens of a column; numeric=True only numbers, False only words, None both."""
    e = pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= min_len))
    if numeric is True:
        e = e.list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")))
    elif numeric is False:
        e = e.list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$")))
    return e.list.unique(maintain_order=True).list.head(n)


def _h(e: pl.Expr, seed: int) -> pl.Expr:
    """Hash every element of a list of strings -> list of u64."""
    return e.list.eval(pl.element().hash(seed))


def _pairs(a: pl.Expr, b: pl.Expr, na: int, nb: int, symmetric: bool) -> pl.Expr:
    """Combined u64 keys for all (a_i, b_j); inputs are lists of u64 hashes.

    XOR is order-free, so symmetric pairs of one list match regardless of word order.
    """
    out = []
    for i in range(na):
        for j in range(nb):
            if symmetric and j <= i:
                continue
            out.append(a.list.get(i, null_on_oob=True).xor(b.list.get(j, null_on_oob=True)))
    return pl.concat_list(out).list.drop_nulls()


# hashed token lists, materialized once per slice of records
TOKEN_LISTS = {
    "nm": lambda: _h(_toks("n_name", 12, None, 2), 11),   # name words
    "aw": lambda: _h(_toks("n_addr", 16, False, 3), 12),  # address words
    "an": lambda: _h(_toks("n_addr", 6, True), 13),       # address numbers
    # glued names ("laxmiseals", "serviceinfra"): first-2 / first-3 / all words without spaces,
    # plus every long single word, all in one hash space so a glued token meets its split form
    "ns": lambda: _h(pl.concat_list(
        pl.col("n_name").str.split(" ").list.head(2).list.join(""),
        pl.col("n_name").str.split(" ").list.head(3).list.join(""),
        pl.col("n_name").str.replace_all(" ", ""),
        pl.col("n_name").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 6)),
    ).list.unique(), 51),
}
L = pl.col
# key type -> (list-of-u64-keys expression over TOKEN_LISTS columns, max document frequency)
KEY_TYPES = {
    "uni": (lambda: pl.concat_list(L("nm"), L("aw"), L("an")), 1000),
    "name2": (lambda: _pairs(L("nm"), L("nm"), 7, 7, True), 2000),
    "numword": (lambda: _pairs(L("an"), L("aw").list.eval(pl.element() * 3), 3, 8, False), 2000),
    "namenum": (lambda: _pairs(L("nm").list.eval(pl.element() * 5), L("an"), 5, 3, False), 2000),
    "nameaddr": (lambda: _pairs(L("nm").list.eval(pl.element() * 11), L("aw"), 4, 6, False), 500),
    "concat": (lambda: L("ns"), 1000),
    "addr2": (lambda: _pairs(L("aw").list.eval(pl.element() * 7), L("aw").list.eval(pl.element() * 7), 6, 6, True), 2000),
}
K_PER_TYPE = 40


def key_table(df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    """(rid, tok) with tok = u64 key; one row per distinct key per record.

    Built in record slices so the key strings of only one slice exist in memory at a time.
    """
    out = []
    for lo in range(0, df.height, KEY_SLICE):
        out.append(
            df.slice(lo, KEY_SLICE).select("rid", **{k: f() for k, f in TOKEN_LISTS.items()})
            .select("rid", expr.list.unique().alias("tok"))
            .explode("tok")
            .drop_nulls("tok")
            .select(pl.col("rid").cast(pl.UInt32), "tok")
        )
    return pl.concat(out)


SPILL = WORK / "spill"


def _type_candidates(s1, s23, expr, max_df, query_rids, k, name, prefix: str = "") -> int:
    """Top-k pairs per rid1 for one key type, spilled to SPILL/<prefix><name>_<block>.parquet by rid1 block.

    The pair score (summed IDF of shared keys) is symmetric, so calling it with the two sides swapped
    gives the top-k Source-1 records per S2/S3 record (reverse blocking).
    """
    t1, t2 = key_table(s1, expr), key_table(s23, expr)
    n_docs = s1.height + s23.height
    df = pl.concat([t1.select("tok"), t2.select("tok")]).group_by("tok").len("df")
    df = df.filter(pl.col("df") <= max_df).with_columns(idf=(n_docs / pl.col("df")).log().cast(pl.Float32))
    t2 = t2.join(df, on="tok").select("tok", pl.col("rid").alias("rid2"), "idf")
    t1 = t1.join(df.select("tok"), on="tok").rename({"rid": "rid1"})
    if query_rids is not None:
        t1 = t1.filter(pl.col("rid1").is_in(query_rids.implode()))
    del df
    t1 = t1.sort("rid1")
    n = 0
    for blk in range(s1.height // CHUNK + 1):
        chunk = t1.filter(pl.col("rid1").is_between(blk * CHUNK, (blk + 1) * CHUNK - 1))
        if chunk.height == 0:
            continue
        res = (
            chunk.join(t2, on="tok")
            .group_by("rid1", "rid2").agg(pl.col("idf").sum().alias(f"bs_{name}"))
            .filter(pl.col(f"bs_{name}").rank("ordinal", descending=True).over("rid1") <= k)
        )
        n += res.height
        res.write_parquet(SPILL / f"{prefix}{name}_{blk:05d}.parquet")
    return n


K_REV_PER_TYPE = 5  # reverse candidates per S2/S3 record per key type
K_REV = 1           # reverse candidates per S2/S3 record kept in the union (recall +0.4% for +0.3 pairs per S1)


def generate_reverse(s1: pl.DataFrame, s23: pl.DataFrame, k: int = K_REV, verbose: bool = False) -> pl.DataFrame:
    """Top-k Source-1 records per S2/S3 record -> (rid1, rid2, rs_<type>..., rscore, r_rank2).

    Source-1-centric blocking keeps the top K S2/S3 records per Source-1 record, which drops S2/S3
    records that only share weak keys (typically empty addresses: name keys only) whenever the Source-1
    record has many stronger candidates. Seen from the S2/S3 side the same record usually has only a
    few plausible owners, so its best Source-1 records are kept here and unioned with the forward set.
    """
    SPILL.mkdir(exist_ok=True)
    for f in SPILL.glob("rev_*.parquet"):
        f.unlink()
    for name, (make, max_df) in KEY_TYPES.items():
        t = time.time()
        n = _type_candidates(s23, s1, make(), max_df, None, K_REV_PER_TYPE, name, prefix="rev_")
        if verbose:
            print(f"    reverse keys {name}: {n:,} pairs ({time.time() - t:.0f}s)", flush=True)
    rs_cols = [f"rs_{n}" for n in KEY_TYPES]
    out = []
    for blk in range(s23.height // CHUNK + 1):
        merged = None
        for name in KEY_TYPES:
            f = SPILL / f"rev_{name}_{blk:05d}.parquet"
            if f.exists():
                c = pl.read_parquet(f).rename({f"bs_{name}": f"rs_{name}"})
                merged = c if merged is None else merged.join(c, on=["rid1", "rid2"], how="full", coalesce=True)
        if merged is None:
            continue
        for c in rs_cols:
            if c not in merged.columns:
                merged = merged.with_columns(pl.lit(0.0, pl.Float32).alias(c))
        merged = (
            merged.with_columns(pl.col(rs_cols).fill_null(0.0))
            .with_columns(rscore=pl.sum_horizontal(rs_cols))
            .with_columns(r_rank2=pl.col("rscore").rank("ordinal", descending=True).over("rid1").cast(pl.Int16))
            .filter(pl.col("r_rank2") <= k)
        )
        # swap back: the query side of this run was S2/S3
        out.append(merged.rename({"rid1": "rid2", "rid2": "rid1"}).select("rid1", "rid2", *rs_cols, "rscore", "r_rank2"))
    for f in SPILL.glob("rev_*.parquet"):
        f.unlink()
    return pl.concat(out)


RANKER_PATH = WORK / "block_ranker.txt"
BS_COLS = [f"bs_{n}" for n in KEY_TYPES]
RANK_FEATS = BS_COLS + [f"{c}_rel" for c in BS_COLS] + ["bscore", "n_types", "bscore_rel", "brank_raw", "n_union1"]


def rank_features(m: pl.DataFrame) -> pl.DataFrame:
    """Per-candidate blocking features: key-type scores, relative to the S1's best, and counts."""
    m = m.with_columns(pl.col(BS_COLS).fill_null(0.0)).with_columns(
        bscore=pl.sum_horizontal(BS_COLS),
        n_types=pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int8) for c in BS_COLS]),
    )
    return m.with_columns(
        *[(pl.col(c) / (pl.col(c).max().over("rid1") + 1e-6)).cast(pl.Float32).alias(f"{c}_rel") for c in BS_COLS],
        bscore_rel=(pl.col("bscore") / pl.col("bscore").max().over("rid1")).cast(pl.Float32),
        brank_raw=pl.col("bscore").rank("ordinal", descending=True).over("rid1").cast(pl.Int32),
        n_union1=pl.len().over("rid1").cast(pl.Int32),
    )


KEEP_COLS = ["rid1", "rid2", *BS_COLS, "bscore", "n_types", "bprob", "brank"]
NO_RANK = 999  # brank / r_rank2 of a pair that the forward / reverse pass did not produce (> any real rank)


def add_reverse(fwd: pl.DataFrame, rev: pl.DataFrame, k_rev: int = 1) -> pl.DataFrame:
    """Union of forward candidates (top-K per Source-1 record) and reverse top-k_rev per S2/S3 record.

    Reverse-only pairs take their per-key-type scores from the reverse pass (same definition), have
    no blocking-ranker probability (null) and brank = NO_RANK. r_rank2 is the Source-1 record's rank
    among the S2/S3 record's reverse candidates (NO_RANK when not among them).
    """
    rev = rev.filter(pl.col("r_rank2") <= k_rev)
    new = rev.join(fwd.select("rid1", "rid2"), on=["rid1", "rid2"], how="anti").select(
        "rid1", "rid2", *[pl.col("rs" + c[2:]).alias(c) for c in BS_COLS])
    new = new.with_columns(
        bscore=pl.sum_horizontal(BS_COLS),
        n_types=pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int8) for c in BS_COLS]),
        bprob=pl.lit(None, pl.Float32),
        brank=pl.lit(NO_RANK, pl.UInt32),
    )
    out = pl.concat([fwd.select(KEEP_COLS), new.select(KEEP_COLS)], how="vertical_relaxed")
    return out.join(rev.select("rid1", "rid2", "r_rank2"), on=["rid1", "rid2"], how="left") \
              .with_columns(pl.col("r_rank2").fill_null(NO_RANK).cast(pl.Int16))


def generate(s1: pl.DataFrame, s23: pl.DataFrame, k: int, query_mask: pl.Expr | None = None,
             verbose: bool = False, use_ranker: bool = True, keep_all: bool = False,
             deep_k: int = 0, deep_select=None) -> pl.DataFrame:
    """Top-k candidates per Source-1 record -> (rid1, rid2, bs_<type>..., bscore, n_types, bprob, brank).

    Each key type's candidates are spilled to disk by rid1 block; blocks are then merged, ranked by
    the learned blocking ranker (block_ranker.py) when available, and cut to the top k.
    deep_k / deep_select: candidates ranked k+1..deep_k are passed to deep_select(frame), which returns
    the ones to keep as well (a cheap string-similarity check): true pairs beyond the cut are few but
    nearly all look alike.
    """
    SPILL.mkdir(exist_ok=True)
    for f in SPILL.glob("*.parquet"):
        f.unlink()
    query_rids = s1.filter(query_mask)["rid"] if query_mask is not None else None
    for name, (make, max_df) in KEY_TYPES.items():
        t = time.time()
        n = _type_candidates(s1, s23, make(), max_df, query_rids, K_PER_TYPE, name)
        if verbose:
            print(f"    keys {name}: {n:,} pairs ({time.time() - t:.0f}s)", flush=True)

    booster = None
    if use_ranker and RANKER_PATH.exists():
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(RANKER_PATH))
    out = []
    for blk in range(s1.height // CHUNK + 1):
        merged = None
        for name in KEY_TYPES:
            f = SPILL / f"{name}_{blk:05d}.parquet"
            if f.exists():
                c = pl.read_parquet(f)
                merged = c if merged is None else merged.join(c, on=["rid1", "rid2"], how="full", coalesce=True)
        if merged is None:
            continue
        for c in BS_COLS:
            if c not in merged.columns:
                merged = merged.with_columns(pl.lit(0.0, pl.Float32).alias(c))
        merged = rank_features(merged)
        if booster is not None:
            merged = merged.with_columns(bprob=pl.Series(booster.predict(merged.select(RANK_FEATS).to_numpy()), dtype=pl.Float32))
        else:
            merged = merged.with_columns(bprob=pl.col("bscore_rel"))
        merged = merged.with_columns(brank=pl.col("bprob").rank("ordinal", descending=True).over("rid1"))
        top = merged.filter(pl.col("brank") <= k)
        if deep_k > k and deep_select is not None:
            deep = merged.filter(pl.col("brank").is_between(k + 1, deep_k))
            if deep.height:
                top = pl.concat([top, deep_select(deep)])
        out.append(top if keep_all else top.select(KEEP_COLS))
    for f in SPILL.glob("*.parquet"):
        f.unlink()
    return pl.concat(out)


def blocking_recall(split: str = "train", k_values=(10, 20, 30, 50, 80, 160)):
    """Recall of true pairs among top-k candidates for validation Source-1 entities."""
    from data import countries, is_valid, truth_pairs

    maps = load_maps()
    truth = truth_pairs(split).filter(is_valid()).collect()
    for country in countries(split):
        t = time.time()
        s1, s23 = load_country(split, country, maps, cols=("entity_id", "n_name", "n_addr"))
        cand = generate(s1, s23, k=max(k_values), query_mask=is_valid("entity_id"), verbose=True)
        tm = time.time() - t
        cand = cand.join(s1.select(pl.col("rid").alias("rid1"), pl.col("entity_id").alias("source1_entity_id")), on="rid1") \
                   .join(s23.select(pl.col("rid").alias("rid2"), pl.col("entity_id").alias("mid")), on="rid2")
        tr = truth.join(s1.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id")
        hit = tr.join(cand, on=["source1_entity_id", "mid"], how="left")
        cand.join(tr.select("source1_entity_id", "mid", label=pl.lit(1, pl.Int8)), on=["source1_entity_id", "mid"], how="left")             .with_columns(pl.col("label").fill_null(0)).write_parquet(WORK / f"blockcand_{country}.parquet")
        n_q = s1.filter(is_valid("entity_id")).height
        print(f"{country}: {n_q:,} queries, {tr.height:,} true pairs, {tm:.0f}s")
        for k in k_values:
            r = (hit["brank"] <= k).sum() / tr.height
            print(f"   top-{k:<3} recall {r:.4f}   avg candidates {cand.filter(pl.col('brank') <= k).height / n_q:.1f}")
        print(f"   union of all key types: {hit['brank'].is_not_null().mean():.4f}")
        miss = hit.filter(pl.col("brank").is_null()).head(3000)
        miss.write_parquet(WORK / f"block_miss_{country}.parquet")
        del s1, s23, cand


if __name__ == "__main__":
    blocking_recall()
