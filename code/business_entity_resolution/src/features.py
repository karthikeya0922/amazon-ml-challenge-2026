"""Step 4: pair features for (Source-1 record, candidate S2/S3 record).

All features are language-agnostic string/set similarities, so a model trained on US + India
can be applied to an unseen country (France). No country indicator is used as a feature.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

SCORERS = {
    "ratio": fuzz.ratio,
    "tset": fuzz.token_set_ratio,
    "tsort": fuzz.token_sort_ratio,
    "partial": fuzz.partial_ratio,
    "jw": JaroWinkler.normalized_similarity,
    "lev": Levenshtein.normalized_similarity,
}


def _cpdist(a: list[str], b: list[str], scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def string_sims(a: list[str], b: list[str], prefix: str, which=SCORERS) -> dict[str, np.ndarray]:
    return {f"{prefix}_{k}": _cpdist(a, b, f) for k, f in which.items()}


def idf_table(s1: pl.DataFrame, s23: pl.DataFrame, col: str) -> pl.DataFrame:
    """Per-country IDF of the tokens in `col` over Source-1 + S2/S3 records."""
    n = s1.height + s23.height
    toks = pl.concat([s1.select(pl.col(col).str.split(" ").list.unique()), s23.select(pl.col(col).str.split(" ").list.unique())])
    return (
        toks.explode(col).filter(pl.col(col) != "").group_by(col).len("df")
        .select(pl.col(col).alias("tok"), (n / pl.col("df")).log().cast(pl.Float32).alias("idf"))
    )


def weighted_overlap(p: pl.DataFrame, a: str, b: str, idf: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """IDF-weighted coverage: shared/left, shared/right, and the idf of the rarest unmatched S1 token."""
    x = p.select(pl.int_range(pl.len()).alias("pid"), pl.col(a).str.split(" ").list.unique().alias("A"),
                 pl.col(b).str.split(" ").list.unique().alias("B"))

    def wsum(expr_col: str, name: str):
        return (
            x.select("pid", pl.col(expr_col)).explode(expr_col).rename({expr_col: "tok"})
            .join(idf, on="tok", how="left").with_columns(pl.col("idf").fill_null(12.0))
            .group_by("pid").agg(pl.col("idf").sum().alias(name), pl.col("idf").max().alias(name + "_max"))
        )

    x = x.with_columns(S=pl.col("A").list.set_intersection("B"), L=pl.col("A").list.set_difference("B"),
                       R=pl.col("B").list.set_difference("A"))
    out = x.select("pid")
    for c, nm in [("A", "wa"), ("B", "wb"), ("S", "ws"), ("L", "wl"), ("R", "wr")]:
        out = out.join(wsum(c, nm), on="pid", how="left")
    out = out.sort("pid").fill_null(0.0)
    return pl.DataFrame({
        f"{prefix}_wcov1": (out["ws"] / (out["wa"] + 1e-6)).cast(pl.Float32),
        f"{prefix}_wcov2": (out["ws"] / (out["wb"] + 1e-6)).cast(pl.Float32),
        f"{prefix}_wshared": out["ws"].cast(pl.Float32),
        f"{prefix}_wmiss1_max": out["wl_max"].cast(pl.Float32),  # rarest S1 token missing on the other side
        f"{prefix}_wmiss2_max": out["wr_max"].cast(pl.Float32),  # rarest other-side token missing in S1
    })


def number_feats(p: pl.DataFrame) -> pl.DataFrame:
    """House/unit number agreement between the two addresses."""
    nums = lambda c: pl.col(c).str.extract_all(r"\b\d+\b").list.unique()
    x = p.select(n1=nums("a1"), n2=nums("a2"))
    x = x.with_columns(inter=pl.col("n1").list.set_intersection("n2").list.len(),
                       l1=pl.col("n1").list.len(), l2=pl.col("n2").list.len(),
                       first_eq=(pl.col("n1").list.first() == pl.col("n2").list.first()).fill_null(False))
    return x.select(
        num_inter=pl.col("inter").cast(pl.Int16),
        num_jacc=(pl.col("inter") / (pl.col("l1") + pl.col("l2") - pl.col("inter"))).fill_nan(0.0).fill_null(0.0).cast(pl.Float32),
        num_conflict=((pl.col("l1") > 0) & (pl.col("l2") > 0) & (pl.col("inter") == 0)).cast(pl.Int8),
        num_first_eq=pl.col("first_eq").cast(pl.Int8),
        num_n1=pl.col("l1").cast(pl.Int8), num_n2=pl.col("l2").cast(pl.Int8),
    )


AKEY_RE = r"\b\d+ [^\d\s]+"  # first "house-number street-word" of an address


def s1_stats(s1: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Country-level Source-1 statistics used to judge how ambiguous a name / address is."""
    return {
        "name_cnt": s1.group_by("n_name").len("cnt"),
        "akey_cnt": s1.select(akey=pl.col("n_addr").str.extract(AKEY_RE, 0)).drop_nulls().group_by("akey").len("cnt"),
        "vocab": s1.select(tok=pl.col("n_name").str.split(" ")).explode("tok").unique().with_columns(in_s1=pl.lit(1, pl.Int8)),
    }


def ambiguity_feats(p: pl.DataFrame, stats: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """How many Source-1 records share each side's exact name / address key, and name OOV rate."""
    nc, ac = stats["name_cnt"], stats["akey_cnt"]
    x = p.select("n1", "n2", k1=pl.col("a1").str.extract(AKEY_RE, 0), k2=pl.col("a2").str.extract(AKEY_RE, 0)).with_row_index("pid")
    x = (
        x.join(nc.rename({"n_name": "n1", "cnt": "n1_s1cnt"}), on="n1", how="left")
        .join(nc.rename({"n_name": "n2", "cnt": "n2_s1cnt"}), on="n2", how="left")
        .join(ac.rename({"akey": "k1", "cnt": "a1_s1cnt"}), on="k1", how="left")
        .join(ac.rename({"akey": "k2", "cnt": "a2_s1cnt"}), on="k2", how="left")
    )
    oov = (
        x.select("pid", tok=pl.col("n2").str.split(" ")).explode("tok")
        .join(stats["vocab"], on="tok", how="left")
        .group_by("pid").agg(n2_oov=1 - pl.col("in_s1").fill_null(0).mean())
    )
    x = x.join(oov, on="pid", how="left").sort("pid")
    return x.select(
        n1_s1cnt=pl.col("n1_s1cnt").fill_null(0).cast(pl.Int32),
        n2_s1cnt=pl.col("n2_s1cnt").fill_null(0).cast(pl.Int32),
        a1_s1cnt=pl.col("a1_s1cnt").fill_null(0).cast(pl.Int32),
        a2_s1cnt=pl.col("a2_s1cnt").fill_null(0).cast(pl.Int32),
        akey_eq=(pl.col("k1") == pl.col("k2")).fill_null(False).cast(pl.Int8),
        n2_oov=pl.col("n2_oov").fill_null(1.0).cast(pl.Float32),
    )


def number_close(p: pl.DataFrame) -> dict[str, np.ndarray]:
    """Similarity of the first numbers of both addresses (catches 989 vs 89, 447 vs 448)."""
    x = p.select(d1=pl.col("a1").str.extract(r"\b(\d+)\b", 1).fill_null(""), d2=pl.col("a2").str.extract(r"\b(\d+)\b", 1).fill_null(""))
    d1, d2 = x["d1"].to_list(), x["d2"].to_list()
    num = x.select(
        pl.col("d1").cast(pl.Int64, strict=False).alias("i1"), pl.col("d2").cast(pl.Int64, strict=False).alias("i2"))
    diff = (num["i1"] - num["i2"]).abs().fill_null(-1).cast(pl.Int64).to_numpy()
    return {
        "num1_lev": _cpdist(d1, d2, Levenshtein.normalized_similarity),
        "num1_absdiff": np.where(diff < 0, -1, np.minimum(diff, 100000)).astype(np.float32),
    }


def build_features(cand: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame,
                   name_idf: pl.DataFrame, addr_idf: pl.DataFrame, stats: dict | None = None) -> pl.DataFrame:
    """cand: (rid1, rid2, bscore, brank). Returns cand + feature columns (same row order)."""
    p = (
        cand.join(s1.select(pl.col("rid").alias("rid1"), pl.col("n_name").alias("n1"), pl.col("n_addr").alias("a1"),
                            pl.col("addr_parts").alias("p1")), on="rid1", how="left")
        .join(s23.select(pl.col("rid").alias("rid2"), pl.col("entity_id").alias("mid"), pl.col("n_name").alias("n2"),
                         pl.col("n_addr").alias("a2"), pl.col("addr_parts").alias("p2")), on="rid2", how="left")
    )
    n1, n2, a1, a2 = (p[c].to_list() for c in ("n1", "n2", "a1", "a2"))
    feats = {}
    feats.update(string_sims(n1, n2, "name"))
    feats.update(string_sims(a1, a2, "addr", {k: SCORERS[k] for k in ("ratio", "tset", "tsort", "partial", "jw")}))
    # names written without spaces (web-domain / hashtag variants): compare concatenations
    ns1 = [s.replace(" ", "") for s in n1]
    ns2 = [s.replace(" ", "") for s in n2]
    feats["name_nospace_ratio"] = _cpdist(ns1, ns2, fuzz.ratio)
    feats["name_nospace_partial"] = _cpdist(ns1, ns2, fuzz.partial_ratio)
    feats.update(number_close(p))
    del n1, n2, a1, a2, ns1, ns2

    f = pl.DataFrame(feats)
    tok = p.select(
        name_jacc=(pl.col("n1").str.split(" ").list.set_intersection(pl.col("n2").str.split(" ")).list.len()
                   / pl.col("n1").str.split(" ").list.set_union(pl.col("n2").str.split(" ")).list.len()).cast(pl.Float32),
        addr_jacc=(pl.col("a1").str.split(" ").list.set_intersection(pl.col("a2").str.split(" ")).list.len()
                   / pl.col("a1").str.split(" ").list.set_union(pl.col("a2").str.split(" ")).list.len()).fill_nan(0.0).cast(pl.Float32),
        part_inter=pl.col("p1").list.set_intersection("p2").list.len().cast(pl.Int8),
        part_cov2=(pl.col("p1").list.set_intersection("p2").list.len() / pl.col("p2").list.len()).fill_nan(0.0).cast(pl.Float32),
        n_len1=pl.col("n1").str.len_chars().cast(pl.Int16), n_len2=pl.col("n2").str.len_chars().cast(pl.Int16),
        n_tok2=pl.col("n2").str.count_matches(" ").cast(pl.Int8),
        a_len2=pl.col("a2").str.len_chars().cast(pl.Int16),
        a_empty2=(pl.col("a2") == "").cast(pl.Int8),
        n_parts2=pl.col("p2").list.len().cast(pl.Int8),
        nonlatin2=pl.col("n2").str.contains(r"[^\p{Latin}\p{N}\s&+]").cast(pl.Int8),
        is_s3=pl.col("mid").str.starts_with("S3").cast(pl.Int8),
        name_exact=(pl.col("n1") == pl.col("n2")).cast(pl.Int8),
    )
    wn = weighted_overlap(p, "n1", "n2", name_idf, "name")
    wa = weighted_overlap(p, "a1", "a2", addr_idf, "addr")
    nf = number_feats(p)
    parts = [cand, p.select("mid"), f, tok, wn, wa, nf]
    if stats is not None:
        parts.append(ambiguity_feats(p, stats))
    return pl.concat(parts, how="horizontal")


def pair_quality(cand: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.Series:
    """Cheap pair score used for competition features: name tset + addr tset + 100 * number jaccard."""
    p = cand.select("rid1", "rid2").join(
        s1.select(pl.col("rid").alias("rid1"), pl.col("n_name").alias("n1"), pl.col("n_addr").alias("a1")), on="rid1", how="left"
    ).join(s23.select(pl.col("rid").alias("rid2"), pl.col("n_name").alias("n2"), pl.col("n_addr").alias("a2")), on="rid2", how="left")
    q = _cpdist(p["n1"].to_list(), p["n2"].to_list(), fuzz.token_set_ratio) +         _cpdist(p["a1"].to_list(), p["a2"].to_list(), fuzz.token_set_ratio)
    return pl.Series("pair_q", q) + 100 * number_feats(p)["num_jacc"]


def context_features(cand: pl.DataFrame) -> pl.DataFrame:
    """cand: (rid1, rid2, bscore, brank, pair_q) over ALL Source-1 queries -> competition features."""
    return cand.with_columns(
        q_rank1=pl.col("pair_q").rank("ordinal", descending=True).over("rid1").cast(pl.Int16),
        q_max1=pl.col("pair_q").max().over("rid1"),
        q_rank2=pl.col("pair_q").rank("ordinal", descending=True).over("rid2").cast(pl.Int16),
        q_max2=pl.col("pair_q").max().over("rid2"),
        n_cand1=pl.len().over("rid1").cast(pl.Int16),
        n_cand2=pl.len().over("rid2").cast(pl.Int16),
        b_max1=pl.col("bscore").max().over("rid1"),
        b_rank2=pl.col("bscore").rank("ordinal", descending=True).over("rid2").cast(pl.Int16),
        n_strong1=(pl.col("pair_q") >= 150).sum().over("rid1").cast(pl.Int16),
    ).with_columns(
        q_gap1=(pl.col("pair_q") - pl.col("q_max1")).cast(pl.Float32),
        q_gap2=(pl.col("pair_q") - pl.col("q_max2")).cast(pl.Float32),  # 0 when this S1 is the candidate's best S1
        b_rel1=(pl.col("bscore") / pl.col("b_max1")).cast(pl.Float32),
        q_second2=pl.col("pair_q").sort(descending=True).get(1, null_on_oob=True).over("rid2").fill_null(0.0).cast(pl.Float32),
    ).drop("q_max1", "q_max2", "b_max1")
