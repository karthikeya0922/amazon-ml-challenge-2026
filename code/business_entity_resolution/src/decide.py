"""Step 6: turn pair probabilities into per-Source-1 match lists.

1. One owner: every S2/S3 record is kept only for the Source-1 record with its highest probability
   (in the training ground truth each S2/S3 record matches at most one Source-1 entity).
2. Per Source-1 entity, choose how many of its (sorted) candidates to output by maximizing an
   estimate of expected F0.5. With F0.5 = 1.25*TP / (0.25*n_true + n_pred):
       k = 0  ->  E[F] = P(no true match) = prod(1 - p_i)
       k >= 1 ->  E[F] ~ 1.25 * sum_{i<=k} p_i / (0.25 * (sum_all p_i + miss) + k)
   `miss` accounts for true matches that blocking never produced.
"""
import numpy as np
import polars as pl


def one_owner(pairs: pl.DataFrame, prob: str = "p") -> pl.DataFrame:
    best = pl.col(prob).rank("ordinal", descending=True).over("rid2") == 1
    return pairs.filter(best)


def choose(pairs: pl.DataFrame, prob: str = "p", miss: float = 0.0, scale: float = 1.0,
           min_p: float = 0.0) -> pl.DataFrame:
    """Return the selected (rid1, rid2) pairs. `scale` multiplies probabilities (calibration knob)."""
    d = (
        pairs.select("rid1", "rid2", (pl.col(prob) * scale).clip(0.0, 1.0).alias("q"))
        .sort(["rid1", "q"], descending=[False, True])
        .with_columns(
            k=pl.int_range(1, pl.len() + 1).over("rid1"),
            cum=pl.col("q").cum_sum().over("rid1"),
            tot=pl.col("q").sum().over("rid1"),
            p_none=(1 - pl.col("q")).log().sum().over("rid1").exp(),
        )
        .with_columns(ef=1.25 * pl.col("cum") / (0.25 * (pl.col("tot") + miss) + pl.col("k")))
    )
    best = d.group_by("rid1").agg(pl.col("ef").max().alias("ef_best"), pl.col("p_none").first())
    d = d.join(best, on="rid1").with_columns(
        kbest=pl.col("k").filter(pl.col("ef") == pl.col("ef_best")).first().over("rid1")
    )
    return d.filter((pl.col("ef_best") > pl.col("p_none")) & (pl.col("k") <= pl.col("kbest")) & (pl.col("q") >= min_p)) \
            .select("rid1", "rid2")


def choose_threshold(pairs: pl.DataFrame, thr: float, prob: str = "p") -> pl.DataFrame:
    return pairs.filter(pl.col(prob) >= thr).select("rid1", "rid2")
