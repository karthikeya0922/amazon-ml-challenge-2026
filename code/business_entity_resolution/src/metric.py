"""Official metric: F0.5 per Source-1 entity, macro-averaged over all Source-1 entities."""
import polars as pl

BETA2 = 0.25


def f05(n_pred: int, n_true: int, n_hit: int) -> float:
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or n_hit == 0:
        return 0.0
    p, r = n_hit / n_pred, n_hit / n_true
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def score(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred/truth: long frames (source1_entity_id, mid). s1_ids: every S1 entity being evaluated.

    Returns macro F0.5 plus pair-level precision/recall for diagnostics.
    """
    base = pl.DataFrame({"source1_entity_id": s1_ids})
    hits = pred.join(truth, on=["source1_entity_id", "mid"], how="inner").group_by("source1_entity_id").len("n_hit")
    per = (
        base.join(pred.group_by("source1_entity_id").len("n_pred"), on="source1_entity_id", how="left")
        .join(truth.group_by("source1_entity_id").len("n_true"), on="source1_entity_id", how="left")
        .join(hits, on="source1_entity_id", how="left")
        .fill_null(0)
    )
    p = per["n_hit"] / per["n_pred"]
    r = per["n_hit"] / per["n_true"]
    f = ((1 + BETA2) * p * r / (BETA2 * p + r)).fill_nan(0.0).fill_null(0.0)
    f = pl.when(per["n_true"] == 0).then((per["n_pred"] == 0).cast(pl.Float64)).otherwise(f)
    per = per.with_columns(f05=pl.select(f).to_series())
    tp, npred, ntrue = per["n_hit"].sum(), per["n_pred"].sum(), per["n_true"].sum()
    return {
        "f05": per["f05"].mean(),
        "pair_precision": tp / max(npred, 1),
        "pair_recall": tp / max(ntrue, 1),
        "n_entities": per.height,
        "per_entity": per,
    }
