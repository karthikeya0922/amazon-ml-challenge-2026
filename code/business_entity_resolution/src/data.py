"""Low-memory TSV access via polars lazy scans."""
import polars as pl

from config import DATA, VALID_MOD


def scan(split: str, name: str) -> pl.LazyFrame:
    """Lazily scan dataset/<split>/<split>_<name>.tsv with every column as string."""
    return pl.scan_csv(
        DATA / split / f"{split}_{name}.tsv",
        separator="\t",
        infer_schema=False,
        quote_char=None,
        missing_utf8_is_empty_string=True,
    )


def scan_sources(split: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Return (source1, source2+source3) lazy frames."""
    return scan(split, "source1"), pl.concat([scan(split, "source2"), scan(split, "source3")])


def truth_pairs(split: str = "train") -> pl.LazyFrame:
    """Ground truth exploded to one (source1_entity_id, mid) row per true pair."""
    return (
        scan(split, "ground_truth")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .rename({"matched_entity_ids": "mid"})
    )


def countries(split: str) -> list[str]:
    """Distinct country labels in a split's Source 1 (open set, never hard-coded)."""
    return sorted(scan(split, "source1").select(pl.col("country").unique()).collect()["country"].to_list())


def is_valid(col: str = "source1_entity_id") -> pl.Expr:
    """True for Source-1 entities held out for validation (deterministic hash split)."""
    return pl.col(col).hash(7) % VALID_MOD == 0
