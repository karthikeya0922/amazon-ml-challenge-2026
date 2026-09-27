"""Step 2: mine vocabulary mappings from matched training pairs (no external dictionaries).

For every true pair (S1 record, S2/S3 record) we look at the tokens that appear on only one
side. If variant token b (S2/S3 side) co-occurs with S1 token a far more often than chance,
b -> a becomes a normalization rule. This recovers abbreviations (rd -> road), state codes
(texas -> tx, mh -> maharashtra), and native-script words (लिमिटेड -> limited) purely from data.

Three maps are mined, in order:
  part_map  : whole address components ("north carolina" -> "nc")
  addr_map  : address tokens, after part_map is applied
  name_map  : business-name tokens

    python mine_tokens.py
"""
import json
import time

import polars as pl

from config import WORK
from data import is_valid, truth_pairs
from normalize import apply_part_map
from prep import prep_path

SAMPLE_MOD = 2  # use S1 entities with hash % SAMPLE_MOD == 0 (~50% of training pairs)


def load_pairs() -> pl.DataFrame:
    pairs = (
        truth_pairs("train")
        .filter((pl.col("source1_entity_id").hash(11) % SAMPLE_MOD == 0) & ~is_valid())
        .collect()
    )
    s1 = (
        pl.scan_parquet(prep_path("train", "source1"))
        .select(pl.col("entity_id").alias("source1_entity_id"), "country", "n_name", "addr_parts")
        .filter(pl.col("source1_entity_id").is_in(pairs["source1_entity_id"].implode()))
    )
    other = (
        pl.concat([pl.scan_parquet(prep_path("train", s)) for s in ("source2", "source3")])
        .select(pl.col("entity_id").alias("mid"), pl.col("n_name").alias("o_name"), pl.col("addr_parts").alias("o_parts"))
        .filter(pl.col("mid").is_in(pairs["mid"].implode()))
    )
    return pairs.lazy().join(s1, on="source1_entity_id").join(other, on="mid").collect()


def mine(df: pl.DataFrame, s_col: str, o_col: str, max_diff: int, min_support: int, min_conf: float,
         token_filter: pl.Expr | None = None) -> pl.DataFrame:
    """Mine variant(b) -> canonical(a) from list columns s_col (S1 side) and o_col (other side)."""
    d = df.select(
        pl.col(s_col).list.unique().alias("s"), pl.col(o_col).list.unique().alias("o")
    ).with_row_index("pid")
    if token_filter is not None:
        d = d.with_columns(pl.col("s").list.eval(pl.element().filter(token_filter)),
                           pl.col("o").list.eval(pl.element().filter(token_filter)))
    d = d.with_columns(A=pl.col("s").list.set_difference("o"), B=pl.col("o").list.set_difference("s"))

    # how often each token appears on the S1 side / the other side at all
    f1 = d.select(pl.col("s").explode().alias("a")).group_by("a").len("f1_a")
    fo = d.select(pl.col("o").explode().alias("b")).group_by("b").len("fo_b")
    nb = d.select(pl.col("B").explode().alias("b")).group_by("b").len("nb_b")
    f1b = f1.rename({"a": "b", "f1_a": "f1_b"})

    cand = (
        d.filter(pl.col("A").list.len().is_between(1, max_diff) & pl.col("B").list.len().is_between(1, max_diff))
        .select("pid", pl.col("A").alias("a"), pl.col("B").alias("b"))
        .explode("a").explode("b")
        .group_by("a", "b").len("c_ab")
    )
    res = (
        cand.join(fo, on="b").join(nb, on="b").join(f1, on="a").join(f1b, on="b", how="left")
        .with_columns(pl.col("f1_b").fill_null(0), conf=pl.col("c_ab") / pl.col("nb_b"), gconf=pl.col("c_ab") / pl.col("fo_b"))
        .filter((pl.col("c_ab") >= min_support) & (pl.col("conf") >= min_conf) & (pl.col("gconf") >= 0.2)
                & (pl.col("f1_b") < pl.col("f1_a")))
        .sort("c_ab", descending=True)
        .unique("b", keep="first", maintain_order=True)
    )
    return res


def resolve(mapping: dict[str, str]) -> dict[str, str]:
    """Follow chains b -> a -> a' to a fixed point; drop cycles."""
    out = {}
    for b in mapping:
        seen, cur = {b}, mapping[b]
        while cur in mapping and cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        if cur != b:
            out[b] = cur
    return out


def _uncovered_parts(df: pl.DataFrame, parts: str, other_parts: str) -> pl.Series:
    """Components of `parts` whose words are not all present somewhere in `other_parts` (same row).

    A component such as "rue x" is not really missing from the other record when that record has
    "58 rue x": the two only split the address differently, so it must not be mined as a synonym.
    """
    x = df.select(pl.int_range(pl.len()).alias("_i"), pl.col(parts).alias("part"),
                  pl.col(other_parts).list.join(" ").str.split(" ").alias("ow"))
    kept = (
        x.explode("part").drop_nulls("part")
        .filter(pl.col("part").str.split(" ").list.set_difference("ow").list.len() > 0)
        .group_by("_i", maintain_order=True).agg("part")
    )
    return x.select("_i").join(kept, on="_i", how="left").sort("_i")["part"].fill_null([]).alias(parts)


def mine_country(df: pl.DataFrame, strict_parts: bool = False) -> dict[str, dict[str, str]]:
    """Mine part_map, then addr_map (after part_map), then name_map for one country's pairs.

    strict_parts: ignore components whose words all occur in the other record (used for pseudo-pairs,
    where a small sample otherwise yields street -> region rules).
    """
    no_digit = ~pl.element().str.contains(r"\d")
    pdf = df
    if strict_parts:
        pdf = df.with_columns(_uncovered_parts(df, "addr_parts", "o_parts"), _uncovered_parts(df, "o_parts", "addr_parts"))
    parts = mine(pdf, "addr_parts", "o_parts", max_diff=4, min_support=15, min_conf=0.5, token_filter=no_digit)
    part_map = resolve(dict(zip(parts["b"], parts["a"])))

    df = df.with_columns(apply_part_map("addr_parts", part_map), apply_part_map("o_parts", part_map)).with_columns(
        s_addr=pl.col("addr_parts").list.join(" ").str.split(" "),
        o_addr=pl.col("o_parts").list.join(" ").str.split(" "),
        s_name=pl.col("n_name").str.split(" "),
        o_nm=pl.col("o_name").str.split(" "),
    )
    addr = mine(df, "s_addr", "o_addr", max_diff=6, min_support=15, min_conf=0.5, token_filter=no_digit)
    name = mine(df, "s_name", "o_nm", max_diff=6, min_support=4, min_conf=0.5)
    # 1-2 letter fragments ("th" -> "health", "at" -> "atlantic") are too ambiguous to rewrite
    name = name.filter(pl.col("b").str.len_chars() >= 3)
    aligned = align_scripts(df)
    for label, t in [("parts", parts), ("addr", addr), ("name", name), ("aligned", aligned)]:
        print(f"  {label}: {t.height} rules; top: {list(zip(t['b'].head(25), t['a'].head(25)))}")
    name_map = dict(zip(name["b"], name["a"]))
    name_map.update(dict(zip(aligned["b"], aligned["a"])))  # word-by-word alignment wins for other scripts
    return {
        "part_map": part_map,
        "addr_map": resolve(dict(zip(addr["b"], addr["a"]))),
        "name_map": resolve(name_map),
    }



OTHER_SCRIPT = r"[^\p{Latin}\p{N}\p{P}\p{S}\s]"  # a letter from a non-Latin script


def align_scripts(df: pl.DataFrame, min_support: int = 3, min_purity: float = 0.8) -> pl.DataFrame:
    """Word-by-word dictionary for names written in another script (b -> a), from aligned true pairs.

    A name written in another script is a word-by-word rendering of the Latin one ("jai engineering
    private limited" <-> "जय इंजीनियरिंग प्राइवेट लिमिटेड"), so when both names have the same number of
    words the i-th words correspond. Co-occurrence mining cannot separate words that always appear
    together ("private" / "limited") and skips short words; positional alignment can.
    """
    d = df.select(A=pl.col("n_name").str.split(" "), B=pl.col("o_name").str.split(" ")).filter(
        (pl.col("A").list.len() == pl.col("B").list.len()) & pl.col("B").list.join(" ").str.contains(OTHER_SCRIPT))
    al = (
        d.explode("A", "B").rename({"A": "a", "B": "b"})
        .filter(pl.col("b").str.contains(OTHER_SCRIPT) & ~pl.col("a").str.contains(OTHER_SCRIPT) & (pl.col("a") != ""))
        .group_by("b", "a").len("c")
        .with_columns(tot=pl.col("c").sum().over("b"))
        .sort("c", descending=True).unique("b", keep="first", maintain_order=True)
    )
    return al.filter((pl.col("c") >= min_support) & (pl.col("c") / pl.col("tot") >= min_purity)).sort("tot", descending=True)


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # rules include non-Latin scripts
    t = time.time()
    df = load_pairs()
    print(f"pairs loaded: {df.height:,} in {time.time() - t:.0f}s")
    path = WORK / "token_maps.json"
    # keep maps of countries without training labels (mined from test predictions by mine_pseudo.py)
    maps = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for country in sorted(df["country"].unique().to_list()):
        print(f"== {country}")
        maps[country] = mine_country(df.filter(pl.col("country") == country))
    path.write_text(json.dumps(maps, ensure_ascii=False), encoding="utf-8")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
