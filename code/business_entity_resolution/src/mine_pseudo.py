"""Step 9 (unseen countries): mine vocabulary maps from confident test predictions.

A country with no training labels (France in this test set) has no mined token maps. After a first
prediction pass, its highest-confidence one-owner pairs are used as pseudo-pairs and fed to the same
miner used on training pairs. Only the provided test files are used.

    python mine_pseudo.py [country ...] [--min-p 0.97]   # default: every test country absent from train
"""
import argparse
import json

import polars as pl

from config import WORK
from data import countries
from decide import one_owner
from mine_tokens import mine_country
from prep import prep_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("country", nargs="*")
    ap.add_argument("--min-p", type=float, default=0.97)
    args = ap.parse_args()
    for c in args.country or sorted(set(countries("test")) - set(countries("train"))):
        mine_pseudo(c, args.min_p)


def mine_pseudo(c: str, min_p: float):
    pred = pl.read_parquet(WORK / "test_pred.parquet").filter(pl.col("country") == c)
    conf = one_owner(pred).filter(pl.col("p") >= min_p)
    s1ids = pl.read_parquet(WORK / f"s1ids_test_{c}.parquet").select(pl.col("rid").alias("rid1"), pl.col("entity_id").alias("source1_entity_id"))
    s23ids = pl.read_parquet(WORK / f"s23ids_test_{c}.parquet").select(pl.col("rid").alias("rid2"), pl.col("entity_id").alias("mid"))
    pairs = conf.join(s1ids, on="rid1").join(s23ids, on="rid2").select("source1_entity_id", "mid")
    print(f"{c}: {pairs.height:,} pseudo-pairs with p >= {min_p}")

    s1 = pl.scan_parquet(prep_path("test", "source1")).select(
        pl.col("entity_id").alias("source1_entity_id"), "country", "n_name", "addr_parts")
    other = pl.concat([pl.scan_parquet(prep_path("test", s)) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("mid"), pl.col("n_name").alias("o_name"), pl.col("addr_parts").alias("o_parts"))
    df = pairs.lazy().join(s1, on="source1_entity_id").join(other, on="mid").collect()

    maps_path = WORK / "token_maps.json"
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    maps[c] = mine_country(df, strict_parts=True)
    maps_path.write_text(json.dumps(maps, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in maps[c].items()})


if __name__ == "__main__":
    main()
