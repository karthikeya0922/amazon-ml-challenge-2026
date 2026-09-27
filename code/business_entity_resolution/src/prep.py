"""Step 1: normalize every source file once and cache it as parquet in work/.

    python prep.py
"""
import time

import polars as pl

from config import WORK
from data import scan
from normalize import normalize_frame


def prep_path(split: str, src: str):
    return WORK / f"{split}_{src}.parquet"


def main():
    for split in ("train", "test"):
        for src in ("source1", "source2", "source3"):
            t = time.time()
            out = prep_path(split, src)
            normalize_frame(scan(split, src)).sink_parquet(out)
            print(f"{split}/{src}: {pl.scan_parquet(out).select(pl.len()).collect().item():,} rows in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
