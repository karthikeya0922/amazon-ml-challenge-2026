"""Live progress view for a running pipeline (read-only; safe to start/stop any time).

    python progress.py            # refreshes every 5 s, Ctrl+C to exit
    python progress.py train      # which split's log to follow (default: train)
"""
import ctypes
import os
import sys
import time

import polars as pl

from blocking import CHUNK, KEY_TYPES
from config import WORK
from prep import prep_path

LOGS = {"train": "full_train_log.txt", "test": "full_test_log.txt"}


def free_ram_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS()
    m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2**30


def bar(done: int, total: int, width: int = 30) -> str:
    frac = min(done / total, 1.0) if total else 0.0
    return "[" + "#" * int(frac * width) + "." * (width - int(frac * width)) + f"] {done}/{total}"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    log = WORK / LOGS[split]
    sizes = dict(pl.scan_parquet(prep_path(split, "source1")).group_by("country").len().collect().iter_rows())
    start = time.time()
    while True:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        done_countries = [l.split("]")[0][1:] for l in lines if l.startswith("[") and "features done" in l]
        current = next((c for c in sorted(sizes) if c not in done_countries), None)
        spill = list((WORK / "spill").glob("*.parquet")) if (WORK / "spill").exists() else []
        feats = list(WORK.glob(f"feat_{split}_*.parquet"))

        os.system("cls" if os.name == "nt" else "clear")
        print(f"Pipeline progress ({split})   {time.strftime('%H:%M:%S')}   watching {int(time.time() - start)}s   free RAM {free_ram_gb():.1f} GB\n")
        for c in sorted(sizes):
            state = "done" if c in done_countries else ("running" if c == current else "waiting")
            print(f"  {c:<8} {sizes[c]:>10,} Source-1 records   {state}")
        if current:
            n_blocks = sizes[current] // CHUNK + 1
            print(f"\n  Blocking {current} (spilled blocks per key type):")
            for name in KEY_TYPES:
                n = sum(1 for f in spill if f.name.startswith(name + "_"))
                print(f"    {name:<9} {bar(n, n_blocks)}")
        print(f"\n  Feature parts written: {len(feats)}")
        print("\n  Last log lines:")
        for l in lines[-8:]:
            print("    " + l[:150])
        if lines and lines[-1].startswith("exit"):
            print("\n  Finished.")
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
