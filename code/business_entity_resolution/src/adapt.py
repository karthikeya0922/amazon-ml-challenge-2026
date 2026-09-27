"""Step 10 (unseen countries): self-training of the stage-1 models on confident test predictions.

A country without training labels is scored by models that never saw its data. On a simulated
unseen country (train on India only, validate on US) adding the model's own confident predictions
on the new country as pseudo-labels and retraining recovered +0.006 F0.5 after one round and +0.008
after two. Here: pseudo-positives are one-owner pairs with p >= POS_P, pseudo-negatives pairs with
p <= NEG_P (p = current final probability); both stage-1 fold models are retrained on their usual
training sample plus all pseudo-labelled pairs, and the unseen countries' stage-1 probabilities in
work/stage1_test.parquet are replaced. Re-run `stage2.py build test` and `predict.py final` after.

    python adapt.py
(Tried and dropped: self-training stage 2 on the same pseudo-labels lowered the leaderboard score.)
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import WORK
from data import countries, is_valid
from decide import one_owner
from train import N_FOLDS, PARAMS, TRAIN_FRAC, fold_expr, load, parts, to_x

POS_P, NEG_P = 0.95, 0.05


def main():
    t = time.time()
    unseen = sorted(set(countries("test")) - set(countries("train")))
    if not unseen:
        print("no unseen countries")
        return
    base = [lgb.Booster(model_file=str(WORK / f"lgb_s1_f{f}.txt")) for f in range(N_FOLDS)]
    feats = base[0].feature_name()
    pred = pl.read_parquet(WORK / "test_pred.parquet").filter(pl.col("country").is_in(unseen))
    owners = pl.concat([one_owner(g) for _, g in pred.group_by("country")]).select("country", "rid1", "rid2", own=pl.lit(True))
    lab = (pred.join(owners, on=["country", "rid1", "rid2"], how="left")
           .with_columns(pl.col("own").fill_null(False))
           .with_columns(pl=pl.when(pl.col("own") & (pl.col("p") >= POS_P)).then(1).when(pl.col("p") <= NEG_P).then(0))
           .filter(pl.col("pl").is_not_null()).select("country", "rid1", "rid2", pl.col("pl").cast(pl.Int8)))
    pseudo = pl.concat([pl.read_parquet(p, columns=["country", "rid1", "rid2", *feats]).join(lab, on=["country", "rid1", "rid2"])
                        for p in parts("test") if any(f"_{c}_" in p.name for c in unseen)])
    print(f"pseudo-labels for {unseen}: {pseudo.height:,} pairs ({pseudo['pl'].sum():,} positive) of {pred.height:,} ({time.time() - t:.0f}s)", flush=True)

    models = []
    for f in range(N_FOLDS):
        tr = load("train", ~is_valid() & (fold_expr() == f) & (pl.col("source1_entity_id").hash(3) % 10 < TRAIN_FRAC), feats + ["label"])
        x = np.concatenate([to_x(tr, feats), to_x(pseudo, feats)])
        y = np.concatenate([tr["label"].to_numpy(), pseudo["pl"].to_numpy()])
        del tr
        m = lgb.train(PARAMS, lgb.Dataset(x, y, feature_name=feats), num_boost_round=base[f].current_iteration())
        m.save_model(str(WORK / f"lgb_s1_adapt_f{f}.txt"))
        models.append(m)
        del x, y
        print(f"adapted fold {f} ({time.time() - t:.0f}s)", flush=True)

    out = []
    for p in parts("test"):
        if not any(f"_{c}_" in p.name for c in unseen):
            continue
        d = pl.read_parquet(p, columns=["country", "rid1", "rid2", *feats])
        out.append(d.select("country", "rid1", "rid2",
                            p1=pl.Series(np.mean([m.predict(to_x(d, feats)) for m in models], axis=0), dtype=pl.Float32)))
    new = pl.concat(out)
    s1p = pl.read_parquet(WORK / "stage1_test.parquet").filter(~pl.col("country").is_in(unseen))
    pl.concat([s1p, new]).write_parquet(WORK / "stage1_test.parquet")
    print(f"replaced stage-1 probabilities of {new.height:,} unseen-country pairs ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
