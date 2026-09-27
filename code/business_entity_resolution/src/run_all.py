"""Run the whole pipeline end-to-end (data -> blocking -> matching -> output).

    python run_all.py              # everything
    python run_all.py --from train # resume from a step
"""
import argparse
import os
import subprocess
import sys
import time

# (name, command, extra environment)
STEPS = [
    ("prep", ["prep.py"], {}),
    ("mine", ["mine_tokens.py"], {}),
    ("ranker", ["block_ranker.py"], {}),
    ("cand_train", ["pipeline.py", "train"], {"CAND_ONLY": "1"}),       # forward + reverse blocking, context
    ("prune", ["prune.py"], {}),                                          # learned pruner on those candidates
    ("feat_train", ["pipeline.py", "train"], {"REUSE_CAND": "1"}),      # features of the pruned candidates
    ("train", ["train.py"], {}),                                          # stage 1, out-of-fold
    ("s2_train_feat", ["stage2.py", "build", "train"], {}),
    ("s2_train", ["stage2.py", "train"], {}),                             # stage 2, out-of-fold + decision rule
    ("pipe_test", ["pipeline.py", "test"], {}),
    ("s1_test", ["predict.py", "stage1"], {}),
    ("s2_test_feat", ["stage2.py", "build", "test"], {}),
    ("predict1", ["predict.py", "final"], {}),
    ("pseudo", ["mine_pseudo.py"], {}),                                   # maps for countries without labels
    ("pipe_unseen", ["pipeline.py", "test", "unseen"], {}),
    ("s1_test2", ["predict.py", "stage1"], {}),
    ("s2_test_feat2", ["stage2.py", "build", "test"], {}),
    ("predict2", ["predict.py", "final"], {}),
    ("adapt", ["adapt.py"], {}),                                          # self-training for countries without labels
    ("s2_test_feat3", ["stage2.py", "build", "test"], {}),
    ("predict3", ["predict.py", "final"], {}),
    ("adapt2", ["adapt.py"], {}),                                         # second round: pseudo-labels from round 1
    ("s2_test_feat4", ["stage2.py", "build", "test"], {}),
    ("predict", ["predict.py", "final"], {}),
    # tried and dropped: "adapt.py stage2" (stage 2 self-trained on the same pseudo-labels) lowered the
    # leaderboard score from 0.977619 to 0.977481 (France only changed: about -0.001 F0.5 on France)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _, _ in STEPS])
    ap.add_argument("--to", dest="end", default=STEPS[-1][0], choices=[s for s, _, _ in STEPS])
    args = ap.parse_args()
    names = [s for s, _, _ in STEPS]
    for name, cmd, env in STEPS[names.index(args.start):names.index(args.end) + 1]:
        t = time.time()
        print(f"=== {name}: {' '.join(f'{k}={v}' for k, v in env.items())} python {' '.join(cmd)}", flush=True)
        subprocess.run([sys.executable, "-u", "-W", "ignore", *cmd], check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8", **env})
        print(f"=== {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
