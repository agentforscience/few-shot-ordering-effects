"""Create the shared experimental design (test items, demo sets, permutations).

The same design is used for every model, so differences between models are
not confounded by different demonstrations or orders.
Output: results/design.json
"""
import itertools
import json
import math
import os
import random

import numpy as np

from tasks import ROOT, load_cls, load_gen

SEED = 42
CLS_K = [1, 2, 3, 4, 6, 8]
CLS_SETS, CLS_MAX_PERMS, CLS_N_TEST = 10, 8, 200
GEN_K = [1, 2, 3, 4, 6]
GEN_SETS, GEN_MAX_PERMS, GEN_N_TEST = 6, 6, 80


def perms_for(k, max_p, rng):
    """All permutations if k! <= max_p, else max_p distinct random ones (identity first)."""
    if math.factorial(k) <= max_p:
        return [list(p) for p in itertools.permutations(range(k))]
    out = {tuple(range(k))}
    while len(out) < max_p:
        out.add(tuple(rng.sample(range(k), k)))
    out = sorted(out, key=lambda p: p != tuple(range(k)))
    return [list(p) for p in out]


def balanced_test(df, n, rng_np):
    labs = sorted(df.label.unique())
    per = n // len(labs)
    idx = []
    for l in labs:
        cand = df.index[df.label == l].to_numpy()
        idx += list(rng_np.choice(cand, size=min(per, len(cand)), replace=False))
    if len(idx) < n:  # RTE val is small; top up randomly
        rest = np.setdiff1d(df.index.to_numpy(), idx)
        idx += list(rng_np.choice(rest, size=n - len(idx), replace=False))
    return sorted(int(i) for i in idx)


def main():
    rng = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)
    design = {"seed": SEED, "cls": {}, "gen": {}}
    for task in ["sst2", "rte", "mnli"]:
        tr, te = load_cls(task)
        test_idx = balanced_test(te, CLS_N_TEST, rng_np)
        conds = []
        for k in CLS_K:
            for s in range(CLS_SETS):
                demos = [int(i) for i in rng_np.choice(len(tr), size=k, replace=False)]
                conds.append(dict(K=k, set=s, demos=demos, perms=perms_for(k, CLS_MAX_PERMS, rng)))
        design["cls"][task] = dict(test_idx=test_idx, conds=conds, n_pool=len(tr))
        print(task, "pool", len(tr), "test", len(test_idx), "conds", len(conds),
              "prompts", sum(len(c["perms"]) for c in conds) * len(test_idx))
    tr, te = load_gen()
    test_idx = sorted(int(i) for i in rng_np.choice(len(te), size=GEN_N_TEST, replace=False))
    conds = []
    for k in GEN_K:
        for s in range(GEN_SETS):
            demos = [int(i) for i in rng_np.choice(len(tr), size=k, replace=False)]
            conds.append(dict(K=k, set=s, demos=demos, perms=perms_for(k, GEN_MAX_PERMS, rng)))
    design["gen"]["samsum"] = dict(test_idx=test_idx, conds=conds, n_pool=len(tr))
    print("samsum pool", len(tr), "prompts", sum(len(c["perms"]) for c in conds) * len(test_idx))
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/design.json", "w") as f:
        json.dump(design, f)


if __name__ == "__main__":
    main()
