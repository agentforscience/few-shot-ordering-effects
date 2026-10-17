"""Extension design: 10 additional demo sets (ids 10-19) per K for the classification
tasks, using a different seed but the same test items. Output: results/design_ext.json"""
import json
import random

import numpy as np

from make_design import CLS_K, CLS_MAX_PERMS, perms_for
from tasks import ROOT, load_cls

SEED, N_EXTRA, FIRST_ID = 1042, 10, 10

base = json.load(open(f"{ROOT}/results/design.json"))
rng, rng_np = random.Random(SEED), np.random.default_rng(SEED)
ext = {"seed": SEED, "cls": {}, "gen": base["gen"]}
for task, td in base["cls"].items():
    tr, _ = load_cls(task)
    conds = []
    for k in CLS_K:
        for s in range(FIRST_ID, FIRST_ID + N_EXTRA):
            demos = [int(i) for i in rng_np.choice(len(tr), size=k, replace=False)]
            conds.append(dict(K=k, set=s, demos=demos, perms=perms_for(k, CLS_MAX_PERMS, rng)))
    ext["cls"][task] = dict(test_idx=td["test_idx"], conds=conds, n_pool=len(tr))
json.dump(ext, open(f"{ROOT}/results/design_ext.json", "w"))
print("ok", {t: len(v["conds"]) for t, v in ext["cls"].items()})
