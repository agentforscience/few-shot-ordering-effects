"""Analysis of ordering effects. Reads results/raw/*_{cls,gen}.parquet.

Writes tables to results/tables/*.csv, a summary JSON (results/summary.json)
and figures to figures/.

Key quantities (per model x task x K x demo set):
  sd      : SD (ddof=1) of accuracy across the evaluated permutations
  range   : best - worst permutation accuracy
  pairgap : mean |acc(pi) - acc(pi')| over permutation pairs (unbiased in #perms)
  flip    : mean fraction of test items whose prediction differs between two orders
Position influence (per model x task x K):
  beta_i  : effect on the model's (class-centred) label log-prob of "demo at
            position i has that label", with test-item x class fixed effects.
            Bootstrap CIs resample demo sets.
"""
import glob
import itertools
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, TAB, FIG = f"{ROOT}/results/raw", f"{ROOT}/results/tables", f"{ROOT}/figures"
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
SIZES = {"Qwen2.5-0.5B-Instruct": 0.5, "Qwen2.5-1.5B-Instruct": 1.5, "Qwen2.5-3B-Instruct": 3.0,
         "Qwen2.5-7B-Instruct": 7.0, "Qwen2.5-14B-Instruct": 14.0}
RNG = np.random.default_rng(0)
NBOOT = 2000
TASK_ORDER = ["sst2", "rte", "mnli", "samsum"]


def boot_ci(x, f=np.mean, n=NBOOT):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    b = [f(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


# ------------------------------------------------------------------ loading
def load():
    """Main runs (+ extension demo sets 10-19, files *_ext_cls.parquet) and generation runs."""
    cls, gen = [], []
    for m in SIZES:
        for suffix in ["", "_ext"]:
            f = f"{RAW}/{m}{suffix}_cls.parquet"
            if os.path.exists(f):
                d = pd.read_parquet(f)
                if suffix:
                    d = d[d.K > 0]  # zero-shot rows duplicated in the extension run
                d["model"] = m
                cls.append(d)
        f = f"{RAW}/{m}_gen.parquet"
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d["model"] = m
            gen.append(d)
    cls = pd.concat(cls, ignore_index=True)
    gen = pd.concat(gen, ignore_index=True) if gen else None
    cls["size"] = cls.model.map(SIZES)
    return cls, gen


def load_probes():
    pc, pg = [], []
    for m in SIZES:
        f = f"{RAW}/{m}_probe_cls.parquet"
        if os.path.exists(f):
            d = pd.read_parquet(f)
            d["model"] = m
            pc.append(d)
        f = f"{RAW}/{m}_probe_gen.parquet"
        if os.path.exists(f):
            d = pd.read_parquet(f)
            if "marker_logp" in d:  # skip the superseded binary-marker version
                d["model"] = m
                pg.append(d)
    return (pd.concat(pc, ignore_index=True) if pc else None,
            pd.concat(pg, ignore_index=True) if pg else None)


def _ratio_rows(model, task, K, per_set, num, den, name):
    """per_set: DataFrame index=set, columns=positions(1..K) of mean influence. Bootstrap over sets."""
    sets = per_set.index.values
    est = per_set.mean()
    boots = np.array([per_set.loc[RNG.choice(sets, len(sets))].mean().values for _ in range(NBOOT)])
    rows = []
    for i, pos in enumerate(per_set.columns):
        rows.append(dict(model=model, task=task, K=K, pos=int(pos), beta=est[pos],
                         lo=np.percentile(boots[:, i], 2.5), hi=np.percentile(boots[:, i], 97.5)))
    a, b = list(per_set.columns).index(num), list(per_set.columns).index(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = boots[:, a] / boots[:, b]
    rows.append(dict(model=model, task=task, K=K, pos=name, beta=est[num] / est[den],
                     lo=np.percentile(rb, 2.5), hi=np.percentile(rb, 97.5),
                     p_boot=float(np.mean(boots[:, a] - boots[:, b] <= 0))))
    return rows


def probe_cls_analysis(pc, cls):
    """Label-flip probe: influence = increase of log P(new label) when demo i's label is flipped."""
    base = cls[cls.K.isin([4, 8]) & (cls.perm < 2) & (cls.set < 10)][
        ["model", "task", "K", "set", "perm", "test_id", "scores", "pred"]].rename(
        columns={"scores": "base_scores", "pred": "base_pred"})
    m = pc.merge(base, on=["model", "task", "K", "set", "perm", "test_id"], how="inner")
    assert len(m) == len(pc), (len(m), len(pc))
    new_sc = np.array([json.loads(s)[n] for s, n in zip(m.scores, m.new_label)])
    old_sc = np.array([json.loads(s)[n] for s, n in zip(m.base_scores, m.new_label)])
    m["d_logp"] = np.maximum(new_sc, -30) - np.maximum(old_sc, -30)
    m["d_pred"] = (m.pred == m.new_label).astype(float) - (m.base_pred == m.new_label).astype(float)
    m["pos1"] = m.pos + 1
    rows, rows_pred = [], []
    for (model, task, K), g in m.groupby(["model", "task", "K"]):
        ps = g.groupby(["set", "pos1"]).d_logp.mean().unstack()
        rows += _ratio_rows(model, task, K, ps, K, 1, "last/first")
        pp = g.groupby(["set", "pos1"]).d_pred.mean().unstack() * 100
        rows_pred += _ratio_rows(model, task, K, pp, K, 1, "last/first")
    return pd.DataFrame(rows), pd.DataFrame(rows_pred)


def probe_gen_analysis(pg):
    ctrl = pg[pg.pos == -1][["model", "K", "set", "perm", "test_id", "marker_logp"]].rename(
        columns={"marker_logp": "ctrl"})
    m = pg[pg.pos >= 0].merge(ctrl, on=["model", "K", "set", "perm", "test_id"])
    m["d"] = m.marker_logp - m.ctrl
    m["pos1"] = m.pos + 1
    rows = []
    for (model, K), g in m.groupby(["model", "K"]):
        ps = g.groupby(["set", "pos1"]).d.mean().unstack()
        rows += _ratio_rows(model, "samsum", K, ps, 1, K, "first/last")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ order sensitivity
def order_stats(perm_scores, item_matrix=None):
    """perm_scores: array (P,) of per-order metric; item_matrix: (P, N) predictions."""
    p = np.asarray(perm_scores, float)
    out = dict(mean=p.mean(), sd=p.std(ddof=1) if len(p) > 1 else 0.0,
               range=p.max() - p.min(), n_perm=len(p))
    pairs = list(itertools.combinations(range(len(p)), 2))
    out["pairgap"] = np.mean([abs(p[i] - p[j]) for i, j in pairs]) if pairs else 0.0
    if item_matrix is not None:
        m = np.asarray(item_matrix)
        out["flip"] = np.mean([(m[i] != m[j]).mean() for i, j in pairs]) if pairs else 0.0
    return out


def cls_sensitivity(cls):
    rows = []
    for (model, task, K, s), g in cls[cls.K > 0].groupby(["model", "task", "K", "set"]):
        piv = g.pivot(index="perm", columns="test_id", values="pred")
        acc = g.groupby("perm").correct.mean().loc[piv.index].values * 100
        rows.append(dict(model=model, task=task, K=K, set=s, **order_stats(acc, piv.values)))
    return pd.DataFrame(rows)


def summarize_sens(sens, metric_cols=("sd", "range", "pairgap", "flip")):
    out = []
    for (model, task, K), g in sens.groupby(["model", "task", "K"]):
        r = dict(model=model, task=task, K=K, acc=g["mean"].mean(),
                 across_set_sd=g["mean"].std(ddof=1), n_sets=len(g), n_perm=g.n_perm.iloc[0])
        for c in metric_cols:
            if c in g:
                r[c] = g[c].mean()
                lo, hi = boot_ci(g[c])
                r[c + "_lo"], r[c + "_hi"] = lo, hi
        r["range_max"] = g["range"].max()
        out.append(r)
    return pd.DataFrame(out)


def order_reliability(long, value_col, task_col="task", n_splits=20):
    """Is there a systematically 'good' order, or only item-level noise?

    For each (model, task, K>=3, set): matrix M[perm, item] of per-item scores.
    (a) split-half reliability: Spearman between per-perm means on two random item
        halves (averaged over splits), and Spearman-Brown corrected;
    (b) null range: shuffle the perm assignment independently within each item
        (destroys any order-level structure, keeps per-item variability) and
        recompute best-worst range.
    """
    rows = []
    for (model, task, K, s), g in long[long.K >= 3].groupby(["model", task_col, "K", "set"]):
        M = g.pivot(index="perm", columns="test_id", values=value_col).values.astype(float)
        P, N = M.shape
        rs = []
        for _ in range(n_splits):
            idx = RNG.permutation(N)
            a, b = M[:, idx[: N // 2]].mean(1), M[:, idx[N // 2:]].mean(1)
            if a.std() > 0 and b.std() > 0:
                rs.append(stats.spearmanr(a, b).correlation)
        r = np.nanmean(rs) if rs else np.nan
        null = []
        for _ in range(50):
            Mn = np.take_along_axis(M, RNG.random(M.shape).argsort(0), axis=0)
            mn = Mn.mean(1)
            null.append(mn.max() - mn.min())
        mm = M.mean(1)
        rows.append(dict(model=model, task=task, K=K, set=s, split_half_rho=r,
                         range=(mm.max() - mm.min()), null_range=np.mean(null)))
    return pd.DataFrame(rows)


def noise_floor(cls, gen):
    """Compare run-to-run numerical noise (identical prompts re-run, *_rerun files: K=4,
    sets 0-2, all 8 orders) with order-induced differences on exactly the same conditions.

    For metric m per (set, perm): order diff = O_i - O_j + n_i - n_j, rerun diff = n - n'.
    noise_share = E[rerun diff^2] / E[order diff^2]  (fraction of order variance that is noise).
    """
    rows = []
    for m in SIZES:
        f = f"{RAW}/{m}_rerun_cls.parquet"
        if not os.path.exists(f):
            continue
        r = pd.read_parquet(f)
        base = cls[(cls.model == m) & (cls.K == 4) & (cls.set < 3)]
        k = ["task", "K", "set", "perm", "test_id"]
        mm = r.merge(base, on=k, suffixes=("_r", ""))
        assert len(mm) == len(r), (m, len(mm), len(r))
        for task, g in mm.groupby("task"):
            acc = g.groupby(["set", "perm"])[["correct", "correct_r"]].mean() * 100
            rer = (acc.correct - acc.correct_r).values
            od = []
            for s_, a in acc.groupby(level=0):
                v = a.correct.values
                od += [v[i] - v[j] for i, j in itertools.combinations(range(len(v)), 2)]
            od = np.array(od)
            piv = g.pivot_table(index="perm", columns=["set", "test_id"], values="pred").values
            oflip = np.mean([(piv[i] != piv[j]).mean() for i, j in itertools.combinations(range(len(piv)), 2)])
            rows.append(dict(model=m, task=task, rerun_flip=(g.pred != g.pred_r).mean() * 100,
                             order_flip=oflip * 100, rerun_absgap=np.abs(rer).mean(),
                             order_pairgap=np.abs(od).mean(),
                             noise_share=np.mean(rer ** 2) / np.mean(od ** 2)))
        f = f"{RAW}/{m}_rerun_gen.parquet"
        if gen is not None and os.path.exists(f):
            from rouge_score import rouge_scorer as _rs
            import sys
            sys.path.insert(0, f"{ROOT}/src")
            from tasks import load_gen
            _, te = load_gen()
            sc = _rs.RougeScorer(["rougeL"], use_stemmer=True)
            r = pd.read_parquet(f)
            r["rougeL_r"] = [sc.score(str(te.iloc[t].summary), o)["rougeL"].fmeasure * 100
                             for t, o in zip(r.test_id, r.output)]
            base = gen[(gen.model == m) & (gen.K == 4) & (gen.set < 3)]
            mm = r.merge(base, on=["K", "set", "perm", "test_id"], suffixes=("_x", ""))
            acc = mm.groupby(["set", "perm"])[["rougeL", "rougeL_r"]].mean()
            rer = (acc.rougeL - acc.rougeL_r).values
            od = []
            for s_, a in acc.groupby(level=0):
                v = a.rougeL.values
                od += [v[i] - v[j] for i, j in itertools.combinations(range(len(v)), 2)]
            od = np.array(od)
            rows.append(dict(model=m, task="samsum", rerun_flip=(mm.output_x != mm.output).mean() * 100,
                             order_flip=np.nan, rerun_absgap=np.abs(rer).mean(),
                             order_pairgap=np.abs(od).mean(),
                             noise_share=np.mean(rer ** 2) / np.mean(od ** 2)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ position influence (classification)
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def cls_position_design(g, K):
    """Long format rows: one per (prompt, class). Returns X (n,K), y (n,), set ids, test-class groups."""
    scores = np.array([json.loads(s) for s in g.scores])
    C = scores.shape[1]
    scores = np.maximum(scores, -30.0)
    yc = scores - scores.mean(1, keepdims=True)  # class-centred log-prob
    dl = np.array([json.loads(s) for s in g.demo_labels])  # (n, K)
    Xs, ys, grp, sets = [], [], [], []
    for c in range(C):
        Xs.append((dl == c).astype(float) - 1.0 / C)
        ys.append(yc[:, c])
        grp.append(g.test_id.values * 10 + c)
        sets.append(g.set.values)
    X, y = np.vstack(Xs), np.concatenate(ys)
    grp, sets = np.concatenate(grp), np.concatenate(sets)
    return X, y, grp, sets


def demean(a, grp):
    df = pd.DataFrame(a if a.ndim > 1 else a[:, None])
    return (df - df.groupby(grp).transform("mean")).values.squeeze()


def fit_influence(X, y, grp, sets, within_set=False):
    """beta_i with test x class FE (or set x test x class FE). Bootstrap over sets."""
    g2 = grp if not within_set else grp * 100 + sets
    Xd, yd = demean(X, g2), demean(y, g2)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    beta = _ols(Xd, yd)
    uniq = np.unique(sets)
    idx_by_set = {s: np.where(sets == s)[0] for s in uniq}
    boots = []
    for _ in range(400):
        samp = RNG.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([idx_by_set[s] for s in samp])
        # re-label duplicated sets so FE groups stay distinct is unnecessary for test-FE; fine.
        boots.append(_ols(Xd[ix], yd[ix]))
    return beta, np.array(boots)


def cls_influence(cls):
    rows, boots_store = [], {}
    for (model, task, K), g in cls[cls.K >= 2].groupby(["model", "task", "K"]):
        X, y, grp, sets = cls_position_design(g, K)
        beta, boots = fit_influence(X, y, grp, sets)
        boots_store[(model, task, K)] = boots
        ratio_b = boots[:, -1] / np.where(np.abs(boots[:, 0]) < 1e-9, np.nan, boots[:, 0])
        diff_b = boots[:, -1] - boots[:, 0]
        for i in range(K):
            rows.append(dict(model=model, task=task, K=K, pos=i + 1, beta=beta[i],
                             lo=np.percentile(boots[:, i], 2.5), hi=np.percentile(boots[:, i], 97.5)))
        rows.append(dict(model=model, task=task, K=K, pos="last/first", beta=beta[-1] / beta[0],
                         lo=np.nanpercentile(ratio_b, 2.5), hi=np.nanpercentile(ratio_b, 97.5)))
        rows.append(dict(model=model, task=task, K=K, pos="last-first", beta=beta[-1] - beta[0],
                         lo=np.percentile(diff_b, 2.5), hi=np.percentile(diff_b, 97.5)))
    return pd.DataFrame(rows)


def cls_agreement(cls):
    """Descriptive: P(prediction == label of demo at position i) per position."""
    rows = []
    for (model, task, K), g in cls[cls.K >= 2].groupby(["model", "task", "K"]):
        dl = np.array([json.loads(s) for s in g.demo_labels])
        agree = (dl == g.pred.values[:, None]).mean(0)
        for i in range(K):
            rows.append(dict(model=model, task=task, K=K, pos=i + 1, agree=agree[i]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ generation
TOKRE = re.compile(r"[a-z']+")
STOP = set("""a an the and or but of to in on at for with from by is are was were be been being it its this that
these those he she they them his her their we you i me my our your will would can could should not no yes
do does did have has had as so if then than there here about into over after before up down out just also
going go get got""".split())


def toks(s):
    return [t for t in TOKRE.findall(str(s).lower()) if t not in STOP and len(t) > 2]


def gen_analysis(gen):
    import sys
    sys.path.insert(0, f"{ROOT}/src")
    from tasks import load_gen
    tr, te = load_gen()
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    gen = gen.copy()
    gen["rougeL"] = [sc.score(str(te.iloc[t].summary), o)["rougeL"].fmeasure * 100
                     for t, o in zip(gen.test_id, gen.output)]
    gen["out_len"] = gen.output.map(lambda s: len(str(s).split()))
    # ---- sensitivity
    rows = []
    for (model, K, s), g in gen[gen.K > 0].groupby(["model", "K", "set"]):
        per = g.groupby("perm").rougeL.mean().values
        piv = g.pivot(index="perm", columns="test_id", values="output")
        rows.append(dict(model=model, task="samsum", K=K, set=s, **order_stats(per, piv.values)))
    sens = pd.DataFrame(rows)
    # ---- position influence
    sum_len = tr.summary.map(lambda s: np.log(len(str(s).split()))).values
    sum_tok = [set(toks(s)) for s in tr.summary]
    dia_tok = {t: set(toks(te.iloc[t].dialogue)) for t in gen.test_id.unique()}
    inf_rows, copy_rows = [], []
    for (model, K), g in gen[gen.K >= 2].groupby(["model", "K"]):
        demos = np.array([json.loads(d) for d in g.demos])
        X = sum_len[demos]
        X = X - X.mean()
        y = np.log(np.maximum(g.out_len.values, 1)).astype(float)
        grp, sets = g.test_id.values, g.set.values
        beta, boots = fit_influence(X, y, grp, sets)
        for i in range(K):
            inf_rows.append(dict(model=model, task="samsum", K=K, pos=i + 1, beta=beta[i],
                                 lo=np.percentile(boots[:, i], 2.5), hi=np.percentile(boots[:, i], 97.5)))
        rb = boots[:, 0] / np.where(np.abs(boots[:, -1]) < 1e-9, np.nan, boots[:, -1])
        inf_rows.append(dict(model=model, task="samsum", K=K, pos="first/last", beta=beta[0] / beta[-1],
                             lo=np.nanpercentile(rb, 2.5), hi=np.nanpercentile(rb, 97.5)))
        db = boots[:, 0] - boots[:, -1]
        inf_rows.append(dict(model=model, task="samsum", K=K, pos="first-last", beta=beta[0] - beta[-1],
                             lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5)))
        # lexical copying from demo i's summary (excluding words present in the test dialogue)
        cp = np.zeros((len(g), K))
        for r, (o, t, dm) in enumerate(zip(g.output, g.test_id, demos)):
            ot = set(toks(o)) - dia_tok[t]
            if not ot:
                continue
            for i in range(K):
                cp[r, i] = len(ot & sum_tok[dm[i]]) / max(len(set(toks(o))), 1)
        # per set: mean copy by position (each demo visits positions across perms)
        per_set = pd.DataFrame(cp).groupby(sets).mean()
        for i in range(K):
            lo, hi = boot_ci(per_set[i].values)
            copy_rows.append(dict(model=model, K=K, pos=i + 1, copy=cp[:, i].mean() * 100,
                                  lo=lo * 100, hi=hi * 100))
    return gen, sens, pd.DataFrame(inf_rows), pd.DataFrame(copy_rows)


# ------------------------------------------------------------------ plotting
COLORS = {0.5: "#9ecae1", 1.5: "#6baed6", 3.0: "#3182bd", 7.0: "#08519c", 14.0: "#08306b"}


def short(m):
    return m.replace("Qwen2.5-", "").replace("-Instruct", "")


def plot_vs_k(summ, metric, ylabel, fname):
    tasks = [t for t in TASK_ORDER if t in summ.task.unique()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), tasks):
        for m in sorted(summ.model.unique(), key=lambda m: SIZES.get(m, 0)):
            g = summ[(summ.task == t) & (summ.model == m)].sort_values("K")
            if g.empty:
                continue
            c = COLORS.get(SIZES.get(m), "gray")
            ax.plot(g.K, g[metric], "-o", color=c, label=short(m), ms=4)
            if metric + "_lo" in g:
                ax.fill_between(g.K, g[metric + "_lo"], g[metric + "_hi"], color=c, alpha=0.12)
        ax.set_title(t.upper())
        ax.set_xlabel("shots K")
        ax.set_xticks(sorted(summ[summ.task == t].K.unique()))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[-1].legend(fontsize=8, title="model")
    fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}", dpi=150)
    plt.close(fig)


def plot_influence(inf, fname, title_extra=""):
    inf = inf[inf.pos.map(lambda p: isinstance(p, (int, np.integer)))].copy()
    tasks = [t for t in TASK_ORDER if t in inf.task.unique()]
    Ks = [4, 6] if "samsum" in tasks and len(tasks) == 1 else [4, 8]
    fig, axes = plt.subplots(len(Ks), len(tasks), figsize=(4 * len(tasks), 3.2 * len(Ks)), squeeze=False)
    for r, K in enumerate(Ks):
        for c, t in enumerate(tasks):
            ax = axes[r, c]
            for m in sorted(inf.model.unique(), key=lambda m: SIZES.get(m, 0)):
                g = inf[(inf.task == t) & (inf.model == m) & (inf.K == K)].sort_values("pos")
                if g.empty:
                    continue
                col = COLORS.get(SIZES.get(m), "gray")
                ax.errorbar(g.pos.astype(int), g.beta, yerr=[g.beta - g.lo, g.hi - g.beta], fmt="-o",
                            color=col, ms=3, capsize=2, label=short(m))
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(1, K + 1))
            ax.set_title(f"{t.upper()}  K={K}{title_extra}")
            ax.set_xlabel("demo position (1 = first)")
            ax.grid(alpha=0.3)
        axes[r, 0].set_ylabel("influence β")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}", dpi=150)
    plt.close(fig)


def main():
    cls, gen = load()
    print("models:", sorted(cls.model.unique()))
    # zero-shot + accuracy table
    acc = cls.groupby(["model", "task", "K"]).correct.mean().unstack("K") * 100
    acc.round(1).to_csv(f"{TAB}/accuracy_by_K.csv")
    res = {}
    sens = cls_sensitivity(cls)
    sens.to_csv(f"{TAB}/cls_sensitivity_per_set.csv", index=False)
    summ = summarize_sens(sens)

    gsumm = None
    if gen is not None:
        gen, gsens, ginf, gcopy = gen_analysis(gen)
        gsens.to_csv(f"{TAB}/gen_sensitivity_per_set.csv", index=False)
        gsumm = summarize_sens(gsens)
        ginf.to_csv(f"{TAB}/gen_length_influence.csv", index=False)
        gcopy.to_csv(f"{TAB}/gen_copy_by_position.csv", index=False)
        (gen.groupby(["model", "K"]).rougeL.mean().unstack("K")).round(2).to_csv(f"{TAB}/gen_rougeL_by_K.csv")
        gen[["model", "K", "set", "perm", "test_id", "rougeL", "out_len"]].to_csv(
            f"{TAB}/gen_scores_long.csv", index=False)
    allsumm = pd.concat([summ, gsumm]) if gsumm is not None else summ
    allsumm["size"] = allsumm.model.map(SIZES)
    allsumm.sort_values(["task", "size", "K"]).round(3).to_csv(f"{TAB}/sensitivity_summary.csv", index=False)

    cl = cls[cls.K >= 3].assign(c100=lambda d: d.correct * 100.0)
    rel = [order_reliability(cl, "c100")]
    if gen is not None:
        rel.append(order_reliability(gen.assign(task="samsum"), "rougeL"))
    rel = pd.concat(rel)
    rel.to_csv(f"{TAB}/order_reliability_per_set.csv", index=False)
    relsum = rel.groupby(["task", "model"])[["split_half_rho", "range", "null_range"]].mean()
    # Spearman-Brown full-test reliability from the averaged split-half correlation
    relsum["sb_rho"] = 2 * relsum.split_half_rho / (1 + relsum.split_half_rho)
    relsum["range_over_null"] = relsum["range"] / relsum["null_range"]
    relsum.round(3).to_csv(f"{TAB}/order_reliability_summary.csv")
    res["reliability"] = relsum.round(3).reset_index().to_dict("records")
    print(relsum.round(3).to_string())

    nf = noise_floor(cls, gen)
    if len(nf):
        nf.round(3).to_csv(f"{TAB}/noise_floor.csv", index=False)
        res["noise_floor"] = nf.round(3).to_dict("records")
        print(nf.round(2).to_string())

    inf = cls_influence(cls)
    inf.to_csv(f"{TAB}/cls_position_influence.csv", index=False)
    agree = cls_agreement(cls)
    agree.to_csv(f"{TAB}/cls_label_agreement_by_position.csv", index=False)

    # ---------------- hypothesis tests ----------------
    # H1: variance vs K. Pairwise gap per set; test rise 2->4 and plateau 4->8.
    allsens = pd.concat([sens, gsens]) if gen is not None else sens
    h1 = []
    for (task, model), g in allsens[allsens.K >= 2].groupby(["task", "model"]):
        rise = stats.spearmanr(g[g.K <= 4].K, g[g.K <= 4].pairgap)
        hi = g[g.K >= 4]
        kw = stats.kruskal(*[x.pairgap.values for _, x in hi.groupby("K")])
        plateau_rho = stats.spearmanr(hi.K, hi.pairgap)
        means = g.groupby("K").pairgap.mean()
        h1.append(dict(task=task, model=model, **{f"gap_K{k}": v for k, v in means.items()},
                       rise_rho=rise.correlation, rise_p=rise.pvalue,
                       ge4_rho=plateau_rho.correlation, ge4_p=plateau_rho.pvalue, ge4_kw_p=kw.pvalue))
    h1 = pd.DataFrame(h1)
    h1.round(4).to_csv(f"{TAB}/H1_variance_vs_K.csv", index=False)
    # pooled over models (model as blocking factor via within-model rank)
    pooled = []
    for task, g in allsens[allsens.K >= 2].groupby("task"):
        g = g.copy()
        g["z"] = g.groupby("model").pairgap.transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
        r1 = stats.spearmanr(g[g.K <= 4].K, g[g.K <= 4].z)
        r2 = stats.spearmanr(g[g.K >= 4].K, g[g.K >= 4].z)
        pooled.append(dict(task=task, rise_rho=r1.correlation, rise_p=r1.pvalue,
                           ge4_rho=r2.correlation, ge4_p=r2.pvalue,
                           **{f"gap_K{k}": v for k, v in g.groupby("K").pairgap.mean().items()}))
    pd.DataFrame(pooled).round(4).to_csv(f"{TAB}/H1_pooled.csv", index=False)
    res["H1_pooled"] = pd.DataFrame(pooled).round(4).to_dict("records")

    # H2/H3 ratios
    ratios = inf[inf.pos.isin(["last/first", "last-first"])]
    res["H2_cls_last_vs_first"] = ratios.round(3).to_dict("records")
    if gen is not None:
        res["H3_gen_first_vs_last"] = ginf[ginf.pos.isin(["first/last", "first-last"])].round(3).to_dict("records")

    # H4: best-worst gap per task at K>=4 (8 perms)
    h4 = allsumm[allsumm.K >= 4].groupby(["task", "model"])[["range", "range_max", "pairgap", "sd"]].mean()
    h4.round(2).to_csv(f"{TAB}/H4_range_by_task.csv")
    res["H4"] = h4.groupby("task").mean().round(2).to_dict("index")

    # H5: size vs sensitivity
    h5 = []
    for task, g in allsumm[allsumm.K >= 2].groupby("task"):
        gm = g.groupby("size")[["pairgap", "sd", "range", "flip"]].mean()
        if len(gm) >= 3:
            for met in ["pairgap", "sd", "range", "flip"]:
                r = stats.spearmanr(gm.index, gm[met])
                h5.append(dict(task=task, metric=met, rho=r.correlation, p=r.pvalue,
                               **{f"{s}B": v for s, v in gm[met].items()}))
    h5 = pd.DataFrame(h5)
    h5.round(4).to_csv(f"{TAB}/H5_size_vs_sensitivity.csv", index=False)
    res["H5"] = h5.round(4).to_dict("records")
    pc, pg = load_probes()
    pinf = pinf_pred = pginf = None
    if pc is not None:
        pinf, pinf_pred = probe_cls_analysis(pc, cls)
        pinf.to_csv(f"{TAB}/probe_cls_influence_logp.csv", index=False)
        pinf_pred.to_csv(f"{TAB}/probe_cls_influence_pred.csv", index=False)
        res["H2_probe_last_over_first"] = pinf[pinf.pos == "last/first"].round(3).to_dict("records")
    if pg is not None:
        pginf = probe_gen_analysis(pg)
        pginf.to_csv(f"{TAB}/probe_gen_marker_influence.csv", index=False)
        res["H3_probe_first_over_last"] = pginf[pginf.pos == "first/last"].round(3).to_dict("records")
    json.dump(res, open(f"{ROOT}/results/summary.json", "w"), indent=1, default=float)

    # ---------------- figures ----------------
    plot_vs_k(allsumm, "pairgap", "mean |Δ metric| between two orders (pts)", "fig1_pairgap_vs_K.png")
    plot_vs_k(allsumm, "sd", "SD across orders (pts)", "fig1b_sd_vs_K.png")
    plot_vs_k(allsumm, "range", "best − worst order (pts)", "fig2_range_vs_K.png")
    plot_vs_k(allsumm, "flip", "prediction flip rate between orders", "fig3_flip_vs_K.png")
    plot_vs_k(allsumm, "acc", "mean accuracy / ROUGE-L", "fig0_acc_vs_K.png")
    plot_influence(inf, "fig4_cls_position_influence_regression.png")
    if pinf is not None:
        plot_influence(pinf, "fig4b_cls_labelflip_influence.png", " (label flip)")
    if pginf is not None:
        plot_influence(pginf, "fig5b_gen_marker_influence.png", " (marker)")
    if gen is not None:
        plot_influence(ginf, "fig5_gen_length_influence.png", " (length)")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, K in zip(axes, [3, 4, 6]):
            for m in sorted(gcopy.model.unique(), key=lambda m: SIZES.get(m, 0)):
                g = gcopy[(gcopy.model == m) & (gcopy.K == K)]
                col = COLORS.get(SIZES.get(m), "gray")
                ax.errorbar(g.pos, g["copy"], yerr=[g["copy"] - g.lo, g.hi - g["copy"]], fmt="-o",
                            color=col, ms=3, capsize=2, label=short(m))
            ax.set_title(f"SAMSum K={K}: lexical copying")
            ax.set_xlabel("demo position")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("% output words from demo summary")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{FIG}/fig6_gen_copy_by_position.png", dpi=150)
        plt.close(fig)
    # size plot
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for t in [t for t in TASK_ORDER if t in allsumm.task.unique()]:
        g = allsumm[(allsumm.task == t) & (allsumm.K >= 2)].groupby("size")[["pairgap", "flip"]].mean()
        axes[0].plot(g.index, g.pairgap, "-o", label=t)
        if t != "samsum":
            axes[1].plot(g.index, g.flip * 100, "-o", label=t)
    for ax, yl in zip(axes, ["mean pairwise order gap (pts)", "prediction flip rate (%)"]):
        ax.set_xscale("log")
        ax.set_xlabel("parameters (B)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_size_vs_sensitivity.png", dpi=150)
    plt.close(fig)
    print(allsumm.sort_values(["task", "size", "K"])[["task", "model", "K", "acc", "sd", "range", "pairgap", "flip"]].round(2).to_string())
    print(json.dumps({k: v for k, v in res.items() if k in ("H1_pooled", "H4")}, indent=1, default=float))


if __name__ == "__main__":
    main()
