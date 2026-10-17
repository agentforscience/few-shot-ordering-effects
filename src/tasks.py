"""Task definitions, data loading and prompt formatting for the ordering study.

All paths are relative to the workspace root (pass --root or run from root).
"""
import os
import pandas as pd

ROOT = os.environ.get("WS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "datasets")


def _wc(s):
    return len(str(s).split())


# ---------------------------------------------------------------------------
# Classification tasks: each defines header, example formatter, label words.
# Label words are scored as the first token after the "Answer"/"Sentiment" cue.
# ---------------------------------------------------------------------------
CLS_TASKS = {
    "sst2": dict(
        family="sentiment",
        header="Classify the sentiment of each movie review as positive or negative.\n\n",
        fmt=lambda r: f"Review: {r['sentence'].strip()}\nSentiment:",
        labels=[" negative", " positive"],  # index == dataset label id
    ),
    "rte": dict(
        family="nli",
        header="Decide whether the premise entails the hypothesis. Answer yes or no.\n\n",
        fmt=lambda r: f"Premise: {r['sentence1'].strip()}\nHypothesis: {r['sentence2'].strip()}\nAnswer:",
        labels=[" yes", " no"],  # 0=entailment, 1=not_entailment
    ),
    "mnli": dict(
        family="nli",
        header=("Decide whether the premise entails the hypothesis. Answer yes (entailment), "
                "maybe (neutral), or no (contradiction).\n\n"),
        fmt=lambda r: f"Premise: {r['premise'].strip()}\nHypothesis: {r['hypothesis'].strip()}\nAnswer:",
        labels=[" yes", " maybe", " no"],  # 0=entailment, 1=neutral, 2=contradiction
    ),
}


def load_cls(task):
    """Return (train_pool, test_pool) DataFrames with a 'label' column.

    Demo pools are length-filtered so that 8-shot prompts stay short.
    """
    if task == "sst2":
        tr = pd.read_parquet(f"{D}/sst2/train.parquet")
        te = pd.read_parquet(f"{D}/sst2/validation.parquet")
        tr = tr[tr.sentence.map(_wc).between(5, 30)]
    elif task == "rte":
        tr = pd.read_parquet(f"{D}/glue/rte/train-00000-of-00001.parquet")
        te = pd.read_parquet(f"{D}/glue/rte/validation-00000-of-00001.parquet")
        tr = tr[tr.sentence1.map(_wc) <= 50]
    elif task == "mnli":
        tr = pd.read_parquet(f"{D}/glue/mnli/train-00000-of-00001.parquet")
        te = pd.read_parquet(f"{D}/glue/mnli/validation_matched-00000-of-00001.parquet")
        tr = tr[tr.premise.map(_wc).between(5, 35)]
    else:
        raise ValueError(task)
    tr = tr[tr.label >= 0].reset_index(drop=True)
    te = te[te.label >= 0].reset_index(drop=True)
    return tr, te


def cls_demo_text(task, row):
    t = CLS_TASKS[task]
    return t["fmt"](row) + t["labels"][int(row["label"])]


def cls_prompt(task, demo_rows, test_row):
    t = CLS_TASKS[task]
    parts = [cls_demo_text(task, r) for r in demo_rows]
    return t["header"] + "".join(p + "\n\n" for p in parts) + t["fmt"](test_row)


# ---------------------------------------------------------------------------
# Generation task: SAMSum dialogue summarisation.
# ---------------------------------------------------------------------------
GEN_HEADER = "Summarize each dialogue in one or two sentences.\n\n"


def load_gen():
    tr = pd.read_csv(f"{D}/samsum/train.csv").dropna()
    te = pd.read_csv(f"{D}/samsum/test.csv").dropna()
    # keep dialogues short so that 6-shot prompts fit comfortably
    tr = tr[tr.dialogue.map(_wc).between(20, 90) & tr.summary.map(_wc).between(5, 45)]
    te = te[te.dialogue.map(_wc).between(20, 120)]
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def gen_fmt(row):
    d = str(row["dialogue"]).replace("\r\n", "\n").strip()
    return f"Dialogue:\n{d}\nSummary:"


def gen_prompt(demo_rows, test_row):
    parts = [gen_fmt(r) + " " + str(r["summary"]).strip().replace("\n", " ") for r in demo_rows]
    return GEN_HEADER + "".join(p + "\n\n" for p in parts) + gen_fmt(test_row)
