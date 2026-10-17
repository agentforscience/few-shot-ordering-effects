"""Causal position-influence probes (Exp 2b / 3b).

Classification (label-flip probe): for demo sets 0-9 at K in {4, 8} and the first
two permutations of each, flip the label of the demo at position i (to another
label, chosen deterministically) keeping everything else fixed. Influence of
position i = mean increase of the log-prob (over the label set) assigned to the
flipped-to label, relative to the unflipped prompt (available from run_model).

Generation (style-marker probe): for SAMSum K in {4, 6}, rewrite the summary of
the demo at position i to start with the marker "In short, ". Influence of
position i = increase in log P(output starts with " In short,") relative to the
no-marker control (pos=-1), scored with prompt log-probs (greedy outputs almost
never start with the marker, so a continuous measure is needed).

Usage: python src/run_probe.py --model Qwen/Qwen2.5-7B-Instruct
Outputs: results/raw/<tag>_probe_cls.parquet, results/raw/<tag>_probe_gen.parquet
"""
import argparse
import json
import os
import random
import time

import pandas as pd

from tasks import CLS_TASKS, ROOT, GEN_HEADER, cls_prompt, gen_fmt, load_cls, load_gen

CLS_PROBE_K, GEN_PROBE_K, N_PERMS = [4, 8], [4, 6], 2
MARKER = "In short, "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--only_gen", action="store_true")
    ap.add_argument("--cls_k", type=int, nargs="+", default=CLS_PROBE_K)
    ap.add_argument("--gen_k", type=int, nargs="+", default=GEN_PROBE_K)
    ap.add_argument("--max_batched_tokens", type=int, default=1024,
                    help="caps per-step tokens: prompt_logprobs materialise full-vocab logits")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    tag = args.model.split("/")[-1]
    design = json.load(open(f"{ROOT}/results/design.json"))
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, enable_prefix_caching=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096,
              logprobs_mode="processed_logprobs", max_num_batched_tokens=args.max_batched_tokens,
              enable_chunked_prefill=True)
    tok = llm.get_tokenizer()
    rng = random.Random(7)

    rows = []
    for task, td in ([] if args.only_gen else design["cls"].items()):
        tr, te = load_cls(task)
        labels = CLS_TASKS[task]["labels"]
        label_ids = [tok.encode(w, add_special_tokens=False)[0] for w in labels]
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_ids), allowed_token_ids=label_ids)
        tests = [te.iloc[i] for i in td["test_idx"]]
        prompts, keys = [], []
        for c in td["conds"]:
            if c["K"] not in args.cls_k:
                continue
            for pi in range(N_PERMS):
                ordered = [c["demos"][j] for j in c["perms"][pi]]
                base_rows = [tr.iloc[d].copy() for d in ordered]
                for pos in range(c["K"]):
                    old = int(base_rows[pos].label)
                    new = rng.choice([l for l in range(len(labels)) if l != old])
                    demo_rows = [r.copy() for r in base_rows]
                    demo_rows[pos]["label"] = new
                    for ti, trow in zip(td["test_idx"], tests):
                        prompts.append(cls_prompt(task, demo_rows, trow))
                        keys.append((task, c["K"], c["set"], pi, pos, old, new, ti, int(trow.label)))
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        print(f"[{tag}] probe {task}: {len(prompts)} prompts in {time.time()-t0:.0f}s", flush=True)
        for k, o in zip(keys, outs):
            lp = o.outputs[0].logprobs[0]
            scores = [lp[i].logprob if i in lp else float("-inf") for i in label_ids]
            rows.append(dict(task=k[0], K=k[1], set=k[2], perm=k[3], pos=k[4], old_label=k[5], new_label=k[6],
                             test_id=k[7], label=k[8], pred=max(range(len(scores)), key=lambda j: scores[j]),
                             scores=json.dumps(scores)))
    if rows:
        pd.DataFrame(rows).to_parquet(f"{ROOT}/results/raw/{tag}_probe_cls.parquet")

    # ---- generation style-marker probe
    gd = design["gen"]["samsum"]
    tr, te = load_gen()
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    n_marker = len(tok.encode(" " + MARKER.strip(), add_special_tokens=False))
    prompts, keys = [], []
    for c in gd["conds"]:
        if c["K"] not in args.gen_k:
            continue
        for pi in range(N_PERMS):
            ordered = [c["demos"][j] for j in c["perms"][pi]]
            for pos in list(range(c["K"])) + [-1]:  # -1 = no marker (control)
                parts = []
                for j, d in enumerate(ordered):
                    s = str(tr.iloc[d].summary).strip().replace("\n", " ")
                    if j == pos:
                        s = MARKER + s[0].lower() + s[1:]
                    parts.append(gen_fmt(tr.iloc[d]) + " " + s)
                prefix = GEN_HEADER + "".join(p + "\n\n" for p in parts)
                for ti in gd["test_idx"]:
                    prompts.append(prefix + gen_fmt(te.iloc[ti]) + " " + MARKER.strip())
                    keys.append((c["K"], c["set"], pi, pos, ti))
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"[{tag}] probe samsum: {len(prompts)} prompts in {time.time()-t0:.0f}s", flush=True)
    recs = []
    for k, o in zip(keys, outs):
        ids, plp = o.prompt_token_ids, o.prompt_logprobs
        lp = sum(plp[j][ids[j]].logprob for j in range(len(ids) - n_marker, len(ids)))
        recs.append(dict(K=k[0], set=k[1], perm=k[2], pos=k[3], test_id=k[4], marker_logp=lp))
    g = pd.DataFrame(recs)
    g.to_parquet(f"{ROOT}/results/raw/{tag}_probe_gen.parquet")
    print(g.groupby(["K", "pos"]).marker_logp.mean().unstack().round(2), flush=True)


if __name__ == "__main__":
    main()
