"""Run all ordering conditions for one model with vLLM.

Classification: one forward step per prompt; label log-probs are obtained by
restricting the next token to the label tokens (allowed_token_ids) with
logprobs_mode="processed_logprobs", i.e. log-softmax renormalised over the
label set. Prediction = argmax. Greedy/deterministic -> any variance across
permutations is caused by the order alone.

Generation: greedy decoding, max 80 new tokens, stop at newline.

Usage: python src/run_model.py --model Qwen/Qwen2.5-7B-Instruct [--limit N]
Outputs: results/raw/<model_tag>_cls.parquet, results/raw/<model_tag>_gen.parquet
"""
import argparse
import json
import os
import time

import pandas as pd

from tasks import CLS_TASKS, ROOT, cls_prompt, gen_prompt, load_cls, load_gen


def select(conds, args):
    """Subset conditions (debug --limit, or the noise-floor rerun --only_k/--max_sets)."""
    if args.only_k:
        conds = [c for c in conds if c["K"] == args.only_k and (not args.max_sets or c["set"] < args.max_sets)]
    return conds[: args.limit] if args.limit else conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: max conds per task")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--design", default="design.json", help="design file in results/")
    ap.add_argument("--suffix", default="", help="appended to output file names")
    ap.add_argument("--max_batched_tokens", type=int, default=0, help="cap per-step tokens (small GPU slices)")
    ap.add_argument("--only_k", type=int, default=0, help="noise-floor rerun: keep only this K")
    ap.add_argument("--max_sets", type=int, default=0, help="noise-floor rerun: keep sets < max_sets")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tag = args.model.split("/")[-1]
    otag = tag + args.suffix
    out_dir = f"{ROOT}/results/raw"
    os.makedirs(out_dir, exist_ok=True)
    design = json.load(open(f"{ROOT}/results/{args.design}"))

    llm = LLM(model=args.model, dtype="bfloat16", seed=0, enable_prefix_caching=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=4096,
              logprobs_mode="processed_logprobs",
              **({"max_num_batched_tokens": args.max_batched_tokens, "max_num_seqs": 64}
                 if args.max_batched_tokens else {}))
    tok = llm.get_tokenizer()
    meta = {"model": args.model, "vllm_tag": tag, "start": time.time()}

    # ---------------- classification ----------------
    rows = []
    for task, td in design["cls"].items():
        tr, te = load_cls(task)
        label_ids = [tok.encode(w, add_special_tokens=False)[0] for w in CLS_TASKS[task]["labels"]]
        assert len(set(label_ids)) == len(label_ids), (task, label_ids)
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_ids),
                            allowed_token_ids=label_ids)
        tests = [te.iloc[i] for i in td["test_idx"]]
        conds = select(td["conds"], args)
        # zero-shot baseline as K=0
        jobs = [] if args.only_k else [(0, -1, -1, [], [])]
        for ci, c in enumerate(conds):
            for pi, p in enumerate(c["perms"]):
                ordered = [c["demos"][j] for j in p]
                jobs.append((c["K"], c["set"], pi, ordered, [int(tr.iloc[d].label) for d in ordered]))
        prompts, keys = [], []
        for (k, s, pi, ordered, demo_labels) in jobs:
            demo_rows = [tr.iloc[d] for d in ordered]
            for ti, trow in zip(td["test_idx"], tests):
                prompts.append(cls_prompt(task, demo_rows, trow))
                keys.append((task, k, s, pi, ti, int(trow.label), json.dumps(ordered), json.dumps(demo_labels)))
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        print(f"[{tag}] {task}: {len(prompts)} prompts in {time.time()-t0:.0f}s", flush=True)
        for key, o in zip(keys, outs):
            lp = o.outputs[0].logprobs[0]
            scores = [lp[i].logprob if i in lp else float("-inf") for i in label_ids]
            pred = max(range(len(scores)), key=lambda j: scores[j])
            rows.append(dict(task=key[0], K=key[1], set=key[2], perm=key[3], test_id=key[4], label=key[5],
                             demos=key[6], demo_labels=key[7], pred=pred,
                             scores=json.dumps(scores), n_prompt_tok=len(o.prompt_token_ids)))
    df = pd.DataFrame(rows)
    df["correct"] = (df.pred == df.label).astype(int)
    df.to_parquet(f"{out_dir}/{otag}_cls.parquet")
    print(df.groupby(["task", "K"]).correct.mean().unstack().round(3), flush=True)

    # ---------------- generation ----------------
    if not args.skip_gen:
        gd = design["gen"]["samsum"]
        tr, te = load_gen()
        conds = select(gd["conds"], args)
        sp = SamplingParams(max_tokens=80, temperature=0.0, stop=["\n"])
        jobs = [] if args.only_k else [(0, -1, -1, [])]
        for c in conds:
            for pi, p in enumerate(c["perms"]):
                jobs.append((c["K"], c["set"], pi, [c["demos"][j] for j in p]))
        prompts, keys = [], []
        for (k, s, pi, ordered) in jobs:
            demo_rows = [tr.iloc[d] for d in ordered]
            for ti in gd["test_idx"]:
                prompts.append(gen_prompt(demo_rows, te.iloc[ti]))
                keys.append((k, s, pi, ti, json.dumps(ordered)))
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        print(f"[{tag}] samsum: {len(prompts)} prompts in {time.time()-t0:.0f}s", flush=True)
        g = pd.DataFrame([dict(K=k[0], set=k[1], perm=k[2], test_id=k[3], demos=k[4],
                               output=o.outputs[0].text.strip(), n_prompt_tok=len(o.prompt_token_ids))
                          for k, o in zip(keys, outs)])
        g.to_parquet(f"{out_dir}/{otag}_gen.parquet")
    meta["seconds"] = time.time() - meta["start"]
    json.dump(meta, open(f"{out_dir}/{otag}_meta.json", "w"))
    print(f"[{tag}] done in {meta['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
