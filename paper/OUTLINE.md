# Outline: Few-shot demo order effects (Qwen2.5-Instruct 0.5B-14B)
1. Intro: hook (order is uncontrolled variable) -> gap (no joint study of K, task, size, causal position influence, noise) -> approach (controlled protocol, pairwise gap, causal probes, reliability + noise floor) -> preview (H1 artifact, NLI>sentiment but thresholds fail, recency not general, generation recency not primacy, size helps SST-2/RTE only) -> contributions.
2. Related work: order sensitivity (Lu 2022), biases/calibration (Zhao 2021), position in long context (Liu 2023), what ICL uses (Min 2022, Brown 2020), selection (Liu 2022 KATE, Rubin 2022), nondeterminism.
3. Method: formulation (demo set D, order pi, metric m), sensitivity metrics (SD, range, pairwise gap), reliability (split-half, null), noise floor, causal probes (label-flip, style marker), setup (models, tasks, design, decoding).
4. Results: accuracy (Tab), H1 (Fig 1/2, Tab), H4 (Tab), reliability + noise floor (Tab), H2 (Fig + Tab), H3 (Fig + Tab), H5 (Fig + Tab).
5. Discussion: what survives, relation to prior work, practical guidance, limitations, broader impact.
6. Conclusion.
