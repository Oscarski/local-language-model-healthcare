# Final Audited Experiment Report
## Routing for an Edge-Deployed Medical LLM: Recorded Results and Methodological Limits

This is the canonical report for writing the final bachelor thesis. It
interprets the completed professor-run experiment using only preserved
evidence. The original unedited execution report is retained at
`artifacts/professor_run/REPORT_ORIGINAL.md`, and the exact executable source
snapshot associated with that run is preserved in
`artifacts/professor_run/executed_source.tar.gz`.

## Executive Summary

A Mistral-7B-Instruct-v0.3 model was fine-tuned on medical multiple-choice
questions and combined with an uncertainty-routing pipeline. The recorded run
completed end-to-end on 4 x NVIDIA L4 GPUs and produced routing, threshold,
evaluation, and ablation results.

The experiment is useful, but its strongest original safety interpretation is
not supported. The recorded conformal step thresholds the frequency with which
the system answers locally (`local_rate`); it is not a formal guarantee that
locally returned medical answers are correct (`local_accuracy`). Moreover, a
clean formal claim even for the recorded `local_rate` procedure is unsupported:
the probe/calibration construction was not a clean fixed-score split protocol,
supervised layer selection reused part of the later calibration subset, and
the saved threshold fails a same-sample inclusion invariant. The final thesis
therefore presents the completed system together with a methodological audit
of the gap between intended safety claims and the recorded evidence.

## Recorded Protocol

### Hardware and Fine-Tuning

| Item | Recorded value |
|---|---|
| Hardware | 4 x NVIDIA L4, 23.7 GB VRAM each |
| Model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Fine-tuning | LoRA, bf16, SDPA |
| Successful learning rate | `5e-5` |
| Fine-tune training examples | `102,765` |
| Early-stopping set | `ft_val=2,000`, carved from `train_ft` with seed 42 |
| Best eval loss | `0.6366` at epoch `0.81` |

The original preparation step produced a `train_ft` pool of `104,765`; the
fine-tuning script subsequently reserved `2,000` examples as `ft_val`.

### Data and Downstream Pipeline

```text
MedMCQA single-answer pool: 120,765
  |-- train_ft pool: 104,765 -> 102,765 train + 2,000 ft_val
  `-- probe candidates: 16,000
        |-- 2,693 removed as normalized exact matches to train_ft
        `-- retained probe_set: 13,307
        |-- routing_train: 8,000
        |-- iso_cal: 2,000
        `-- conformal_cal: 3,307
```

MedMCQA test labels are hidden (`cop=-1`), so the run used MedMCQA validation
(`n=4,183`) as nominal in-distribution evaluation. MedQA-USMLE (`n=1,273`)
and MMLU medical (`n=945`) were external evaluation sets.

### Signals and Router

Four signals were extracted historically: restricted entropy `H`, logit gap,
hidden-state probe score, and `p_true`. The executed router used three
features, `[H, probe, p_true]`, after dropping `gap` because its absolute
correlation with entropy was `0.9564`. Layer `24` was selected for the
hidden-state probe.

The recorded router used logistic regression followed by isotonic
calibration. At `alpha=0.10`, the historical threshold output was
`q_hat=0.5814`, corresponding to decision threshold `0.4186`.

## Recorded Evaluation Results

| Evaluation set | n | Standalone accuracy | AUROC | AUGRC | Brier | ECE | local_rate | local_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedMCQA val | 4,183 | 0.5687 | 0.7429 | 0.15619 | 0.2041 | 0.0548 | 0.7927 | 0.6339 |
| MedQA-USMLE | 1,273 | 0.5664 | 0.6949 | 0.16899 | 0.2301 | 0.1032 | 0.9065 | 0.5832 |
| MMLU medical | 945 | 0.6857 | 0.7823 | 0.09549 | 0.1705 | 0.0425 | 0.9206 | 0.7172 |

These values are preserved observations from `results/evaluation.json`.
`local_rate` is the fraction answered locally; `local_accuracy` is the
empirical correctness among those local answers.

## Methodological Audit

### The Conformal Claim

The historical procedure used nonconformity `1 - routing_score` and selected
queries whose score exceeded a quantile-derived threshold. Its reported
operating quantity is the frequency of local decisions. It does not directly
control incorrect local answers or certify clinical safety.

Accordingly, the raw `cp_guarantee_valid: true` field associated with
MedMCQA val is retained as a legacy result field, not adopted as the final
thesis conclusion.

### Why The Recorded Split-CP Validity Is Unsupported

Five recorded-protocol issues prevent a formal claim for `local_rate`:

1. The OOF correctness-probe scores were generated across the full retained
   `probe_set` before that set was repartitioned into `routing_train`,
   `iso_cal`, and `conformal_cal`. Consequently, `conformal_cal` was not an
   untouched calibration subset for a score construction fixed independently
   of that full pool.
2. Both recorded downstream partitioning calls used the correctness outcome
   `y` for stratification. This is a disclosed dependency of the historical
   protocol, not a corrected prospective calibration design.
3. Layer `24` was selected through supervised evaluation on a deterministic
   random sample of `2,000` examples from the full `probe_set`. Reconstructing
   those indices and the saved downstream split shows that `483` of these
   examples later occur in `conformal_cal`; the calibration subset was
   therefore not untouched by model/feature selection.
4. Calibration used OOF probe scores on `probe_set`, while evaluation used the
   final probe trained on the full `probe_set`. The score construction is not
   fixed between calibration and evaluation.
5. For `n=3,307` and `alpha=0.10`, inclusion of the saved conformal quantile
   on the same calibration sample requires at least
   `ceil((n+1)*(1-alpha)) = 2,978` selected examples. The raw output reports
   `local_rate_on_cal=0.8902`, uniquely corresponding to only `2,944`
   selections: a shortfall of `34`. This is consistent with a boundary/tie
   handling or floating-point complement issue in the historical threshold
   application, and it is an execution-level invariant failure.

These findings do not alter raw results; they restrict the claims that may be
made from them.

### Descriptive Accepted-Error Analysis

Using only recorded aggregate counts, the saved operating point can be
described in terms of errors still answered locally:

| Evaluation set | Total model errors | Errors answered locally | Errors not answered locally | Descriptive fraction not answered locally |
|---|---:|---:|---:|---:|
| MedMCQA val | 1,804 | 1,214 | 590 | 0.3271 |
| MedQA-USMLE | 552 | 481 | 71 | 0.1286 |
| MMLU medical | 297 | 246 | 51 | 0.1717 |

This is a secondary reconstruction from saved point estimates, not a new
experiment and not a clinical safety guarantee.

### Post-Run Validation Overlap Check

A bounded CPU-only audit using public MedMCQA text and committed split indices
checked the later choice of validation as nominal in-distribution evaluation.
The audit is pinned to dataset revision
`91c6572c454088bf71b679ad90aa8dffcd0d5868` and records fingerprints and
normalized-input hashes so that an upstream data change cannot silently alter
this finding.
Normalized exact matching found `1` validation question present in `train_ft`
and `0` present in `probe_set`. This is disclosed as a small
training-to-evaluation overlap (`1/4,183`); it does not justify altering the
immutable recorded metrics. Similarity-ranked pairs are review candidates
only, not automatically classified contamination.

### Threshold-Transfer Diagnostic

The recorded diagnostic compared:

| Setup | Threshold source | local_rate on MedMCQA val | local_accuracy |
|---|---|---:|---:|
| A | probe-derived `conformal_cal` | 0.7927 | 0.6339 |
| B | val 5-fold threshold-transfer analysis | 0.9144 | 0.5982 |

This indicates that the historical threshold did not transfer to MedMCQA val
as anticipated. It is not conclusive proof that one distribution shift alone
caused the discrepancy: the calibration-set reuse and changed score
construction above confound that interpretation. Setup B is descriptive; it
does not restore formal validity for deployment.

### `p_true`

The historical Blocker 3 intended to validate Yes/No vocabulary mass. In the
recorded implementation it computes a softmax over only two logits and sums
the result, making `mass=1.0` inevitable. Thus its `INCLUDE` verdict is
recorded behavior, not valid evidence of feature quality.

The ablation provides a limited empirical observation: the `p_true`-only
variant has in-dist AUROC `0.5931`, and the refitted `H+probe+p_true` feature
subset has point-estimate AUROC `0.7407`, versus `0.7419` for refitted
`H+probe`. This is reported as no observed improvement, not a statistical
proof of zero contribution.

### External Evaluation Results

The MMLU medical split has the best recorded point metrics. This observation
is valid. Possible explanations, including greater familiarity from
pretraining or benchmark exposure, were not tested in this run and remain
hypotheses. Raw outputs retain historical `near-OOD` and `far-OOD` labels;
thesis-facing text should refer to external MedQA-USMLE and external MMLU
medical evaluations rather than asserting a validated shift severity.

### Mondrian Analysis

The historical Mondrian analysis is retained as executed. Its domain mapping
did not include all spellings present in MedMCQA (`Gynaecology & Obstetrics`,
`Anaesthesia`, and `Orthopaedics`), which caused affected examples to fall
through to the default group. Its results therefore cannot be presented as a
corrected five-domain analysis.

## What The Experiment Supports

- A completed, documented fine-tuning and routing experiment for medical MCQ.
- Point estimates for routing discrimination and local-answer behavior on
  three evaluation sets.
- An empirical warning that a selection-rate threshold must not be presented
  as a medical-correctness guarantee.
- A case study of how feature validation and calibration protocol choices can
  compromise safety interpretation.
- A transparent post-run disclosure of one normalized exact
  training-to-validation question overlap.

## What It Does Not Support

- A formal guarantee of correct or clinically safe local medical answers.
- A clean formal conformal guarantee for the recorded local-rate procedure.
- A causal claim explaining MMLU or MedQA behavior.
- Recomputed corrected results: checkpoints and feature arrays are not present
  in the repository.

## Artifacts and Reproducibility

Raw professor-run artifacts remain under `results/`, `figures/`, and `logs/`
and are integrity-recorded in
`artifacts/professor_run/raw_artifacts.sha256`. The original report remains in
`artifacts/professor_run/REPORT_ORIGINAL.md`.

Committed CPU-only audit outputs are preserved under `audited/`. They
summarize preserved aggregate evidence and must not be described as a model
rerun. Their checksums are maintained separately from raw professor outputs;
normal verification is read-only. Raw plotted interpretations are also
historical artifacts; the corrected figure-use guidance is under
`audited/results/`. Further provenance detail is in `PROVENANCE.md`, and
executed protocol values suitable for citation are in
`configs/executed_professor_run.yaml`.
Known corrections for immutable raw artifacts are listed in
`audited/results/raw_artifact_errata.md`.
