# Audited Methodological Decision Log

This log is authoritative for thesis writing. It distinguishes what was
executed in the professor run from how those results may be interpreted.

## Recorded Execution Decisions

### D-01 - Preserve the completed professor run

**Decision:** Treat the imported GPU run as immutable recorded evidence.

**Rationale:** No further training will be performed, and the recorded
artifacts are the experimental basis of the thesis.

### D-02 - Successful fine-tuning configuration

**Decision:** Report the successful run as LoRA bf16 with SDPA on 4 x NVIDIA
L4 at `learning_rate=5e-5`, using `102,765` training examples and `2,000`
`ft_val` examples for early stopping.

**Evidence:** Training logs and `configs/finetune_config_ddp.yaml`.

### D-03 - Evaluation dataset substitution

**Decision:** Report MedMCQA validation (`n=4,183`) as nominal
in-distribution evaluation because the official MedMCQA test labels are hidden.

**Evidence:** Extraction logs report `cop=-1` for the official test split.

### D-04 - Downstream repartition used in results

**Decision:** Report the executed downstream partition as
`routing_train=8,000`, `iso_cal=2,000`, `conformal_cal=3,307`.

**Rationale:** The earlier preparation split `9,307/4,000` remains provenance
but was superseded by the repartition used in the recorded downstream run.

### D-05 - Recorded routing feature set

**Decision:** Report best layer `24` and historical routing features
`[H, probe, p_true]`, with `gap` dropped after high correlation with entropy.

**Evidence:** `results/best_layer.json` and `results/routing_metrics.json`.

## Interpretation Decisions

### D-06 - Raw outputs remain immutable

**Decision:** Do not overwrite raw `results/`, `figures/`, `logs/`, or the
original professor report. New derived material is written under `audited/`.

### D-07 - Terminology: local rate versus local accuracy

**Decision:** Use:

- `local_rate = P(answer locally)`;
- `local_accuracy = P(correct | answer locally)`.

**Rationale:** They are different measured quantities. The former cannot be
presented as evidence of clinical correctness.

### D-08 - No medical safety guarantee claim

**Decision:** Do not state that the historical conformal procedure guarantees
correct or safe medical answers.

**Rationale:** It thresholds a routing score to produce a local-selection
frequency. It does not control errors among local answers, and its recorded
`local_rate_on_cal=0.8902` is below the nominal `0.90` target at alpha `0.10`.

### D-09 - No formal recorded local-rate guarantee claim

**Decision:** Do not state that the recorded split-conformal procedure has a
validated formal guarantee even for `local_rate`.

**Rationale:** Supervised layer selection used `2,000` examples sampled from
the whole `probe_set`, including `483` examples later assigned to
`conformal_cal`. Calibration used OOF probe scores while evaluation used a
final full-probe-set model. Finally, the recorded same-sample operating point
selects `2,944/3,307` examples where boundary-inclusive quantile application
requires at least `2,978`; it fails a numerical invariant by `34` examples.

### D-10 - Threshold-transfer diagnostic wording

**Decision:** Describe KROK 8b as a threshold-transfer diagnostic, not as
proof of a single exchangeability cause or restoration of a deployed
guarantee.

**Rationale:** Calibration and evaluation used different probe-score
constructions, which confounds causal interpretation.

### D-11 - Historical `p_true` limitation

**Decision:** Retain `p_true` as part of the recorded router but state that its
Blocker 3 validation is invalid: the recorded mass statistic sums a two-token
softmax and is always one.

**Rationale:** Altering the feature would require a new downstream run.

### D-12 - OOD interpretation

**Decision:** State only that MMLU medical has better recorded point metrics.
Treat pretraining exposure or benchmark familiarity as untested hypotheses.

### D-13 - Mondrian limitation

**Decision:** Preserve the historical Mondrian outputs while disclosing that
its domain mapping omitted observed spelling variants and routed those cases
to a fallback group.

### D-14 - Ablation wording

**Decision:** Refer to `[H, probe, p_true]` in the ablation table as a
refitted feature-subset variant, not the deployed artifact itself.

**Rationale:** Its operating point differs from `results/evaluation.json`.

### D-15 - Calibration methods

**Decision:** Report ROC-isotonic as planned but not executed; `regcal` was not
installed. Report isotonic point estimates without claiming universal
improvement across all external splits.

### D-16 - Contamination statement

**Decision:** Report only supported evidence: no recorded exact cross-dataset
matches and no confirmed near duplicates in the sampled check. Do not claim
complete absence of contamination.

**Post-run addendum:** A CPU-only exact-normalized comparison against the
repurposed MedMCQA validation evaluation set found `1/4,183` validation
questions in `train_ft` and `0` in `probe_set`. Retain recorded evaluation
metrics and disclose this overlap.

### D-17 - Tie-sensitive historical aggregate metrics

**Decision:** Treat historical threshold-derived operating points and
risk-coverage aggregates as recorded point estimates, not corrected
tie-policy-robust recomputations.

**Rationale:** Isotonic calibration creates tied scores, and per-example score
arrays are not present to recalculate thresholds or curves under a declared
tie policy.

## Reproducibility Decision

### D-18 - Evidence available without rerun

**Decision:** The repository is complete for thesis writing from recorded
evidence, not for full numerical replay.

**Rationale:** Checkpoints and extracted feature arrays are not committed.
CPU-only audit outputs may summarize existing JSON results but cannot replace
missing model-level artifacts.
