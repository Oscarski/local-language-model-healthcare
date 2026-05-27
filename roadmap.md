# Final Thesis Roadmap

This document records the completed experiment and the remaining work needed
to write the thesis. It is not a plan to rerun the GPU pipeline.

## Executed Experiment

| Step | Status | Recorded outcome |
|---|---|---|
| Data preparation | Completed | `train_ft=104,765`, `probe_set=13,307` after exact-match removal |
| Fine-tuning | Completed | 4 x L4, bf16, SDPA, `LR=5e-5`; `102,765` train and `2,000` `ft_val` |
| Feature extraction | Completed | `H`, `gap`, `probe` input states, conditional `p_true`; MedMCQA test skipped for labels |
| Layer sweep / probe | Completed | best layer `24`; pooled OOF probe AUROC `0.7920` |
| Routing | Completed | `[H, probe, p_true]`; `routing_train=8,000`, `iso_cal=2,000` |
| Historical thresholding | Completed | `conformal_cal=3,307`; `q_hat=0.5814` at alpha `0.10` |
| Evaluation | Completed | MedMCQA val, MedQA-USMLE, MMLU medical |
| Ablations / figures | Completed | recorded outputs in `results/` and `figures/` |

## Superseded Planning Values

The repository intentionally retains evidence of two pipeline stages:

| Stage | Files / values | Interpretation |
|---|---|---|
| Initial preparation split | `routing_train_idx.json=9,307`, `iso_cal_idx.json=4,000` | Original KROK 2 split preserved as provenance |
| Executed downstream repartition | local index files `8,000 / 2,000 / 3,307` | Split actually used for routing, isotonic calibration, and thresholding |

Other original planning assumptions were superseded during execution:

- MedMCQA test could not be used for labelled evaluation because `cop=-1`;
  MedMCQA val became nominal in-distribution evaluation.
- The successful training configuration was the DDP L4 setup at `LR=5e-5`,
  not the earlier `2e-4` planning value.
- The recorded router deployed three extracted features after dropping `gap`.

## Methodological Audit Findings

1. `local_rate` and `local_accuracy` are distinct; historical thresholding
   concerns the former and does not certify the latter.
2. Historical Blocker 3 for `p_true` was tautological because it summed a
   two-class softmax; the feature's use is recorded, not validated by that
   blocker.
3. At alpha `0.10`, the historical calibration output reports
   `local_rate_on_cal=0.8902` against the nominal `0.90` target; formal
   validity is therefore not claimed. It represents `2,944` selections where
   same-sample quantile inclusion requires at least `2,978` (shortfall `34`).
4. The val threshold-transfer diagnostic documents failed transfer of the
   probe-derived operating point, but does not isolate one causal mechanism.
5. Reconstructed indices show that `483/2,000` layer-sweep examples later
   belong to `conformal_cal`; the recorded formal local-rate CP claim is not
   supportable because calibration was not untouched by supervised selection.
6. Calibration used OOF probe scores while evaluation used the final full-set
   probe, creating an additional score-construction mismatch.
7. OOF probe scores were constructed for all `13,307` probe examples before
   the executed three-way downstream partition, which was stratified by
   correctness outcome. This is a further disclosed dependency of the
   historical calibration design.
8. Raw `near-OOD`/`far-OOD` names are retained only as historical labels;
   final prose uses external MedQA-USMLE and external MMLU medical evaluation.
9. The recorded MMLU superiority is observational; causal explanations are
   hypotheses.
10. Historical Mondrian results use an incomplete domain-name mapping and are
   not corrected post hoc.
11. The completed CPU-only validation overlap check finds one normalized exact
   match from `train_ft` into MedMCQA val and none from `probe_set`; this is a
   disclosed limitation, not a rerun of evaluation.

## Thesis Writing Inputs

| Thesis section | Canonical input |
|---|---|
| Experimental setup | `REPORT.md`, `PROVENANCE.md`, `results/extraction_metadata.json` |
| Recorded results | `REPORT.md`, `results/evaluation.json`, raw figures |
| Ablations | `results/ablations.json`, qualified discussion in `REPORT.md` |
| Methodological limitations | `REPORT.md`, `decisions.md`, `audited/results/claim_status_table.md` |
| Reproducibility boundary | `PROVENANCE.md`, `configs/executed_professor_run.yaml`, `artifacts/professor_run/executed_source.tar.gz`, artifact checksum manifest |

## Completed Finalization Layer

- CPU-only audit material under `audited/` validates recorded evidence without
  overwriting raw outputs.
- Raw and committed audited evidence are verified read-only by default; any
  regeneration writes to a separate explicit output directory.
- Raw artifact hashes, split integrity, threshold-invariant failure, and
  formal-validity status are machine-checkable.
- The audit includes raw-artifact errata, protocol-dependency findings, and
  descriptive accepted-error/failure-capture counts.
- `REPORT.md` and `decisions.md` are the canonical thesis-writing sources;
  raw plots require audited caption/usage guidance.

## Remaining Thesis Work

- Use the canonical report and decision log while drafting the thesis.
- Incorporate the completed validation-overlap audit disclosure in the
  methodology/limitations text, including its pinned MedMCQA revision.

## Not Performed

- No additional fine-tuning.
- No feature re-extraction.
- No routing or conformal refitting.
- No corrected Mondrian rerun.
- No corrected `p_true` rerun.

Those computations require uncommitted model/feature artifacts or a new GPU
execution and are outside the finalization of this repository.
