# Safe Enough to Answer Locally?
## A Methodological Audit of Routing for an Edge-Deployed Medical LLM

> Bachelor's thesis project - Oskar Koscianski
> Recorded model run: Mistral-7B-Instruct-v0.3, LoRA fine-tuning, 4 x NVIDIA L4

This repository preserves a completed experiment in which a fine-tuned medical
multiple-choice LLM was equipped with uncertainty-based routing. It is now
organized as the source package for writing the final thesis: the empirical
results are preserved, while the thesis critically audits what the recorded
conformal procedure can and cannot support.

The central distinction is:

- `local_rate = P(answer locally)`: fraction of questions answered locally.
- `local_accuracy = P(correct | answer locally)`: quality of those local answers.

The historical pipeline selected a threshold for `local_rate`. It does **not**
establish a formal guarantee of medical-answer correctness. The recorded run
also does not support a clean formal split-conformal guarantee for
`local_rate`: layer-selection data overlapped its later calibration subset and
the recorded threshold failed a same-sample inclusion invariant.

## Recorded Experiment

### Data Flow

```text
MedMCQA train, single-answer items: 120,765
  |-- train_ft pool: 104,765
  |     |-- fine-tune train: 102,765
  |     `-- ft_val: 2,000 (early stopping)
  `-- initial probe candidate split: 16,000
        |-- removed normalized exact matches: 2,693
        `-- retained probe_set: 13,307
        |-- routing_train: 8,000
        |-- iso_cal: 2,000
        `-- conformal_cal: 3,307

Evaluation:
  MedMCQA val: 4,183       nominal in-distribution evaluation
  MedQA-USMLE test: 1,273  external evaluation
  MMLU medical: 945         external evaluation
```

MedMCQA's official test split has hidden labels (`cop=-1`) and was not usable
for labelled evaluation.

### Executed Configuration

| Item | Recorded value |
|---|---|
| Fine-tuning hardware | 4 x NVIDIA L4 |
| Training precision / attention | bf16 / SDPA |
| Successful learning rate | `5e-5` |
| Best eval loss | `0.6366` at epoch `0.81` |
| Best hidden-state layer | `24` |
| Recorded routing features | `[H, probe, p_true]` |
| Dropped feature | `gap` (`|r(H, gap)| = 0.9564`) |
| Historical threshold at alpha 0.10 | `q_hat=0.5814`, decision threshold `0.4186` |

## Recorded Results

These are point estimates from the immutable raw result files.

| Split | n | Standalone accuracy | AUROC | AUGRC | local_rate | local_accuracy |
|---|---:|---:|---:|---:|---:|---:|
| MedMCQA val | 4,183 | 0.5687 | 0.7429 | 0.15619 | 0.7927 | 0.6339 |
| MedQA-USMLE | 1,273 | 0.5664 | 0.6949 | 0.16899 | 0.9065 | 0.5832 |
| MMLU medical | 945 | 0.6857 | 0.7823 | 0.09549 | 0.9206 | 0.7172 |

## Audit Findings

1. The recorded threshold operates on local-answer frequency, not on the
   correctness or clinical safety of locally returned answers.
2. The recorded Blocker 3 for `p_true` calculated a two-token softmax and then
   summed it; its `mass=1.0` pass is tautological. The signal remained in the
   historical router, and the recorded ablation shows no observed AUROC gain
   over `H+probe`.
3. The recorded split-conformal interpretation is not formally supported:
   `483` of the `2,000` supervised layer-sweep examples later belonged to
   `conformal_cal`, and calibration used OOF probe scores while evaluation
   used a final probe fit on the full probe set.
4. The threshold-transfer diagnostic reports `local_rate=0.7927` when a
   probe-derived threshold is applied to MedMCQA val and `0.9144` in val
   cross-fold analysis. This documents a transfer problem; it does not isolate
   a single causal explanation or restore a deployed safety guarantee.
5. The historical threshold output reports `2,944/3,307` local decisions
   (`local_rate_on_cal=0.8902`) at its nominal `0.90` target on the
   calibration sample itself; inclusion of the quantile boundary required at
   least `2,978/3,307`. It fails an internal numerical sanity check.
6. MMLU performs best on the recorded point metrics. Any explanation based on
   pretraining familiarity is a hypothesis, not an experimental result.
7. The historical Mondrian analysis used an incomplete subject-name mapping;
   its numbers are retained as recorded artifacts, not corrected results.
8. A CPU-only post-run overlap check found `1/4,183` normalized exact
   train-to-validation match and `0` probe-to-validation matches; the recorded
   evaluation is retained with this disclosure.

## Artifacts

| Location | Meaning |
|---|---|
| `results/`, `figures/`, `logs/` | Immutable outputs of the professor run |
| `artifacts/professor_run/REPORT_ORIGINAL.md` | Original, unedited run report |
| `artifacts/professor_run/raw_artifacts.sha256` | Integrity manifest |
| `REPORT.md` | Canonical audited report for thesis writing |
| `configs/executed_professor_run.yaml` | Citation-safe manifest of the executed protocol |
| `decisions.md` | Audited methodological decision log |
| `roadmap.md` | Completed-run and writing roadmap |
| `audited/` | CPU-only derived audit outputs, not new experiments |

The repository does not contain the model checkpoints or extracted feature
arrays required to replay downstream computations. See `PROVENANCE.md`.

## CPU-Only Audit Commands

```bash
python scripts/10_audit_recorded_results.py
python scripts/11_audit_validation_overlap.py
pytest -q
```

These commands inspect committed evidence and create files only under
`audited/`. They do not rerun model training, inference, routing fitting, or
conformal calibration.

## Thesis Use

The canonical description of the completed experiment and its limitations is
in `REPORT.md`. Raw figures are historical evidence; consult
`audited/results/figure_usage_catalog.md` before using one in the thesis. The
contribution is a rigorous case study of an uncertainty routing pipeline and
an audit of why a formal-sounding selection procedure does not, by itself,
establish safe local medical answering.
