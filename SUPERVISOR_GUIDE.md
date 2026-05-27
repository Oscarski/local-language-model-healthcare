# Historical Supervisor Execution Guide

This file documents the GPU phase that was performed by the supervisor before
the final results were imported. It is retained for provenance; it is **not**
a request to rerun training or feature extraction.

## Recorded Execution Outcome

| Item | Recorded outcome |
|---|---|
| Hardware | 4 x NVIDIA L4 |
| Successful configuration | `configs/finetune_config_ddp.yaml`, bf16, SDPA, `LR=5e-5` |
| Fine-tune data | `102,765` train + `2,000` `ft_val` |
| Best checkpoint metric | eval loss `0.6366` at epoch `0.81` |
| Extracted best layer | `24` |
| Downstream split used | `8,000 / 2,000 / 3,307` |

## Historical GPU Workflow

The run consisted of:

```bash
bash scripts/launch_finetune_ddp.sh
bash scripts/launch_extract_ddp.sh
python scripts/04_probe.py
python scripts/05_routing.py
python scripts/06_conformal.py
python scripts/07_evaluate.py
python scripts/07b_exchangeability_check.py
python scripts/08_ablations.py
python scripts/09_thesis_plots.py
```

The resulting tracked JSON summaries, plots, and execution logs are preserved
under `results/`, `figures/`, and `logs/`.

## Artifacts Not Present In Git

The professor machine held additional files documented in the original run
report:

```text
checkpoints/final/
checkpoints/final_probe.pkl
checkpoints/routing_lr.pkl
checkpoints/calibrator.pkl
data/features/*.npz
data/features/*_hidden.npy
data/features/probe_scores_oof.npy
```

These files were excluded from Git and are not available in the current
repository. Therefore the final repository supports thesis writing and
aggregate CPU-only auditing, but not numerical replay of model-dependent
stages.

## Recorded Limitations

- The historical `p_true` blocker accepted a tautological mass statistic; its
  output is preserved but is not valid feature-quality evidence.
- Historical conformal results must be read as local-selection-rate outputs,
  not a guarantee of correct clinical answers.
- The recorded run also does not retain a validated formal local-rate
  conformal claim: supervised layer selection overlapped the later calibration
  subset and the saved threshold fails its same-sample inclusion invariant.
- The original execution narrative is preserved at
  `artifacts/professor_run/REPORT_ORIGINAL.md`; the corrected thesis-facing
  interpretation is in `REPORT.md`.
