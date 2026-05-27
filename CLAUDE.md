# Medical LLM Routing - Final Thesis Repository

This repository contains a completed professor-run experiment plus an
append-only methodological audit layer. The GPU pipeline has already run; do
not modify or rerun it as part of thesis finalization.

## Canonical Sources

- `REPORT.md` - audited final report for thesis writing.
- `PROVENANCE.md` - raw artifact boundary and replay limitations.
- `decisions.md` - authoritative interpretation decisions.
- `roadmap.md` - completed-run and writing roadmap.
- `artifacts/professor_run/REPORT_ORIGINAL.md` - original execution report.

## Recorded Run

```text
Hardware:        4 x NVIDIA L4
Fine-tuning:     LoRA bf16, SDPA, LR=5e-5
Training data:   102,765 train + 2,000 ft_val
Best layer:      24
Router features: [H, probe, p_true]  (gap dropped)
Downstream:      routing_train=8,000, iso_cal=2,000, conformal_cal=3,307
Evaluation:      MedMCQA val, MedQA-USMLE, MMLU medical
```

## Immutable Raw Artifacts

Never overwrite:

- `results/*.json`
- `figures/*`
- `logs/*`
- `data/splits/*_local_idx.json`
- `artifacts/professor_run/REPORT_ORIGINAL.md`

The raw run is integrity-recorded by
`artifacts/professor_run/raw_artifacts.sha256`.

## Audit Layer

New derived analyses must:

- read only committed recorded outputs;
- write only to `audited/`;
- state that they are secondary analyses of aggregate outputs, not model
  reruns;
- avoid formal medical-safety claims.

Run:

```bash
python scripts/10_audit_recorded_results.py
pytest -q
```

## Interpretation Rules

- `local_rate` means fraction answered locally.
- `local_accuracy` means correctness among answers returned locally.
- The historical thresholding procedure does not certify correct clinical
  answers.
- It also does not support a clean formal `local_rate` guarantee: supervised
  layer selection overlaps `conformal_cal` in 483 of 2,000 sweep samples and
  the raw threshold result fails a same-sample inclusion invariant.
- Historical `p_true` is conditional `P(Yes | {Yes, No})`; its blocker mass
  check was tautological.
- The val cross-fold analysis is a threshold-transfer diagnostic, not proof of
  a single cause or restoration of a deployed guarantee.
- MMLU causal explanations are hypotheses, not recorded findings.

## Missing Replay Artifacts

`checkpoints/` and `data/features/` are not committed. No model-level
recomputation, corrected routing, or corrected conformal analysis is possible
from this repository alone.

Use `configs/executed_professor_run.yaml` for executed protocol values. The
other top-level routing/dataset configs contain superseded planning values.
