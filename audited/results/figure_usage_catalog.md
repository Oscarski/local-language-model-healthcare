# Figure Usage Catalog

| Raw figure | Thesis use | Required qualification |
|---|---|---|
| `01_finetune_eval_loss` | usable | Recorded training curve. |
| `02_coverage_risk_curves` | qualified | Recorded/tie-sensitive curve; no formal safety claim. |
| `03_cp_exchangeability` | do not use uncorrected caption | It is a diagnostic, not restored validity. |
| `04_reliability_diagrams`, `05_routing_score_distributions`, `06_layer_sweep` | qualified | Recorded descriptive outputs. |
| `07_signal_subset_ablation` | qualified | Feature-subset refit, not deployed artifact replay. |
| `08_mondrian_vs_split_cp`, `09_per_domain_coverage` | do not use as corrected result | Historical incomplete domain map. |
| `10_utility_vs_alpha`, `11_calibration_ablation` | qualified | Recorded threshold-dependent analyses only. |
