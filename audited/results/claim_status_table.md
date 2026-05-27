# Claim Status Table

Secondary CPU-only analysis of recorded aggregate evidence; not a model rerun.

| Status | Claim | Basis |
|---|---|---|
| supported_observation | Recorded in-dist local_rate is 0.7927 and local_accuracy is 0.6339. | results/evaluation.json |
| supported_observation | MMLU medical has the best recorded point AUROC and AUGRC. | results/evaluation.json |
| unsupported_interpretation | The recorded pipeline guarantees correct or clinically safe local answers. | The measured operating quantity is local selection frequency. |
| unsupported_interpretation | The recorded local-rate split-CP result has formal validity. | Calibration reuse, score-construction drift and threshold-invariant failure. |
| known_recorded_code_limitation | Blocker 3 validated p_true through Yes/No mass. | The code sums a two-token softmax; recorded mass is tautological. |
| unsupported_interpretation | The val cross-fold result restores deployment validity. | It is a descriptive threshold-transfer diagnostic. |
| not_testable_without_missing_artifacts | Corrected threshold/tie handling changes final model-level results. | Per-example feature arrays and checkpoints are absent. |
