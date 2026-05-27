# Formal Validity Assessment

**Formal medical-correctness guarantee supported:** no.

**Formal recorded local-rate CP guarantee supported:** no.

- oof_probe_constructed_before_downstream_calibration_partition: OOF probe scores were generated for the full probe_set before routing_train/iso_cal/conformal_cal were formed.
- downstream_partition_stratified_by_correctness_label: The recorded downstream partition used correctness label y for stratification.
- calibration_set_reused_after_supervised_layer_selection: 483 of 2000 layer-sweep examples occur in conformal_cal
- calibration_and_evaluation_score_construction_differ: Calibration used OOF probe scores; evaluation used a final full-probe-set probe.
- same_sample_threshold_inclusion_invariant_failed: Recorded selected=2944, minimum boundary-inclusive=2978, shortfall=34.
