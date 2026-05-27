# Protocol Dependency Assessment

**Clean fixed-score split-conformal protocol supported:** no.

- oof_probe_constructed_before_downstream_partition: scripts/04_probe.py generates OOF probe scores over all 13,307 probe examples before scripts/05_routing.py constructs routing_train/iso_cal/conformal_cal.
- downstream_partition_stratified_by_correctness_label: Both train_test_split calls in scripts/05_routing.py use stratify=y or stratify=y_rest.

These are dependencies of the recorded scoring/calibration construction. They are disclosed limitations; no corrected rerun is asserted.
