from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "10_audit_recorded_results.py"
GUARD_SCRIPT = ROOT / "scripts" / "recorded_run_guard.py"
OVERLAP_SCRIPT = ROOT / "scripts" / "11_audit_validation_overlap.py"
HISTORICAL_SCRIPTS = [
    "scripts/01_prepare.py",
    "scripts/01b_eda.py",
    "scripts/02_finetune.py",
    "scripts/03_extract.py",
    "scripts/04_probe.py",
    "scripts/05_routing.py",
    "scripts/06_conformal.py",
    "scripts/07_evaluate.py",
    "scripts/07b_exchangeability_check.py",
    "scripts/08_ablations.py",
    "scripts/09_thesis_plots.py",
]


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_recorded_results", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("recorded_run_guard", GUARD_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_overlap_module():
    spec = importlib.util.spec_from_file_location("audit_validation_overlap", OVERLAP_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash_bytes(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_artifact_manifest_is_intact() -> None:
    audit = _load_audit_module()
    result = audit.verify_raw_manifest()
    assert result["checked"] == result["expected_paths"]
    assert result["missing_paths"] == []
    assert result["unexpected_paths"] == []
    assert result["unsafe_paths"] == []
    assert result["mismatches"] == []
    manifest = (ROOT / "artifacts" / "run" / "raw_artifacts.sha256").read_text()
    assert "scripts/config.json" in manifest


def test_raw_artifacts_are_anchored_to_imported_baseline_commit() -> None:
    audit = _load_audit_module()
    result = audit.verify_git_baseline_anchor()
    assert result["raw_paths_unchanged_from_baseline"] is True
    assert result["archived_report_matches_baseline_report"] is True
    assert result["valid"] is True


def test_committed_audit_manifest_is_intact() -> None:
    audit = _load_audit_module()
    result = audit.verify_committed_audit_manifest()
    assert result["checked"] == result["expected_paths"]
    assert result["missing_paths"] == []
    assert result["unexpected_paths"] == []
    assert result["unsafe_paths"] == []
    assert result["mismatches"] == []


def test_split_certificate_distinguishes_initial_and_executed_splits() -> None:
    audit = _load_audit_module()
    certificate = audit.build_split_certificate(ROOT / "data" / "splits")

    assert certificate["train_ft"]["n"] == 104765
    assert certificate["probe_set"]["n"] == 13307
    assert certificate["properties"]["train_ft_probe_disjoint"] is True
    assert certificate["initial_preparation_split"]["routing_train"] == 9307
    assert certificate["initial_preparation_split"]["iso_cal"] == 4000
    assert certificate["properties"]["initial_partitions_probe_set"] is True
    assert certificate["executed_downstream_local_partition"]["routing_train"] == 8000
    assert certificate["executed_downstream_local_partition"]["iso_cal"] == 2000
    assert certificate["executed_downstream_local_partition"]["conformal_cal"] == 3307
    assert certificate["properties"]["executed_partitions_probe_positions"] is True
    assert certificate["valid"] is True


def test_cpu_audit_writes_only_to_requested_output_directory(tmp_path: Path) -> None:
    audit = _load_audit_module()
    before = audit.verify_raw_manifest()
    output_dir = tmp_path / "audited"
    result = audit.generate_audit(output_dir)
    after = audit.verify_raw_manifest()

    assert before == after
    assert result["results_crosscheck"]["checks_passed"] is True
    assert result["raw_integrity"]["mismatches"] == []
    assert (output_dir / "results" / "artifact_manifest.json").exists()
    assert (output_dir / "results" / "split_integrity_certificate.json").exists()
    assert (output_dir / "results" / "formal_validity_assessment.json").exists()
    assert (output_dir / "results" / "threshold_invariant_check.md").exists()
    assert (output_dir / "results" / "figure_usage_catalog.md").exists()
    assert (output_dir / "results" / "claim_status_table.md").exists()
    assert (output_dir / "results" / "accepted_errors_and_failure_capture.md").exists()
    assert (output_dir / "results" / "protocol_dependency_assessment.md").exists()
    assert (output_dir / "results" / "raw_artifact_errata.md").exists()
    assert (output_dir / "figures" / "local_rate_vs_local_accuracy.pdf").exists()
    assert all(str(path).startswith(str(output_dir)) for path in output_dir.rglob("*"))


def test_crosscheck_reports_legacy_flag_without_adopting_claim() -> None:
    audit = _load_audit_module()
    certificate = audit.build_split_certificate(ROOT / "data" / "splits")
    crosscheck = audit.build_crosscheck(ROOT / "results", certificate)

    assert crosscheck["checks_passed"] is True
    assert crosscheck["recorded_best_layer"] == 24
    assert crosscheck["recorded_routing_features"] == ["H", "probe", "p_true"]
    assert crosscheck["recorded_q_hat_alpha_0_10"] == 0.5814
    assert crosscheck["legacy_cp_guarantee_valid_true_splits"] == ["in-dist"]
    assert "no validated formal claim" in crosscheck["legacy_field_interpretation"]


def test_recorded_evaluation_values_are_preserved() -> None:
    evaluation = json.loads((ROOT / "results" / "evaluation.json").read_text())
    assert evaluation["splits"]["in-dist"]["local_rate"] == 0.7927
    assert evaluation["splits"]["in-dist"]["local_acc"] == 0.6339
    assert evaluation["splits"]["near-OOD"]["local_rate"] == 0.9065
    assert evaluation["splits"]["far-OOD"]["auroc"] == 0.7823


def test_formal_validity_assessment_records_overlap_and_threshold_failure() -> None:
    audit = _load_audit_module()
    overlap = audit.build_layer_selection_overlap(ROOT / "data" / "splits")
    threshold = audit.build_threshold_invariant(
        json.loads((ROOT / "results" / "conformal_thresholds.json").read_text())
    )
    validity = audit.build_formal_validity_assessment(overlap, threshold)

    assert overlap["overlap_n"] == 483
    assert overlap["calibration_untouched_by_layer_selection"] is False
    assert threshold["recorded_selected_count"] == 2944
    assert threshold["minimum_selected_count_if_saved_quantile_boundary_is_included"] == 2978
    assert threshold["shortfall"] == 34
    assert threshold["passes_same_sample_inclusion_invariant"] is False
    assert validity["recorded_local_rate_cp_guarantee_supported"] is False
    findings = {item["finding"] for item in validity["reasons"]}
    assert "oof_probe_constructed_before_downstream_calibration_partition" in findings
    assert "downstream_partition_stratified_by_correctness_label" in findings


def test_failure_capture_is_descriptive_and_reproducible() -> None:
    audit = _load_audit_module()
    evaluation = json.loads((ROOT / "results" / "evaluation.json").read_text())
    result = audit.build_failure_capture(evaluation)
    assert result["splits"]["in-dist"]["total_model_errors"] == 1804
    assert result["splits"]["in-dist"]["errors_answered_locally"] == 1214
    assert result["splits"]["in-dist"]["errors_not_answered_locally"] == 590
    assert result["splits"]["in-dist"]["descriptive_failure_capture_rate"] == 0.3271


def test_incomplete_or_unsafe_manifest_is_rejected(tmp_path: Path) -> None:
    audit = _load_audit_module()
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("0" * 64 + "  ../outside.txt\n")
    result = audit.verify_raw_manifest(manifest)

    assert result["unsafe_paths"] == ["../outside.txt"]
    with pytest.raises(audit.AuditValidationError):
        audit._require_valid(result, {"checks_passed": True, "issues": []})


def test_corrupt_split_certificate_fails_before_publication(tmp_path: Path) -> None:
    audit = _load_audit_module()
    copied = tmp_path / "splits"
    shutil.copytree(ROOT / "data" / "splits", copied)
    routing = json.loads((copied / "routing_train_local_idx.json").read_text())
    conformal = json.loads((copied / "conformal_cal_local_idx.json").read_text())
    conformal[0] = routing[0]
    (copied / "conformal_cal_local_idx.json").write_text(json.dumps(conformal))

    with pytest.raises(audit.AuditValidationError):
        audit.build_split_certificate(copied)


def test_audit_refuses_output_inside_repository_or_symlink(tmp_path: Path) -> None:
    audit = _load_audit_module()
    with pytest.raises(audit.AuditValidationError):
        audit.generate_audit(ROOT / "audited")
    link = tmp_path / "inside-link"
    link.symlink_to(ROOT / "audited", target_is_directory=True)
    with pytest.raises(audit.AuditValidationError):
        audit.generate_audit(link)


def test_validation_overlap_report_refuses_repository_output(tmp_path: Path) -> None:
    overlap = _load_overlap_module()
    with pytest.raises(RuntimeError):
        overlap.write_report({"exact_normalized_overlap_counts": {}}, ROOT / "audited")
    link = tmp_path / "inside-link"
    link.symlink_to(ROOT / "audited", target_is_directory=True)
    with pytest.raises(RuntimeError):
        overlap.validate_external_output_root(link)


def test_recorded_output_guard_refuses_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _load_guard_module()
    protected = tmp_path / "evaluation.json"
    protected.write_text("{}")
    monkeypatch.setenv("ALLOW_RECORDED_RUN_OVERWRITE", "1")

    with pytest.raises(SystemExit, match="immutable recorded-run artifact paths"):
        guard.protect_recorded_outputs([protected], "test-step")


def test_historical_blocker_cannot_overwrite_recorded_config() -> None:
    config = ROOT / "scripts" / "config.json"
    before = _hash_bytes(config)
    process = subprocess.run(
        [sys.executable, "scripts/00_verify.py", "--blocker", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "immutable recorded-run artifact paths" in process.stderr + process.stdout
    assert _hash_bytes(config) == before


@pytest.mark.parametrize("script", HISTORICAL_SCRIPTS)
def test_historical_pipeline_entrypoints_are_disabled(script: str) -> None:
    process = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "historical executed source" in process.stderr + process.stdout


@pytest.mark.parametrize(
    "launcher", ["scripts/launch_finetune_ddp.sh", "scripts/launch_extract_ddp.sh"]
)
def test_historical_launchers_are_disabled_before_writes(launcher: str) -> None:
    process = subprocess.run(
        ["bash", launcher],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "Historical launcher disabled" in process.stderr + process.stdout


def test_verify_only_preserves_committed_audit_outputs() -> None:
    audit_files = sorted(path for path in (ROOT / "audited").rglob("*") if path.is_file())
    before = {path: _hash_bytes(path) for path in audit_files}
    process = subprocess.run(
        [sys.executable, "scripts/10_audit_recorded_results.py", "--verify-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert before == {path: _hash_bytes(path) for path in audit_files}


def test_make_verify_is_read_only_for_committed_evidence() -> None:
    evidence_files = [
        ROOT / "scripts" / "config.json",
        *sorted(path for path in (ROOT / "audited").rglob("*") if path.is_file()),
    ]
    before = {path: _hash_bytes(path) for path in evidence_files}
    process = subprocess.run(
        ["make", "verify"],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert before == {path: _hash_bytes(path) for path in evidence_files}


def test_validation_overlap_audit_is_pinned_in_code_and_committed_output() -> None:
    overlap = _load_overlap_module()
    result = json.loads((ROOT / "audited" / "results" / "validation_overlap_audit.json").read_text())
    provenance = result["input_provenance"]
    assert provenance["dataset_revision"] == overlap.DATASET_REVISION
    assert {key: provenance[key] for key in overlap.PINNED_INPUT_PROVENANCE} == overlap.PINNED_INPUT_PROVENANCE
