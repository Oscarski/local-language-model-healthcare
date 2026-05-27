"""CPU-only audit of immutable aggregate outputs from the recorded GPU run.

This script validates provenance and cross-file consistency before publishing
secondary material under ``audited/``. It does not load model checkpoints or
feature arrays and never writes to ``results/``, ``figures/`` or ``logs/``.
Committed audit outputs can be verified read-only with ``--verify-only``;
regeneration requires an explicit output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent.parent
RAW_MANIFEST = ROOT / "artifacts" / "professor_run" / "raw_artifacts.sha256"
AUDITED_MANIFEST = ROOT / "artifacts" / "audited_layer" / "committed_outputs.sha256"
EXECUTED_CONFIG = ROOT / "configs" / "executed_professor_run.yaml"
EXECUTED_SOURCE_ARCHIVE = ROOT / "artifacts" / "professor_run" / "executed_source.tar.gz"
BASELINE_COMMIT = "a34323d"
BASELINE_RAW_PATHS = ["results", "figures", "logs", "data/splits", "scripts/config.json"]


class AuditValidationError(RuntimeError):
    """Raised when recorded evidence is incomplete or internally inconsistent."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_external_output_root(output_root: Path) -> Path:
    """Reject audit regeneration into this repository, including symlink aliases."""
    resolved = output_root.expanduser().resolve()
    root = ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise AuditValidationError(
            "Audit regeneration output must be outside the repository; "
            "committed audited outputs are immutable submission evidence."
        )
    return resolved


def _expected_immutable_paths() -> set[str]:
    expected: set[str] = set()
    for directory in ("results", "figures", "logs"):
        expected.update(
            str(path.relative_to(ROOT))
            for path in (ROOT / directory).glob("*")
            if path.is_file()
        )
    expected.update(
        {
            "data/splits/meta.json",
            "data/splits/train_ft_idx.json",
            "data/splits/probe_idx.json",
            "data/splits/routing_train_idx.json",
            "data/splits/iso_cal_idx.json",
            "data/splits/routing_train_local_idx.json",
            "data/splits/iso_cal_local_idx.json",
            "data/splits/conformal_cal_local_idx.json",
            "scripts/config.json",
            "artifacts/professor_run/REPORT_ORIGINAL.md",
        }
    )
    return expected


def verify_raw_manifest(manifest_path: Path = RAW_MANIFEST) -> dict[str, Any]:
    entries: dict[str, str] = {}
    malformed: list[str] = []
    unsafe_paths: list[str] = []
    duplicates: list[str] = []
    for line_no, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            malformed.append(f"line {line_no}")
            continue
        expected, relative_path = parts[0], parts[1].strip()
        path_obj = Path(relative_path)
        if path_obj.is_absolute() or ".." in path_obj.parts:
            unsafe_paths.append(relative_path)
            continue
        if relative_path in entries:
            duplicates.append(relative_path)
        entries[relative_path] = expected

    expected_paths = _expected_immutable_paths()
    manifest_paths = set(entries)
    mismatches: list[dict[str, str]] = []
    for relative_path, expected_hash in sorted(entries.items()):
        path = ROOT / relative_path
        actual_hash = _sha256(path) if path.exists() else "MISSING"
        if actual_hash != expected_hash:
            mismatches.append(
                {"path": relative_path, "expected": expected_hash, "actual": actual_hash}
            )

    try:
        manifest_display = str(manifest_path.relative_to(ROOT))
    except ValueError:
        manifest_display = str(manifest_path)
    return {
        "manifest": manifest_display,
        "checked": len(entries),
        "expected_paths": len(expected_paths),
        "missing_paths": sorted(expected_paths - manifest_paths),
        "unexpected_paths": sorted(manifest_paths - expected_paths),
        "malformed_lines": malformed,
        "unsafe_paths": unsafe_paths,
        "duplicate_paths": duplicates,
        "mismatches": mismatches,
    }


def verify_git_baseline_anchor() -> dict[str, Any]:
    """Check raw evidence against the imported professor-run baseline commit."""
    diff = subprocess.run(
        ["git", "diff", "--quiet", BASELINE_COMMIT, "--", *BASELINE_RAW_PATHS],
        cwd=ROOT,
        check=False,
    )
    original = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:REPORT.md"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    archived = ROOT / "artifacts" / "professor_run" / "REPORT_ORIGINAL.md"
    return {
        "baseline_commit": BASELINE_COMMIT,
        "raw_paths_unchanged_from_baseline": diff.returncode == 0,
        "archived_report_matches_baseline_report": (
            original.returncode == 0 and original.stdout == archived.read_bytes()
        ),
        "valid": (
            diff.returncode == 0
            and original.returncode == 0
            and original.stdout == archived.read_bytes()
        ),
    }


def verify_committed_audit_manifest(
    manifest_path: Path = AUDITED_MANIFEST,
) -> dict[str, Any]:
    entries: dict[str, str] = {}
    malformed: list[str] = []
    unsafe_paths: list[str] = []
    duplicates: list[str] = []
    for line_no, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            malformed.append(f"line {line_no}")
            continue
        expected, relative_path = parts[0], parts[1].strip()
        path_obj = Path(relative_path)
        if path_obj.is_absolute() or ".." in path_obj.parts:
            unsafe_paths.append(relative_path)
            continue
        if relative_path in entries:
            duplicates.append(relative_path)
        entries[relative_path] = expected

    expected_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "audited").rglob("*")
        if path.is_file()
    }
    expected_paths.update(
        {
            "configs/executed_professor_run.yaml",
            "scripts/10_audit_recorded_results.py",
            "scripts/11_audit_validation_overlap.py",
            "artifacts/professor_run/EXECUTED_SOURCE.md",
            "artifacts/professor_run/executed_source.sha256",
            "artifacts/professor_run/executed_source.tar.gz",
        }
    )
    manifest_paths = set(entries)
    mismatches: list[dict[str, str]] = []
    for relative_path, expected_hash in sorted(entries.items()):
        path = ROOT / relative_path
        actual_hash = _sha256(path) if path.exists() else "MISSING"
        if actual_hash != expected_hash:
            mismatches.append(
                {"path": relative_path, "expected": expected_hash, "actual": actual_hash}
            )
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "checked": len(entries),
        "expected_paths": len(expected_paths),
        "missing_paths": sorted(expected_paths - manifest_paths),
        "unexpected_paths": sorted(manifest_paths - expected_paths),
        "malformed_lines": malformed,
        "unsafe_paths": unsafe_paths,
        "duplicate_paths": duplicates,
        "mismatches": mismatches,
    }


def _unique_ints(path: Path) -> list[int]:
    values = _read_json(path)
    if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
        raise AuditValidationError(f"{path.relative_to(ROOT)} is not an integer index list")
    if len(values) != len(set(values)):
        raise AuditValidationError(f"{path.relative_to(ROOT)} contains duplicate indices")
    return values


def build_split_certificate(splits_dir: Path) -> dict[str, Any]:
    train_ft = _unique_ints(splits_dir / "train_ft_idx.json")
    probe = _unique_ints(splits_dir / "probe_idx.json")
    initial_routing = _unique_ints(splits_dir / "routing_train_idx.json")
    initial_iso = _unique_ints(splits_dir / "iso_cal_idx.json")
    routing_local = _unique_ints(splits_dir / "routing_train_local_idx.json")
    iso_local = _unique_ints(splits_dir / "iso_cal_local_idx.json")
    conformal_local = _unique_ints(splits_dir / "conformal_cal_local_idx.json")
    meta = _read_json(splits_dir / "meta.json")

    probe_set = set(probe)
    local_domain = set(range(len(probe)))
    executed_sets = [set(routing_local), set(iso_local), set(conformal_local)]
    properties = {
        "train_ft_probe_disjoint": set(train_ft).isdisjoint(probe_set),
        "initial_partitions_probe_set": (
            set(initial_routing).isdisjoint(initial_iso)
            and set(initial_routing) | set(initial_iso) == probe_set
        ),
        "executed_local_indices_in_range": set.union(*executed_sets) <= local_domain,
        "executed_partitions_probe_positions": (
            executed_sets[0].isdisjoint(executed_sets[1])
            and executed_sets[0].isdisjoint(executed_sets[2])
            and executed_sets[1].isdisjoint(executed_sets[2])
            and set.union(*executed_sets) == local_domain
        ),
        "metadata_sizes_match_indices": (
            meta["n_train_ft"] == len(train_ft)
            and meta["n_probe"] == len(probe)
            and meta["n_routing_train"] == len(initial_routing)
            and meta["n_iso_cal"] == len(initial_iso)
        ),
    }
    certificate = {
        "train_ft": {"n": len(train_ft), "unique": True},
        "probe_set": {"n": len(probe), "unique": True},
        "initial_preparation_split": {
            "routing_train": len(initial_routing),
            "iso_cal": len(initial_iso),
            "status": "superseded_for_recorded_downstream_execution",
        },
        "executed_downstream_local_partition": {
            "routing_train": len(routing_local),
            "iso_cal": len(iso_local),
            "conformal_cal": len(conformal_local),
            "status": "used_in_recorded_downstream_results",
        },
        "properties": properties,
        "valid": all(properties.values()),
    }
    if not certificate["valid"]:
        failed = [name for name, value in properties.items() if not value]
        raise AuditValidationError(f"Split-integrity validation failed: {', '.join(failed)}")
    return certificate


def build_layer_selection_overlap(splits_dir: Path) -> dict[str, Any]:
    import numpy as np

    n_probe = len(_unique_ints(splits_dir / "probe_idx.json"))
    conformal_local = set(_unique_ints(splits_dir / "conformal_cal_local_idx.json"))
    sampled = set(
        np.random.default_rng(42).choice(n_probe, size=min(2000, n_probe), replace=False).tolist()
    )
    overlap = sampled & conformal_local
    return {
        "selection_stage": "supervised_hidden_layer_sweep",
        "seed": 42,
        "probe_set_n": n_probe,
        "layer_sweep_sample_n": len(sampled),
        "conformal_cal_n": len(conformal_local),
        "overlap_n": len(overlap),
        "calibration_untouched_by_layer_selection": len(overlap) == 0,
    }


def _recover_count(rate: float, denominator: int, decimals: int = 4) -> int:
    candidates = [
        count for count in range(denominator + 1) if round(count / denominator, decimals) == rate
    ]
    if len(candidates) != 1:
        raise AuditValidationError(
            f"Cannot uniquely recover count for rounded rate={rate}, n={denominator}"
        )
    return candidates[0]


def build_threshold_invariant(thresholds: dict[str, Any]) -> dict[str, Any]:
    alpha = float(thresholds["primary_alpha"])
    n = int(thresholds["n_conformal_cal"])
    row = thresholds["split_cp"][f"alpha_{alpha}"]
    selected = _recover_count(float(row["local_rate_on_cal"]), n)
    minimum_boundary_inclusive = math.ceil((n + 1) * (1.0 - alpha))
    return {
        "alpha": alpha,
        "n_conformal_cal": n,
        "recorded_local_rate_on_cal": row["local_rate_on_cal"],
        "recorded_selected_count": selected,
        "minimum_selected_count_if_saved_quantile_boundary_is_included": minimum_boundary_inclusive,
        "shortfall": max(0, minimum_boundary_inclusive - selected),
        "passes_same_sample_inclusion_invariant": selected >= minimum_boundary_inclusive,
        "interpretation": (
            "A failure is consistent with boundary/tie or floating-point complement "
            "handling in the historical application; raw results are not recomputed."
        ),
    }


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - radius, 4), round(center + radius, 4)]


def build_operating_intervals(evaluation: dict[str, Any]) -> dict[str, Any]:
    intervals: dict[str, Any] = {}
    for split, values in evaluation["splits"].items():
        n = int(values["n"])
        n_local = int(values["n_local"])
        correct = _recover_count(values["standalone_acc"], n)
        local_correct = _recover_count(values["local_acc"], n_local)
        intervals[split] = {
            "standalone_accuracy": {
                "count": f"{correct}/{n}",
                "estimate": values["standalone_acc"],
                "wilson_95_ci": _wilson_interval(correct, n),
            },
            "local_rate": {
                "count": f"{n_local}/{n}",
                "estimate": values["local_rate"],
                "wilson_95_ci": _wilson_interval(n_local, n),
            },
            "local_accuracy": {
                "count": f"{local_correct}/{n_local}",
                "estimate": values["local_acc"],
                "wilson_95_ci": _wilson_interval(local_correct, n_local),
            },
        }
    return intervals


def build_failure_capture(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Describe errors withheld from local answers at the recorded operating point."""
    captured: dict[str, Any] = {}
    for split, values in evaluation["splits"].items():
        n = int(values["n"])
        n_local = int(values["n_local"])
        correct = _recover_count(values["standalone_acc"], n)
        local_correct = _recover_count(values["local_acc"], n_local)
        total_errors = n - correct
        local_errors = n_local - local_correct
        errors_not_answered_locally = total_errors - local_errors
        captured[split] = {
            "total_model_errors": total_errors,
            "errors_answered_locally": local_errors,
            "errors_not_answered_locally": errors_not_answered_locally,
            "descriptive_failure_capture_rate": round(
                errors_not_answered_locally / total_errors, 4
            ),
        }
    return {
        "interpretation": (
            "Descriptive secondary statistic reconstructed from recorded aggregate counts; "
            "it is not a clinical safety guarantee or a newly evaluated policy."
        ),
        "splits": captured,
    }


def build_protocol_dependency_assessment() -> dict[str, Any]:
    return {
        "clean_fixed_score_split_conformal_protocol_supported": False,
        "recorded_dependencies": [
            {
                "finding": "oof_probe_constructed_before_downstream_partition",
                "evidence": (
                    "scripts/04_probe.py generates OOF probe scores over all 13,307 "
                    "probe examples before scripts/05_routing.py constructs "
                    "routing_train/iso_cal/conformal_cal."
                ),
            },
            {
                "finding": "downstream_partition_stratified_by_correctness_label",
                "evidence": (
                    "Both train_test_split calls in scripts/05_routing.py use "
                    "stratify=y or stratify=y_rest."
                ),
            },
        ],
        "interpretation": (
            "These are dependencies of the recorded scoring/calibration construction. "
            "They are disclosed limitations; no corrected rerun is asserted."
        ),
    }


def build_crosscheck(results_dir: Path, certificate: dict[str, Any]) -> dict[str, Any]:
    evaluation = _read_json(results_dir / "evaluation.json")
    routing = _read_json(results_dir / "routing_metrics.json")
    thresholds = _read_json(results_dir / "conformal_thresholds.json")
    extraction = _read_json(results_dir / "extraction_metadata.json")
    layer = _read_json(results_dir / "best_layer.json")
    ablations = _read_json(results_dir / "ablations.json")
    executed_config = yaml.safe_load(EXECUTED_CONFIG.read_text())
    issues: list[str] = []

    if layer["best_layer"] != extraction["best_layer"]:
        issues.append("best_layer mismatch between best_layer.json and extraction_metadata.json")
    executed = certificate["executed_downstream_local_partition"]
    expected_sizes = {key: executed[key] for key in ("routing_train", "iso_cal", "conformal_cal")}
    if routing["splits"] != expected_sizes:
        issues.append("routing split sizes disagree with executed local index files")
    recorded_features = routing["routing_feature_names"]
    for name, value in (
        ("evaluation", evaluation["routing_feature_names"]),
        ("conformal_thresholds", thresholds["routing_feature_names"]),
        ("ablations default", ablations["default_routing_features"]),
    ):
        if value != recorded_features:
            issues.append(f"routing features disagree in {name}")
    q_hat = thresholds["split_cp"]["alpha_0.1"]["q_hat"]
    if evaluation["q_hat"] != q_hat or ablations["q_hat_global"] != q_hat:
        issues.append("q_hat disagrees across conformal, evaluation and ablation outputs")

    stem_map = {"in-dist": "val", "near-OOD": "test_medqa", "far-OOD": "test_mmlu"}
    cp_global = ablations["ablations"]["table3_cp_variant"]["split_global"]
    ablation_map = {"in-dist": "in_dist", "near-OOD": "near_OOD", "far-OOD": "far_OOD"}
    for split, values in evaluation["splits"].items():
        if round(values["n_local"] / values["n"], 4) != values["local_rate"]:
            issues.append(f"{split}: local_rate does not equal n_local/n")
        if round(1.0 - values["local_rate"], 4) != values["escalation_rate"]:
            issues.append(f"{split}: escalation rate does not complement local_rate")
        if extraction["splits"][stem_map[split]]["accuracy"] != values["standalone_acc"]:
            issues.append(f"{split}: extraction accuracy disagrees with evaluation")
        row = cp_global[ablation_map[split]]
        for metric in ("n_local", "local_rate", "local_acc", "escalation_rate"):
            if row[metric] != values[metric]:
                issues.append(f"{split}: global CP ablation disagrees on {metric}")
    if executed_config["routing"]["features"] != recorded_features:
        issues.append("citation config routing features disagree with recorded results")
    if executed_config["routing"]["best_layer"] != layer["best_layer"]:
        issues.append("citation config best layer disagrees with recorded results")
    if executed_config["routing"]["recorded_q_hat_alpha_0_10"] != q_hat:
        issues.append("citation config q_hat disagrees with recorded results")
    citation_sizes = {
        key: executed_config["routing"][key]
        for key in ("routing_train", "iso_cal", "conformal_cal")
    }
    if citation_sizes != expected_sizes:
        issues.append("citation config downstream partition disagrees with recorded indices")

    return {
        "checks_passed": not issues,
        "issues": issues,
        "recorded_best_layer": layer["best_layer"],
        "recorded_routing_features": recorded_features,
        "recorded_q_hat_alpha_0_10": q_hat,
        "citation_config_sha256": _sha256(EXECUTED_CONFIG),
        "legacy_cp_guarantee_valid_true_splits": [
            split for split, data in evaluation["splits"].items()
            if data.get("cp_guarantee_valid") is True
        ],
        "legacy_field_interpretation": "Raw legacy label only; no validated formal claim is adopted.",
    }


def build_formal_validity_assessment(
    overlap: dict[str, Any], threshold_invariant: dict[str, Any]
) -> dict[str, Any]:
    return {
        "medical_correctness_guarantee_supported": False,
        "recorded_local_rate_cp_guarantee_supported": False,
        "reasons": [
            {
                "finding": "oof_probe_constructed_before_downstream_calibration_partition",
                "evidence": (
                    "OOF probe scores were generated for the full probe_set before "
                    "routing_train/iso_cal/conformal_cal were formed."
                ),
            },
            {
                "finding": "downstream_partition_stratified_by_correctness_label",
                "evidence": (
                    "The recorded downstream partition used correctness label y "
                    "for stratification."
                ),
            },
            {
                "finding": "calibration_set_reused_after_supervised_layer_selection",
                "evidence": (
                    f"{overlap['overlap_n']} of {overlap['layer_sweep_sample_n']} "
                    "layer-sweep examples occur in conformal_cal"
                ),
            },
            {
                "finding": "calibration_and_evaluation_score_construction_differ",
                "evidence": "Calibration used OOF probe scores; evaluation used a final full-probe-set probe.",
            },
            {
                "finding": "same_sample_threshold_inclusion_invariant_failed",
                "evidence": (
                    f"Recorded selected={threshold_invariant['recorded_selected_count']}, "
                    "minimum boundary-inclusive="
                    f"{threshold_invariant['minimum_selected_count_if_saved_quantile_boundary_is_included']}, "
                    f"shortfall={threshold_invariant['shortfall']}."
                ),
            },
        ],
    }


def build_claims() -> list[dict[str, str]]:
    return [
        {"status": "supported_observation", "claim": "Recorded in-dist local_rate is 0.7927 and local_accuracy is 0.6339.", "basis": "results/evaluation.json"},
        {"status": "supported_observation", "claim": "MMLU medical has the best recorded point AUROC and AUGRC.", "basis": "results/evaluation.json"},
        {"status": "supported_observation", "claim": "The recorded policy withheld some model errors from local answering at its saved threshold.", "basis": "Reconstructed aggregate counts in accepted_errors_and_failure_capture.md."},
        {"status": "unsupported_interpretation", "claim": "The recorded pipeline guarantees correct or clinically safe local answers.", "basis": "The measured operating quantity is local selection frequency."},
        {"status": "unsupported_interpretation", "claim": "The recorded local-rate split-CP result has formal validity.", "basis": "Score construction precedes the downstream partition, outcome-stratified partitioning, calibration reuse, score-construction drift and threshold-invariant failure."},
        {"status": "known_recorded_code_limitation", "claim": "Blocker 3 validated p_true through Yes/No mass.", "basis": "The code sums a two-token softmax; recorded mass is tautological."},
        {"status": "known_recorded_code_limitation", "claim": "near-OOD/far-OOD labels denote verified distribution-shift severity.", "basis": "They are historical labels; thesis-facing text uses external MedQA and external MMLU medical evaluations."},
        {"status": "unsupported_interpretation", "claim": "The val cross-fold result restores deployment validity.", "basis": "It is a descriptive threshold-transfer diagnostic."},
        {"status": "not_testable_without_missing_artifacts", "claim": "Corrected threshold/tie handling changes final model-level results.", "basis": "Per-example feature arrays and checkpoints are absent."},
    ]


def _write_markdown(output: Path, intervals: dict[str, Any], claims: list[dict[str, str]],
                    certificate: dict[str, Any], validity: dict[str, Any],
                    threshold: dict[str, Any], failure_capture: dict[str, Any],
                    protocol: dict[str, Any]) -> None:
    claim_lines = ["# Claim Status Table", "", "Secondary CPU-only analysis of recorded aggregate evidence; not a model rerun.", "", "| Status | Claim | Basis |", "|---|---|---|"]
    claim_lines.extend(f"| {r['status']} | {r['claim']} | {r['basis']} |" for r in claims)
    (output / "claim_status_table.md").write_text("\n".join(claim_lines) + "\n")

    interval_lines = ["# Operating-Point Intervals", "", "Wilson 95% intervals reconstructed from recorded rounded rates and counts.", "", "| Split | Quantity | Count | Estimate | Wilson 95% CI |", "|---|---|---:|---:|---|"]
    for split, metrics in intervals.items():
        for metric, value in metrics.items():
            interval_lines.append(f"| {split} | {metric} | {value['count']} | {value['estimate']:.4f} | [{value['wilson_95_ci'][0]:.4f}, {value['wilson_95_ci'][1]:.4f}] |")
    (output / "operating_point_intervals.md").write_text("\n".join(interval_lines) + "\n")

    initial = certificate["initial_preparation_split"]
    executed = certificate["executed_downstream_local_partition"]
    (output / "config_execution_reconciliation.md").write_text(
        "# Config and Execution Reconciliation\n\n"
        "The initial split is retained as provenance; the local-index partition is the recorded downstream execution.\n\n"
        "| Stage | routing_train | iso_cal | conformal_cal | Use |\n|---|---:|---:|---:|---|\n"
        f"| Initial preparation | {initial['routing_train']:,} | {initial['iso_cal']:,} | - | Superseded downstream |\n"
        f"| Recorded downstream | {executed['routing_train']:,} | {executed['iso_cal']:,} | {executed['conformal_cal']:,} | Used in raw results |\n\n"
        "Use `configs/executed_professor_run.yaml` for citation; older configs are pre-run plans.\n"
    )
    reason_lines = "\n".join(f"- {r['finding']}: {r['evidence']}" for r in validity["reasons"])
    (output / "formal_validity_assessment.md").write_text(
        "# Formal Validity Assessment\n\n"
        "**Formal medical-correctness guarantee supported:** no.\n\n"
        "**Formal recorded local-rate CP guarantee supported:** no.\n\n"
        f"{reason_lines}\n"
    )
    (output / "threshold_invariant_check.md").write_text(
        "# Same-Sample Threshold Inclusion Check\n\n"
        "| alpha | n calibration | Recorded selected | Minimum boundary-inclusive | Shortfall | Pass |\n"
        "|---:|---:|---:|---:|---:|---|\n"
        f"| {threshold['alpha']:.2f} | {threshold['n_conformal_cal']:,} | {threshold['recorded_selected_count']:,} | "
        f"{threshold['minimum_selected_count_if_saved_quantile_boundary_is_included']:,} | {threshold['shortfall']} | "
        f"{'yes' if threshold['passes_same_sample_inclusion_invariant'] else 'no'} |\n\n"
        f"{threshold['interpretation']}\n"
    )
    failure_lines = [
        "# Accepted Errors and Descriptive Failure Capture",
        "",
        failure_capture["interpretation"],
        "",
        "| Recorded split label | Total model errors | Errors answered locally | Errors not answered locally | Descriptive fraction not answered locally |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in failure_capture["splits"].items():
        failure_lines.append(
            f"| {split} | {row['total_model_errors']} | {row['errors_answered_locally']} | "
            f"{row['errors_not_answered_locally']} | {row['descriptive_failure_capture_rate']:.4f} |"
        )
    (output / "accepted_errors_and_failure_capture.md").write_text(
        "\n".join(failure_lines) + "\n"
    )
    protocol_lines = [
        "# Protocol Dependency Assessment",
        "",
        "**Clean fixed-score split-conformal protocol supported:** no.",
        "",
    ]
    protocol_lines.extend(
        f"- {row['finding']}: {row['evidence']}" for row in protocol["recorded_dependencies"]
    )
    protocol_lines.extend(["", protocol["interpretation"]])
    (output / "protocol_dependency_assessment.md").write_text(
        "\n".join(protocol_lines) + "\n"
    )
    (output / "raw_artifact_errata.md").write_text(
        "# Raw Artifact Errata\n\n"
        "Raw professor-run JSONs, plots and logs are immutable evidence and have not "
        "been rewritten. Use these corrections when citing them:\n\n"
        "- `cp_guarantee_valid: true` in raw evaluation output is a legacy field; it "
        "is not an adopted formal guarantee.\n"
        "- Raw labels `near-OOD` and `far-OOD` are historical naming. Thesis text "
        "should say external MedQA evaluation and external MMLU medical evaluation.\n"
        "- Figures derived from the historical thresholding procedure are descriptive "
        "outputs only and cannot support a medical-safety conclusion.\n"
        "- The recorded Mondrian output uses an incomplete historical domain map and "
        "is not a corrected per-domain result.\n"
    )
    (output / "figure_usage_catalog.md").write_text(
        "# Figure Usage Catalog\n\n"
        "| Raw figure | Thesis use | Required qualification |\n|---|---|---|\n"
        "| `01_finetune_eval_loss` | usable | Recorded training curve. |\n"
        "| `02_coverage_risk_curves` | qualified | Recorded/tie-sensitive curve; no formal safety claim. |\n"
        "| `03_cp_exchangeability` | do not use uncorrected caption | It is a diagnostic, not restored validity. |\n"
        "| `04_reliability_diagrams`, `05_routing_score_distributions`, `06_layer_sweep` | qualified | Recorded descriptive outputs. |\n"
        "| `07_signal_subset_ablation` | qualified | Feature-subset refit, not deployed artifact replay. |\n"
        "| `08_mondrian_vs_split_cp`, `09_per_domain_coverage` | do not use as corrected result | Historical incomplete domain map. |\n"
        "| `10_utility_vs_alpha`, `11_calibration_ablation` | qualified | Recorded threshold-dependent analyses only. |\n"
    )


def _make_figures(output: Path, evaluation: dict[str, Any], exchange: dict[str, Any],
                  ablations: dict[str, Any]) -> None:
    cache = Path(tempfile.gettempdir()) / "diploma-thesis-audit-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdf_metadata = {
        "Creator": "diploma-thesis recorded-result audit",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }

    output.mkdir(parents=True, exist_ok=True)
    names = list(evaluation["splits"])
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - 0.175 for i in x], [evaluation["splits"][n]["local_rate"] for n in names], 0.35, label="local_rate")
    ax.bar([i + 0.175 for i in x], [evaluation["splits"][n]["local_acc"] for n in names], 0.35, label="local_accuracy")
    ax.set(xticks=list(x), xticklabels=names, ylim=(0, 1), ylabel="Recorded proportion", title="Local selection frequency is not local correctness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "local_rate_vs_local_accuracy.pdf", metadata=pdf_metadata)
    fig.savefig(output / "local_rate_vs_local_accuracy.png", dpi=200)
    plt.close(fig)

    a, b = exchange["setup_A_probe_based"], exchange["setup_B_val_kfold"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Probe-derived\nthreshold", "Val cross-fold\ndiagnostic"], [a["local_rate"], b["local_rate"]])
    ax.axhline(exchange["cp_target"], linestyle="--", color="black", label="nominal local-rate target")
    ax.set(ylim=(0, 1), ylabel="Empirical local_rate on MedMCQA val", title="Recorded threshold-transfer diagnostic (descriptive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "threshold_transfer_diagnostic.pdf", metadata=pdf_metadata)
    fig.savefig(output / "threshold_transfer_diagnostic.png", dpi=200)
    plt.close(fig)

    subset = ablations["ablations"]["table1_signal_subset"]
    rows = ["H", "probe", "p_true", "H+probe", "H+probe+p_true", "H+gap+probe+p_true"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(rows, [subset[row]["per_split"]["in_dist"]["auroc"] for row in rows])
    ax.set(ylim=(0.55, 0.77), ylabel="Recorded in-dist AUROC", title="Refitted feature-subset ablation (not deployed replay)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output / "recorded_ablation_deltas.pdf", metadata=pdf_metadata)
    fig.savefig(output / "recorded_ablation_deltas.png", dpi=200)
    plt.close(fig)


def _require_valid(raw: dict[str, Any], crosscheck: dict[str, Any],
                   baseline: dict[str, Any] | None = None) -> None:
    manifest_failures = [
        key for key in ("missing_paths", "unexpected_paths", "malformed_lines", "unsafe_paths", "duplicate_paths", "mismatches")
        if raw[key]
    ]
    if manifest_failures:
        raise AuditValidationError("Raw artifact manifest validation failed: " + ", ".join(manifest_failures))
    if not crosscheck["checks_passed"]:
        raise AuditValidationError("Recorded result crosscheck failed: " + "; ".join(crosscheck["issues"]))
    if baseline is not None and not baseline["valid"]:
        raise AuditValidationError("Raw evidence differs from imported professor-run baseline commit.")


def _require_manifest_valid(manifest: dict[str, Any], label: str) -> None:
    failures = [
        key for key in ("missing_paths", "unexpected_paths", "malformed_lines",
                        "unsafe_paths", "duplicate_paths", "mismatches")
        if manifest[key]
    ]
    if failures:
        raise AuditValidationError(f"{label} validation failed: " + ", ".join(failures))


def verify_committed_evidence() -> dict[str, Any]:
    raw = verify_raw_manifest()
    baseline = verify_git_baseline_anchor()
    certificate = build_split_certificate(ROOT / "data" / "splits")
    crosscheck = build_crosscheck(ROOT / "results", certificate)
    audited = verify_committed_audit_manifest()
    _require_valid(raw, crosscheck, baseline)
    _require_manifest_valid(audited, "Committed audit manifest")
    return {"raw": raw, "baseline": baseline, "audited": audited, "crosscheck": crosscheck}


def generate_audit(output_root: Path) -> dict[str, Any]:
    output_root = validate_external_output_root(output_root)
    results = ROOT / "results"
    evaluation = _read_json(results / "evaluation.json")
    exchange = _read_json(results / "exchangeability_check.json")
    ablations = _read_json(results / "ablations.json")
    certificate = build_split_certificate(ROOT / "data" / "splits")
    raw = verify_raw_manifest()
    baseline = verify_git_baseline_anchor()
    crosscheck = build_crosscheck(results, certificate)
    _require_valid(raw, crosscheck, baseline)  # Validate all recorded inputs before publishing.

    overlap = build_layer_selection_overlap(ROOT / "data" / "splits")
    threshold = build_threshold_invariant(_read_json(results / "conformal_thresholds.json"))
    validity = build_formal_validity_assessment(overlap, threshold)
    intervals = build_operating_intervals(evaluation)
    failure_capture = build_failure_capture(evaluation)
    protocol = build_protocol_dependency_assessment()

    result_output, figure_output = output_root / "results", output_root / "figures"
    result_output.mkdir(parents=True, exist_ok=True)
    artifact_manifest = {
        "audit_type": "secondary_analysis_of_recorded_aggregate_outputs",
        "raw_professor_run_baseline_commit": "a34323d",
        "upstream_professor_revision_reported": "0dd71e1d68f57e6d3125f625d99f9f9c64ac2e9f",
        "executed_config_manifest": str(EXECUTED_CONFIG.relative_to(ROOT)),
        "executed_config_sha256": _sha256(EXECUTED_CONFIG),
        "executed_source_archive": str(EXECUTED_SOURCE_ARCHIVE.relative_to(ROOT)),
        "executed_source_archive_sha256": _sha256(EXECUTED_SOURCE_ARCHIVE),
        "generator_sha256": _sha256(Path(__file__)),
        "python_runtime": platform.python_version(),
        "source_files": sorted(str(path.relative_to(ROOT)) for path in results.glob("*.json")),
        "raw_integrity": raw,
        "git_baseline_anchor": baseline,
        "missing_for_numerical_replay": ["checkpoints/final/", "checkpoints/final_probe.pkl", "checkpoints/routing_lr.pkl", "checkpoints/calibrator.pkl", "data/features/*.npz", "data/features/*_hidden.npy", "data/features/probe_scores_oof.npy"],
    }
    outputs = {
        "artifact_manifest.json": artifact_manifest,
        "split_integrity_certificate.json": certificate,
        "results_crosscheck.json": crosscheck,
        "layer_selection_overlap.json": overlap,
        "threshold_invariant_check.json": threshold,
        "formal_validity_assessment.json": validity,
        "accepted_errors_and_failure_capture.json": failure_capture,
        "protocol_dependency_assessment.json": protocol,
    }
    for filename, content in outputs.items():
        (result_output / filename).write_text(json.dumps(content, indent=2) + "\n")
    _write_markdown(
        result_output, intervals, build_claims(), certificate, validity, threshold,
        failure_capture, protocol,
    )
    _make_figures(figure_output, evaluation, exchange, ablations)
    generated_names = [
        *(f"results/{filename}" for filename in outputs),
        "results/claim_status_table.md",
        "results/operating_point_intervals.md",
        "results/config_execution_reconciliation.md",
        "results/formal_validity_assessment.md",
        "results/threshold_invariant_check.md",
        "results/figure_usage_catalog.md",
        "results/accepted_errors_and_failure_capture.md",
        "results/protocol_dependency_assessment.md",
        "results/raw_artifact_errata.md",
        "figures/local_rate_vs_local_accuracy.pdf",
        "figures/local_rate_vs_local_accuracy.png",
        "figures/threshold_transfer_diagnostic.pdf",
        "figures/threshold_transfer_diagnostic.png",
        "figures/recorded_ablation_deltas.pdf",
        "figures/recorded_ablation_deltas.png",
    ]
    return {
        "raw_integrity": raw,
        "split_certificate": certificate,
        "results_crosscheck": crosscheck,
        "formal_validity_assessment": validity,
        "outputs": sorted(generated_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CPU-only audited outputs from recorded evidence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit regeneration output root; committed audited/ is not rewritten by default.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify committed raw and audited evidence without writing files.",
    )
    args = parser.parse_args()
    if args.verify_only:
        try:
            verification = verify_committed_evidence()
        except AuditValidationError as exc:
            raise SystemExit(f"Evidence verification refused: {exc}") from exc
        print(f"Immutable raw artifacts verified: {verification['raw']['checked']}")
        print(f"Committed audit artifacts verified: {verification['audited']['checked']}")
        print("Recorded cross-file consistency: pass")
        return
    if args.output_dir is None:
        parser.error("choose --verify-only or provide --output-dir for explicit regeneration")
    try:
        summary = generate_audit(args.output_dir)
    except AuditValidationError as exc:
        raise SystemExit(f"Audit refused: {exc}") from exc
    print(f"Audit outputs written to: {args.output_dir}")
    print(f"Immutable artifacts verified: {summary['raw_integrity']['checked']}")
    print("Formal recorded local-rate CP guarantee supported: no")
    print(f"Generated files: {len(summary['outputs'])}")


if __name__ == "__main__":
    main()
