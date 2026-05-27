# Provenance and Artifact Boundary

## Recorded Professor Run

This repository contains the recorded outputs of the completed GPU experiment
performed by the supervisor on 25-26 May 2026.

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/leontikos/local-language-model-healthcare` |
| Upstream revision reported by imported baseline | `0dd71e1d68f57e6d3125f625d99f9f9c64ac2e9f` |
| Local baseline commit | `a34323d` (`Import final professor run snapshot (0dd71e1)`) |
| Hardware recorded in run report | 4 x NVIDIA L4 |
| Successful fine-tune configuration | bf16, SDPA, LoRA, `learning_rate=5e-5` |

## Immutable Historical Artifacts

The following files are treated as raw outputs of the recorded run. They must
not be overwritten by later analysis:

- `results/*.json`
- `figures/*`
- `logs/*`
- `data/splits/meta.json`
- `data/splits/train_ft_idx.json`
- `data/splits/probe_idx.json`
- `data/splits/routing_train_idx.json`
- `data/splits/iso_cal_idx.json`
- `data/splits/routing_train_local_idx.json`
- `data/splits/iso_cal_local_idx.json`
- `data/splits/conformal_cal_local_idx.json`
- `artifacts/professor_run/REPORT_ORIGINAL.md`

`artifacts/professor_run/raw_artifacts.sha256` records SHA-256 digests for the
raw `results/`, `figures/`, `logs/`, evidence-bearing split files, and the
archived original report.

The logs are intentionally retained as execution provenance. They disclose
absolute paths from the supervisor's machine, but the audit found no committed
API tokens or secrets.

The upstream Git object identified above is not present in this local clone.
The local cryptographic anchor is commit `a34323d`, whose message reports the
upstream revision; this repository does not claim an independent local
cryptographic verification of the upstream object.

## Available and Missing Evidence

Available in this repository:

- source code used by the recorded run;
- final JSON result summaries;
- rendered figures;
- execution logs;
- committed split-index files.

Not available in this repository:

- `checkpoints/final/`;
- `checkpoints/final_probe.pkl`, `routing_lr.pkl`, or `calibrator.pkl`;
- `data/features/*.npz`, `*_hidden.npy`, or `probe_scores_oof.npy`.

Those large artifacts were excluded from Git. Consequently, the repository
preserves the recorded experiment but does not support independent numerical
replay of model inference, routing refitting, or corrected conformal analyses
without an external artifact transfer or a new GPU run.

## Audited Layer

Documentation in the repository root presents the final audited
interpretation of the recorded experiment. New CPU-only derived outputs are
written under `audited/`; they summarize existing committed evidence and are
not new model experiments.

In particular:

- the historical conformal output concerns the local-answer selection rate,
  not a formal guarantee of clinical correctness;
- formal validity of the recorded local-rate conformal procedure is not
  supported, because supervised layer selection overlapped the later
  `conformal_cal` subset and the raw threshold output fails a same-sample
  inclusion invariant;
- the recorded `p_true` inclusion check is preserved as historical behavior,
  while its tautological mass test is documented as a limitation;
- causal interpretations of OOD behavior are treated as hypotheses only.
- `audited/results/validation_overlap_audit.*` records a bounded CPU-only
  check prompted by repurposing MedMCQA validation for evaluation; it found
  one normalized exact `train_ft` match and no `probe_set` match.
