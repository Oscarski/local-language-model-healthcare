# Provenance and Artifact Boundary

## Recorded Run

This repository contains the recorded outputs of the completed GPU experiment
performed by the supervisor on 25-26 May 2026.

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/leontikos/local-language-model-healthcare` |
| Upstream revision reported by imported baseline | `0dd71e1d68f57e6d3125f625d99f9f9c64ac2e9f` |
| Local baseline commit | `a34323d` (imported final run snapshot, upstream `0dd71e1`) |
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
- `scripts/config.json` (recorded blocker outputs; its `p_true` verdict is historical, not validated evidence)
- `artifacts/run/REPORT_ORIGINAL.md`

`artifacts/run/raw_artifacts.sha256` records SHA-256 digests for the
raw `results/`, `figures/`, `logs/`, evidence-bearing split/config files, and
the archived original report.

The exact executable source/configuration tree from local import baseline
commit `a34323d` is archived at
`artifacts/run/executed_source.tar.gz`; its checksum and scope are
documented in `artifacts/run/EXECUTED_SOURCE.md`. The current
top-level scripts are an audited derivative containing guards and explanatory
annotations and must not be confused with the historical executable snapshot.

The logs are intentionally retained as execution provenance. They disclose
absolute paths from the supervisor's machine, but the audit found no committed
API tokens or secrets.

The local cryptographic anchor is commit `a34323d`. Read-only verification
checks that raw evidence remains equal to that imported baseline and that the
archived original report is byte-identical to its baseline copy. The public
upstream revision is recorded for provenance; the local baseline remains the
submission's enforceable integrity anchor.

## Available and Missing Evidence

Available in this repository:

- the archived source code used by the recorded run
  (`artifacts/run/executed_source.tar.gz`); the current top-level
  scripts are an audited, execution-guarded derivative;
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
interpretation of the recorded experiment. Committed CPU-only derived outputs
under `audited/` summarize existing committed evidence and are not new model
experiments. They have a separate checksum manifest at
`artifacts/audited_layer/committed_outputs.sha256`; normal verification is
read-only, while regeneration must target an explicit separate output
directory.

In particular:

- the historical conformal output concerns the local-answer selection rate,
  not a formal guarantee of clinical correctness;
- formal validity of the recorded local-rate conformal procedure is not
  supported, because OOF probe construction preceded the outcome-stratified
  downstream partition, supervised layer selection overlapped the later
  `conformal_cal` subset, score construction changes at evaluation, and the
  raw threshold output fails a same-sample inclusion invariant;
- the recorded `p_true` inclusion check is preserved as historical behavior,
  while its tautological mass test is documented as a limitation;
- causal interpretations of OOD behavior are treated as hypotheses only.
- `near-OOD` and `far-OOD` are preserved historical output labels; final
  prose uses external MedQA and external MMLU medical evaluation.
- `audited/results/validation_overlap_audit.*` records a bounded CPU-only
  check prompted by repurposing MedMCQA validation for evaluation; it found
  one normalized exact `train_ft` match and no `probe_set` match. This audit
  is pinned to MedMCQA revision
  `91c6572c454088bf71b679ad90aa8dffcd0d5868` and records dataset
  fingerprints and normalized-input hashes.

## Read-Only Verification

```bash
make verify-evidence
pytest -q
```

The historical blocker procedures and GPU/downstream pipeline are retained as
source provenance but are blocked from overwriting evidence in this finalized
repository. Explicit audit regeneration writes to a separate output root and
is not needed for thesis writing.
