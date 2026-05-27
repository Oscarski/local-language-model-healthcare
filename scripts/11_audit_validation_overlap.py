"""CPU-only MedMCQA validation overlap audit for the recorded thesis run.

The recorded experiment repurposed MedMCQA validation as nominal in-distribution
evaluation after discovering that test labels were hidden. This post-run check
compares validation questions with the committed train/probe indices. It does
not load the model, change raw outputs, or assert semantic contamination from
similarity candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DATASET_ID = "openlifescienceai/medmcqa"
DATASET_REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
PINNED_INPUT_PROVENANCE = {
    "raw_train_fingerprint": "2cf972b7703aa767",
    "validation_fingerprint": "0598613f1e3b097f",
    "normalized_train_ft_sha256": "a08856a558a7053db9e36edc00f0c0684912443d2d021231609b691572acc8b3",
    "normalized_probe_set_sha256": "d3e818c26b97bd852ab57acf807fe82b1acdb410d9c84f5a4e204892cc0a8aaa",
    "normalized_validation_sha256": "ec18fcf00b193eb2c7291e67b2a3b64b3c9bd3c82fe6e5b0a3e72959add88c2b",
}


def validate_external_output_root(output_dir: Path) -> Path:
    """Refuse to overwrite committed thesis evidence or a symlink into it."""
    resolved = output_dir.expanduser().resolve()
    root = ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise RuntimeError(
            "Validation audit output must be outside the repository; "
            "committed audited outputs are immutable evidence."
        )
    return resolved


def normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def _load_token() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _text_sha256(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_recorded_sets() -> tuple[list[str], list[str], list[str], dict[str, str]]:
    import datasets

    train_ft_idx = json.loads((ROOT / "data" / "splits" / "train_ft_idx.json").read_text())
    probe_idx = json.loads((ROOT / "data" / "splits" / "probe_idx.json").read_text())
    raw_train = datasets.load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    validation = datasets.load_dataset(DATASET_ID, revision=DATASET_REVISION, split="validation")
    single = raw_train.filter(lambda row: row["choice_type"] == "single")
    train_ft = [normalize_question(single[index]["question"]) for index in train_ft_idx]
    probe = [normalize_question(single[index]["question"]) for index in probe_idx]
    val = [normalize_question(row["question"]) for row in validation]
    provenance = {
        "raw_train_fingerprint": raw_train._fingerprint,
        # Informational only: transformation fingerprints can depend on local
        # datasets cache state; selected-text hashes below are the invariant.
        "single_train_fingerprint": single._fingerprint,
        "validation_fingerprint": validation._fingerprint,
        "normalized_train_ft_sha256": _text_sha256(train_ft),
        "normalized_probe_set_sha256": _text_sha256(probe),
        "normalized_validation_sha256": _text_sha256(val),
    }
    changed = [
        key for key, expected in PINNED_INPUT_PROVENANCE.items()
        if provenance.get(key) != expected
    ]
    if changed:
        raise RuntimeError(
            "Pinned MedMCQA audit inputs changed; refusing to publish a new result. "
            f"Mismatched fields: {', '.join(changed)}"
        )
    return train_ft, probe, val, provenance


def _exact_matches(reference: list[str], validation: list[str]) -> list[dict[str, Any]]:
    positions: dict[str, list[int]] = {}
    for index, question in enumerate(reference):
        positions.setdefault(question, []).append(index)
    matches: list[dict[str, Any]] = []
    for val_index, question in enumerate(validation):
        if question in positions:
            matches.append({"validation_index": val_index, "reference_indices": positions[question]})
    return matches


def _near_match_candidates(reference: list[str], validation: list[str], limit: int = 25) -> list[dict[str, Any]]:
    """Return review candidates only; similarity is not a duplicate verdict."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
    ref_matrix = vectorizer.fit_transform(reference)
    val_matrix = vectorizer.transform(validation)
    nearest = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nearest.fit(ref_matrix)
    distances, indices = nearest.kneighbors(val_matrix)
    ranked = sorted(
        (
            (float(1.0 - distances[row][0]), row, int(indices[row][0]))
            for row in range(len(validation))
        ),
        reverse=True,
    )
    return [
        {
            "cosine_similarity": round(score, 5),
            "validation_index": val_index,
            "reference_index": ref_index,
            "requires_manual_review": True,
        }
        for score, val_index, ref_index in ranked[:limit]
        if score < 1.0
    ]


def run_audit(include_near_candidates: bool = True) -> dict[str, Any]:
    _load_token()
    train_ft, probe, validation, provenance = _load_recorded_sets()
    output: dict[str, Any] = {
        "audit_type": "cpu_only_public_dataset_post_run_check",
        "input_provenance": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            **provenance,
        },
        "interpretation": (
            "Exact normalized matches identify direct overlap. TF-IDF nearest-neighbour "
            "items are review candidates only and are not automatically contaminants."
        ),
        "sizes": {"train_ft": len(train_ft), "probe_set": len(probe), "validation": len(validation)},
        "exact_normalized_overlap": {
            "train_ft_vs_validation": _exact_matches(train_ft, validation),
            "probe_set_vs_validation": _exact_matches(probe, validation),
        },
    }
    output["exact_normalized_overlap_counts"] = {
        key: len(value) for key, value in output["exact_normalized_overlap"].items()
    }
    if include_near_candidates:
        output["near_match_candidates_not_verdicts"] = {
            "train_ft_vs_validation": _near_match_candidates(train_ft, validation),
            "probe_set_vs_validation": _near_match_candidates(probe, validation),
        }
    return output


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir = validate_external_output_root(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_overlap_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    counts = result["exact_normalized_overlap_counts"]
    lines = [
        "# MedMCQA Validation Overlap Audit",
        "",
        "CPU-only post-run check using public dataset text and committed split indices.",
        "It is not model evaluation and does not alter recorded experiment outputs.",
        f"Pinned source: `{result['input_provenance']['dataset_id']}` at revision "
        f"`{result['input_provenance']['dataset_revision']}`.",
        "",
        "| Comparison | Exact normalized matches |",
        "|---|---:|",
        f"| train_ft vs validation | {counts['train_ft_vs_validation']} |",
        f"| probe_set vs validation | {counts['probe_set_vs_validation']} |",
        "",
        "TF-IDF nearest-neighbour entries in the JSON, when present, are candidates",
        "for manual review only; they are not evidence of contamination by themselves.",
    ]
    (output_dir / "validation_overlap_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MedMCQA validation overlap without model reruns.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "diploma-thesis-overlap-audit",
        help="Output directory; defaults outside committed audited evidence.",
    )
    parser.add_argument("--exact-only", action="store_true", help="Skip TF-IDF review candidates.")
    args = parser.parse_args()
    args.output_dir = validate_external_output_root(args.output_dir)
    result = run_audit(include_near_candidates=not args.exact_only)
    write_report(result, args.output_dir)
    print(f"Validation overlap audit written to: {args.output_dir}")
    print(json.dumps(result["exact_normalized_overlap_counts"], indent=2))


if __name__ == "__main__":
    main()
