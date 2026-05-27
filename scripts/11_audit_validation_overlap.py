"""CPU-only MedMCQA validation overlap audit for the recorded thesis run.

The recorded experiment repurposed MedMCQA validation as nominal in-distribution
evaluation after discovering that test labels were hidden. This post-run check
compares validation questions with the committed train/probe indices. It does
not load the model, change raw outputs, or assert semantic contamination from
similarity candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent


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


def _load_recorded_sets() -> tuple[list[str], list[str], list[str]]:
    import datasets

    train_ft_idx = json.loads((ROOT / "data" / "splits" / "train_ft_idx.json").read_text())
    probe_idx = json.loads((ROOT / "data" / "splits" / "probe_idx.json").read_text())
    raw_train = datasets.load_dataset("openlifescienceai/medmcqa", split="train")
    validation = datasets.load_dataset("openlifescienceai/medmcqa", split="validation")
    single = raw_train.filter(lambda row: row["choice_type"] == "single")
    train_ft = [normalize_question(single[index]["question"]) for index in train_ft_idx]
    probe = [normalize_question(single[index]["question"]) for index in probe_idx]
    val = [normalize_question(row["question"]) for row in validation]
    return train_ft, probe, val


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
    train_ft, probe, validation = _load_recorded_sets()
    output: dict[str, Any] = {
        "audit_type": "cpu_only_public_dataset_post_run_check",
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_overlap_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    counts = result["exact_normalized_overlap_counts"]
    lines = [
        "# MedMCQA Validation Overlap Audit",
        "",
        "CPU-only post-run check using public dataset text and committed split indices.",
        "It is not model evaluation and does not alter recorded experiment outputs.",
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "audited" / "results")
    parser.add_argument("--exact-only", action="store_true", help="Skip TF-IDF review candidates.")
    args = parser.parse_args()
    result = run_audit(include_near_candidates=not args.exact_only)
    write_report(result, args.output_dir)
    print(f"Validation overlap audit written to: {args.output_dir}")
    print(json.dumps(result["exact_normalized_overlap_counts"], indent=2))


if __name__ == "__main__":
    main()
