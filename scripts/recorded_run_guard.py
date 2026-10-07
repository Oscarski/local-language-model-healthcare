"""Write guard for immutable outputs of the completed run."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, NoReturn


def refuse_historical_execution(step: str) -> NoReturn:
    """Disable reruns of the executed pipeline in the thesis attachment checkout."""
    raise SystemExit(
        f"{step} is preserved as historical executed source, not an active rerun entrypoint.\n"
        "This checkout contains immutable evidence from the completed run. "
        "The exact executable snapshot is archived under artifacts/run/ "
        "and anchored at git commit a34323d.\n"
        "Use `make verify-evidence` for read-only checks or a separate checkout "
        "for any future reproduction."
    )


def protect_recorded_outputs(paths: Iterable[Path], step: str) -> None:
    """Refuse overwriting raw recorded evidence.

    Reproduction must write to a separate checkout or output root. Recorded
    run artifacts in this repository are never overwrite targets.
    """
    protected = sorted(str(path) for path in paths)
    if protected:
        listed = "\n  ".join(protected)
        raise SystemExit(
            f"{step} would write within immutable recorded-run artifact paths:\n  {listed}\n"
            "This repository is finalized for thesis writing. Run "
            "scripts/10_audit_recorded_results.py --verify-only for read-only "
            "verification. Any separate reproduction must use another output root."
        )
