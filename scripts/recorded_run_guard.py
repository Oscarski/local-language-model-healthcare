"""Write guard for immutable outputs of the completed professor run."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

OVERRIDE_ENV = "ALLOW_RECORDED_RUN_OVERWRITE"


def protect_recorded_outputs(paths: Iterable[Path], step: str) -> None:
    """Refuse overwriting raw recorded evidence unless explicitly overridden."""
    existing = sorted(str(path) for path in paths if path.exists())
    if existing and os.environ.get(OVERRIDE_ENV) != "1":
        listed = "\n  ".join(existing)
        raise SystemExit(
            f"{step} would overwrite immutable recorded-run artifacts:\n  {listed}\n"
            "This repository is finalized for thesis writing. Run "
            "scripts/10_audit_recorded_results.py for CPU-only analysis. "
            f"Set {OVERRIDE_ENV}=1 only for a deliberate separate archival reproduction."
        )
