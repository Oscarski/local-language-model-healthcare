.PHONY: setup verify prepare finetune extract probe routing conformal evaluate ablations audit audit-overlap verify-evidence pipeline test lint clean

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev]"

# ── Pipeline steps ───────────────────────────────────────────────────────────
verify:
	python scripts/00_verify.py --blocker 1
	python scripts/00_verify.py --blocker 2

verify-all:
	python scripts/00_verify.py --blocker all

prepare:
	python scripts/01_prepare.py

finetune:
	python scripts/02_finetune.py

finetune-qlora:
	python scripts/02_finetune.py model=qlora_4bit

extract:
	python scripts/03_extract.py

probe:
	python scripts/04_probe.py

routing:
	python scripts/05_routing.py

conformal:
	python scripts/06_conformal.py

evaluate:
	python scripts/07_evaluate.py

ablations:
	python scripts/08_ablations.py

# ── Post-run audit (CPU only; reads recorded outputs, writes audited/) ─────────
audit:
	python scripts/10_audit_recorded_results.py

audit-overlap:
	python scripts/11_audit_validation_overlap.py

verify-evidence: audit
	shasum -a 256 -c artifacts/professor_run/raw_artifacts.sha256

# ── Historical pipeline guard ────────────────────────────────────────────────
pipeline:
	@printf '%s\n' 'Raw professor-run outputs are immutable. Run `make audit` instead.'
	@exit 1

# ── Dev ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/
	ruff format --check src/ scripts/

format:
	ruff format src/ scripts/

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ wandb/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

clean-features:
	rm -rf data/features/*.npz data/features/*.npy

# ── Status ───────────────────────────────────────────────────────────────────
status:
	python scripts/00_verify.py --blocker status
