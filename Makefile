.PHONY: setup verify prepare finetune extract probe routing conformal evaluate ablations audit audit-overlap verify-evidence pipeline test lint clean

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev]"

# ── Recorded evidence verification (read-only) ──────────────────────────────
verify: verify-evidence
	python scripts/00_verify.py --blocker status

verify-all:
	@printf '%s\n' 'Historical blocker reruns are disabled: scripts/config.json is immutable evidence.'
	@exit 1

prepare:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

finetune:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

finetune-qlora:
	@printf '%s\n' 'QLoRA was not part of the recorded run; no rerun is enabled here.'
	@exit 1

extract:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

probe:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

routing:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

conformal:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

evaluate:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

ablations:
	@printf '%s\n' 'Historical pipeline entrypoint disabled in this finalized thesis checkout.'
	@exit 1

# ── Post-run audit (CPU only; regeneration writes outside tracked evidence) ───
audit:
	python scripts/10_audit_recorded_results.py --output-dir /tmp/diploma-thesis-audit-regeneration

audit-overlap:
	python scripts/11_audit_validation_overlap.py --output-dir /tmp/diploma-thesis-overlap-audit

verify-evidence:
	python scripts/10_audit_recorded_results.py --verify-only

# ── Historical pipeline guard ────────────────────────────────────────────────
pipeline:
	@printf '%s\n' 'Raw professor-run outputs are immutable. Run `make verify-evidence` instead.'
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
	@printf '%s\n' 'Feature cleanup disabled: this is a finalized recorded-run checkout.'
	@exit 1

# ── Status ───────────────────────────────────────────────────────────────────
status:
	python scripts/00_verify.py --blocker status
