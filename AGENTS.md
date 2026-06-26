# Brahman-OS Codex Instructions

You are implementing Brahman-OS, a production-grade neurosymbolic runtime for safe autonomous AI agents.

Core architecture:
- SAMS / AkashaDB: tensor-based persistent memory substrate.
- PEG: Pramana Epistemic Guardrail using vector similarity and symbolic rule validation.
- Viveka Score: composite pass/block decision score.
- KarmaLedger: audit log for every agent action, verification result, patch, rollback, and test run.
- Dharma Loop: autonomous software repair loop using tests, patch generation, verification, and rollback.

Engineering rules:
- Do not create notebook-only code.
- Implement importable Python modules under src/brahman_os.
- Do not hardcode API keys.
- Use Pydantic models for schemas.
- Add unit tests for every module.
- Add docstrings and type hints.
- Prefer small, reviewable changes.
- Do not implement fake success paths.
- Every example must be runnable from the command line.
- Every safety decision must produce explainable JSON.
- Run pytest and ruff after each milestone.