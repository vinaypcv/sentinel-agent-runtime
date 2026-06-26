# Sentinel Agent Runtime

**A neurosymbolic runtime for safe autonomous AI agents with memory, epistemic guardrails, auditability, rollback, and benchmarked software repair.**

Sentinel Agent Runtime is an experimental AI safety and agent reliability framework designed to make autonomous agents more **verifiable, observable, and recoverable**. It combines neural similarity, symbolic policy checks, persistent memory, audit logging, and autonomous repair loops into a single runtime architecture.

The project demonstrates how an agent can reason over actions, evaluate risk before execution, record decisions, and recover from failed code changes using a deterministic repair benchmark and optional Gemini-powered patch generation.

---

## Why This Project Matters

Modern LLM agents are powerful, but they still struggle with:

* long-horizon task reliability,
* hallucinated decisions,
* unsafe tool use,
* weak memory consistency,
* poor auditability,
* brittle software repair,
* lack of rollback when actions fail.

Sentinel Agent Runtime explores a production-oriented answer:

> Do not trust the agent directly. Wrap it inside a runtime that verifies, scores, logs, tests, and rolls back every high-impact action.

The system is inspired by neurosymbolic AI, cognitive architectures, software verification, and agent safety research.

---

## Core Idea

Instead of treating an AI agent as only:

```text
prompt + context + model -> action
```

Sentinel treats the agent as a controlled runtime loop:

```text
goal + memory + policy + verifier + tests + audit log -> safe action
```

The runtime evaluates candidate actions through multiple layers:

```text
Input / Goal
   ↓
SAMS Memory + AkashaStore
   ↓
PEG Policy & Epistemic Guardrail
   ↓
Viveka Safety Score
   ↓
Dharma Loop Repair / Execution
   ↓
KarmaLedger Audit Log
   ↓
Rollback or Commit
```

---

## Architecture Overview

```text
sentinel-agent-runtime/
│
├── src/brahman_os/
│   ├── memory/
│   │   ├── sams.py                 # Symbolic Associative Memory Substrate
│   │   └── akasha_store.py          # Persistent memory store
│   │
│   ├── guardrails/
│   │   ├── peg.py                   # Pramana Epistemic Guardrail
│   │   ├── viveka.py                # Composite safety scoring
│   │   └── policy_loader.py         # YAML safety policy loading
│   │
│   ├── ledger/
│   │   └── karma_ledger.py          # Decision/event audit log
│   │
│   ├── repair/
│   │   ├── llm_patch_generator.py   # Mock/Gemini patch generation
│   │   ├── patcher.py               # Unified diff patch application
│   │   ├── rollback.py              # Recovery after failed patch
│   │   └── test_runner.py           # Static/runtime validation
│   │
│   ├── runtime/
│   │   └── orchestrator.py          # End-to-end runtime controller
│   │
│   └── schemas.py                   # Shared typed data contracts
│
├── benchmarks/
│   ├── benchmark_guardrails.py      # Safety/guardrail benchmark
│   ├── benchmark_repair.py          # Repair benchmark runner
│   └── repair_cases/                # 10 controlled software bug cases
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── examples/
│   ├── medical_policy_guardrail/
│   └── software_repair/
│
├── policies/
│   └── medical_safety.yaml
│
├── tests/
│   ├── integration/
│   └── test_*.py
│
└── .github/workflows/ci.yml
```

---

## System Components

### 1. SAMS — Symbolic Associative Memory Substrate

SAMS models memory as a high-dimensional associative structure. It supports storing and retrieving structured knowledge using vector-symbolic operations.

Conceptually, an item is represented as:

```math
m_i = k_i \otimes v_i
```

where:

* `k_i` is a key representation,
* `v_i` is a value representation,
* `⊗` is an outer-product style binding operation.

A memory state is accumulated as:

```math
M = \sum_{i=1}^{n} k_i \otimes v_i
```

Retrieval is approximated by querying the memory with a key:

```math
\hat{v} = M^\top k_q
```

This gives the runtime a lightweight cognitive memory layer that can support persistent context beyond a single prompt.

---

### 2. AkashaStore — Persistent Agent Memory

AkashaStore provides the persistence layer around SAMS. It allows the runtime to store reusable facts, decisions, traces, or previous execution outcomes.

This separates the agent from stateless prompt-only behavior.

```text
Short-term context -> SAMS encoding -> AkashaStore persistence -> future retrieval
```

---

### 3. PEG — Pramana Epistemic Guardrail

PEG evaluates whether an action or output is acceptable under a policy.

It combines:

* symbolic rules,
* semantic similarity checks,
* blocked intent detection,
* domain-specific YAML policies.

Example policy flow:

```text
Candidate action
   ↓
Policy match
   ↓
Semantic risk scan
   ↓
Allow / block / require review
```

PEG is designed to answer:

> Is this action justified, safe, and policy-consistent?

---

### 4. Viveka Score — Composite Safety Scoring

Viveka produces a composite decision score for candidate actions.

A simplified scoring function:

```math
V(x) = \alpha S_{semantic}(x) + \beta S_{symbolic}(x) + \gamma S_{test}(x) - \lambda R(x)
```

where:

* `S_semantic(x)` measures semantic alignment,
* `S_symbolic(x)` measures policy compliance,
* `S_test(x)` measures validation success,
* `R(x)` measures risk,
* `α, β, γ, λ` are weighting parameters.

A decision is accepted only if:

```math
V(x) \geq \tau
```

where `τ` is the configured safety threshold.

This prevents the runtime from accepting an agent action only because it “sounds correct.”

---

### 5. KarmaLedger — Auditability and Trace Logging

KarmaLedger records decisions, validation results, repair attempts, failures, and rollback events.

Each meaningful action becomes an auditable event:

```json
{
  "event_type": "repair_attempt",
  "case_id": "missing_none_check",
  "provider": "mock",
  "peg_passed": true,
  "tests_passed": true,
  "rollback_triggered": false
}
```

This creates an explainability trail for autonomous systems.

The core idea:

```text
No silent actions. Every decision leaves a trace.
```

---

### 6. Dharma Loop — Autonomous Software Repair

Dharma Loop is the repair subsystem. It attempts to fix failing Python code while preserving safety and recoverability.

Repair loop:

```text
Buggy code
   ↓
Failing tests
   ↓
Patch generation
   ↓
Diff validation
   ↓
PEG/Viveka approval
   ↓
Static checks
   ↓
Pytest validation
   ↓
Commit patch or rollback
```

If validation fails, rollback is triggered.

```text
Unsafe patch -> reject -> restore previous state
```

This turns software repair into a controlled agentic workflow rather than an uncontrolled LLM edit.

---

## Repair Benchmark

The repair benchmark evaluates the runtime against 10 controlled Python bug classes.

### Repair Cases

| Case                           | Bug Type                      |
| ------------------------------ | ----------------------------- |
| `bounded_stack_capacity_bug`   | Capacity boundary error       |
| `off_by_one_bug`               | Index/range logic error       |
| `missing_none_check`           | Null safety bug               |
| `wrong_sort_order`             | Incorrect ordering            |
| `wrong_exception_type`         | Incorrect exception semantics |
| `mutation_while_iterating`     | Unsafe collection mutation    |
| `unsafe_eval_usage`            | Security vulnerability        |
| `path_traversal_vulnerability` | Filesystem security flaw      |
| `missing_input_validation`     | Validation gap                |
| `incorrect_aggregation_logic`  | Incorrect computation logic   |

Each case contains:

```text
solution.py
test_solution.py
expected_behavior.md
metadata.json
```

---

## Benchmark Metrics

Each repair run records:

| Metric                 | Meaning                                      |
| ---------------------- | -------------------------------------------- |
| `case_id`              | Repair case identifier                       |
| `provider`             | `mock` or `gemini`                           |
| `patch_generated`      | Whether a patch was produced                 |
| `valid_diff`           | Whether the patch was a valid unified diff   |
| `peg_passed`           | Whether policy guardrails approved the patch |
| `static_checks_passed` | Whether static checks passed                 |
| `tests_passed`         | Whether repaired code passed tests           |
| `rollback_triggered`   | Whether rollback was needed                  |
| `latency_seconds`      | Runtime duration                             |
| `error_message`        | Failure reason, if any                       |
| `decision_json`        | Structured runtime decision                  |

---

## Current Validation Status

Local validation:

```text
Repair benchmark tests: 5 passed
Mock benchmark: succeeded
Ruff: passed
Full test suite: 117 passed, 1 skipped
GitHub Actions CI: passing
Secret scan: clean
```

The mock benchmark validates deterministic repair plumbing. Gemini mode is gated behind explicit live environment variables and is used to evaluate real LLM patch quality.

---

## Demo Commands

### Install

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Run Tests

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

Expected:

```text
117 passed, 1 skipped
```

---

### Run Ruff

```powershell
.\.venv\Scripts\ruff.exe check .
```

---

### Run Guardrail Benchmark

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_guardrails.py
```

---

### Run Mock Repair Benchmark

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_repair.py --provider mock --max-cases 3
```

Run all mock cases:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_repair.py --provider mock
```

---

### Run Gemini Repair Benchmark

Gemini mode is disabled by default to prevent accidental API calls.

```powershell
$env:GEMINI_API_KEY="your_key_here"
$env:GEMINI_MODEL="gemini-2.5-flash"
$env:RUN_LIVE_API_TESTS="1"
```

Then:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_repair.py --provider gemini --max-cases 3
```

---

### Run Streamlit Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

---

## Example Use Cases

### AI Safety Runtime

Use Sentinel as a guardrail layer around autonomous agents that need policy checks before executing high-impact actions.

### Autonomous Software Repair

Use Dharma Loop to test LLM-generated code patches against static checks, policy verification, and runtime tests before accepting changes.

### Agent Auditability

Use KarmaLedger to record why an agent took an action, what checks passed, and whether rollback was required.

### Research Prototype

Use SAMS, PEG, Viveka, and KarmaLedger as a research scaffold for neurosymbolic agent reliability.

---

## Design Principles

### 1. Verify Before Acting

The runtime should not execute high-risk actions without policy and validation checks.

### 2. Make Decisions Observable

Every action should leave an audit trail.

### 3. Prefer Rollback Over Silent Failure

Failed or unsafe repairs should restore the previous state.

### 4. Separate Mock Reliability from LLM Reliability

Mock mode validates deterministic runtime plumbing. Gemini mode evaluates real model behavior separately.

### 5. Treat Safety as a Runtime Property

Safety is not only a prompt. It is a system-level control loop.

---

## CI/CD

The project includes GitHub Actions CI for deterministic validation.

CI runs:

```text
ruff check .
pytest
```

Live Gemini tests are disabled in CI by default:

```text
RUN_LIVE_API_TESTS=0
BRAHMAN_PATCH_PROVIDER=mock
```

This ensures CI is reproducible and does not require secret API keys.

---

## Security Notes

* `.env` files are ignored.
* Raw prototype notebooks are excluded from Git.
* Generated benchmark outputs are ignored.
* Live Gemini calls require explicit environment variables.
* Secret scanning was performed before publishing.

---

## Project Positioning

Sentinel Agent Runtime is not just a chatbot wrapper. It is a runtime architecture for safer agent execution.

It demonstrates:

* neurosymbolic memory,
* policy-based verification,
* composite risk scoring,
* structured audit logging,
* rollback-based software repair,
* benchmark-driven evaluation,
* CI-backed engineering quality.

---

## Roadmap

Planned improvements:

* richer benchmark leaderboard,
* live Gemini repair quality reports,
* dashboard visualizations for benchmark results,
* policy editor UI,
* expanded memory retrieval experiments,
* GitHub issue repair integration,
* Dockerized local runtime,
* OpenTelemetry tracing,
* multi-agent repair evaluation.

---

## Repository

```text
https://github.com/vinaypcv/sentinel-agent-runtime
```

---

## One-Line Summary

**Sentinel Agent Runtime is a neurosymbolic AI agent runtime that combines memory, guardrails, audit logs, rollback, and software repair benchmarks to make autonomous agents safer and more reliable.**
