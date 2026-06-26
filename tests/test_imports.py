"""Smoke tests for the public Brahman-OS module structure."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "brahman_os",
        "brahman_os.schemas",
        "brahman_os.memory",
        "brahman_os.memory.sams",
        "brahman_os.memory.akasha_store",
        "brahman_os.guardrails",
        "brahman_os.guardrails.peg",
        "brahman_os.guardrails.policy_loader",
        "brahman_os.guardrails.viveka",
        "brahman_os.ledger",
        "brahman_os.ledger.karma_ledger",
        "brahman_os.repair",
        "brahman_os.repair.test_runner",
        "brahman_os.repair.patcher",
        "brahman_os.repair.rollback",
        "brahman_os.runtime",
        "brahman_os.runtime.orchestrator",
    ],
)
def test_module_imports(module_name: str) -> None:
    """Every package module should import without side effects."""
    assert import_module(module_name) is not None
