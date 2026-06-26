"""Tests for the Akasha tensor memory store."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from brahman_os.memory.akasha_store import AkashaStore


def test_write_changes_memory_matrix_and_records_event() -> None:
    """A write should update the matrix and preserve explainable metadata."""
    store = AkashaStore(d_model=4, lambda_decay=0.9)
    initial = store.M_t.clone()

    event = store.write(
        torch.tensor([1.0, 2.0, 0.0, -1.0]),
        {
            "memory_id": uuid4(),
            "confidence": 0.8,
            "provenance": ("unit-test",),
            "evidence": ("observation-1",),
        },
    )

    assert not torch.equal(store.M_t, initial)
    assert store.events == [event]
    assert event.confidence == 0.8


def test_read_returns_query_dimension_and_updates_access_count() -> None:
    """Reads should return a decayed vector and increment access metadata."""
    store = AkashaStore(d_model=3)
    store.write(torch.tensor([1.0, 0.5, -0.5]), {"confidence": 0.9})

    first = store.read(torch.ones(3))
    second = store.read(torch.ones(3), delta_time=2.0)

    assert first.embedding is not None
    assert len(first.embedding) == 3
    assert first.access_count == 1
    assert second.access_count == 2
    assert second.decay_factor < first.decay_factor
    assert store.access_count == 2
    json.dumps(store.explain_last_read())


def test_snapshot_and_rollback_restore_tensor_and_metadata() -> None:
    """Rollback should restore matrix, events, access count, and last read."""
    store = AkashaStore(d_model=2)
    store.write(torch.tensor([1.0, 0.0]), {"evidence": ("first",)})
    store.read(torch.ones(2))
    snapshot_id = store.snapshot()
    expected_matrix = store.M_t.clone()
    expected_events = tuple(store.events)

    store.write(torch.tensor([0.0, 2.0]), {"evidence": ("second",)})
    store.read(torch.ones(2))
    store.rollback(snapshot_id)

    assert torch.equal(store.M_t, expected_matrix)
    assert tuple(store.events) == expected_events
    assert store.access_count == 1
    assert store.explain_last_read()["access_count"] == 1


def test_save_and_load_preserve_state(tmp_path: Path) -> None:
    """Persistence should preserve matrix and audited access state."""
    path = tmp_path / "akasha.pt"
    store = AkashaStore(d_model=3, lambda_decay=0.8)
    store.write(torch.tensor([0.5, 1.0, -0.25]), {"provenance": ("test",)})
    store.read(torch.ones(3))
    store.save(path)

    loaded = AkashaStore.load(path)

    assert torch.equal(loaded.M_t, store.M_t)
    assert loaded.events == store.events
    assert loaded.access_count == store.access_count
    assert loaded.store_id == store.store_id


def test_invalid_vectors_and_snapshot_ids_fail_explicitly() -> None:
    """Invalid input dimensions and unknown rollback targets should fail."""
    store = AkashaStore(d_model=3)

    with pytest.raises(ValueError, match="shape"):
        store.write(torch.ones(2), {})
    with pytest.raises(ValueError, match="finite"):
        store.read(torch.tensor([1.0, float("nan"), 0.0]))
    with pytest.raises(KeyError, match="unknown snapshot_id"):
        store.rollback(uuid4())
