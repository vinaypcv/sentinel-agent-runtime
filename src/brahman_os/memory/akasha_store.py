"""In-memory tensor persistence for the Akasha memory substrate."""

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import torch
from torch import Tensor

from brahman_os.schemas import MemoryEvent, MemoryEventType, MemoryReadResult


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Internal immutable copy of AkashaStore state."""

    matrix: Tensor
    events: tuple[MemoryEvent, ...]
    access_count: int
    last_read: MemoryReadResult | None


class AkashaStore:
    """Maintain a decaying associative memory matrix and its audit metadata."""

    def __init__(
        self,
        d_model: int,
        *,
        lambda_decay: float = 0.99,
        temporal_decay_rate: float = 0.01,
        access_decay_rate: float = 0.01,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize an empty ``[d_model, d_model]`` associative memory."""
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 <= lambda_decay <= 1.0:
            raise ValueError("lambda_decay must be between 0.0 and 1.0")
        if temporal_decay_rate < 0.0 or access_decay_rate < 0.0:
            raise ValueError("decay rates must be non-negative")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")

        self.d_model = d_model
        self.lambda_decay = float(lambda_decay)
        self.temporal_decay_rate = float(temporal_decay_rate)
        self.access_decay_rate = float(access_decay_rate)
        self.M_t = torch.zeros((d_model, d_model), dtype=dtype, device=device)
        self.events: list[MemoryEvent] = []
        self.access_count = 0
        self.store_id = uuid4()
        self._snapshots: dict[UUID, _Snapshot] = {}
        self._last_read: MemoryReadResult | None = None

    def write(
        self,
        psi: Tensor,
        metadata: MemoryEvent | Mapping[str, object],
    ) -> MemoryEvent:
        """Apply a decayed outer-product update and record its metadata."""
        vector = self._validate_vector(psi, name="psi")
        event = (
            metadata
            if isinstance(metadata, MemoryEvent)
            else self._event_from_metadata(metadata)
        )

        self.M_t = self.lambda_decay * self.M_t + torch.outer(vector, vector)
        self.events.append(event)
        return event

    def read(self, query: Tensor, delta_time: float = 1.0) -> MemoryReadResult:
        """Retrieve a decayed associative vector and update the access count."""
        vector = self._validate_vector(query, name="query")
        if delta_time < 0.0 or not math.isfinite(delta_time):
            raise ValueError("delta_time must be finite and non-negative")

        temporal_decay = math.exp(-self.temporal_decay_rate * delta_time)
        access_decay = 1.0 / (1.0 + self.access_decay_rate * self.access_count)
        decay_factor = temporal_decay * access_decay
        retrieved = torch.matmul(vector, self.M_t) * decay_factor
        self.access_count += 1

        found = bool(self.events)
        latest_event = self.events[-1] if found else None
        confidence = self._retrieval_confidence(retrieved) if found else 0.0
        result = MemoryReadResult(
            memory_id=self.store_id,
            found=found,
            content="Retrieved distributed memory vector." if found else None,
            embedding=tuple(float(value) for value in retrieved.detach().cpu().tolist())
            if found
            else None,
            confidence=confidence,
            provenance=latest_event.provenance if latest_event is not None else (),
            evidence=latest_event.evidence if latest_event is not None else (),
            access_count=self.access_count,
            decay_factor=decay_factor,
        )
        self._last_read = result
        return result

    def snapshot(self) -> UUID:
        """Capture the current tensor and metadata state and return its identifier."""
        snapshot_id = uuid4()
        self._snapshots[snapshot_id] = _Snapshot(
            matrix=self.M_t.detach().clone(),
            events=tuple(deepcopy(self.events)),
            access_count=self.access_count,
            last_read=deepcopy(self._last_read),
        )
        return snapshot_id

    def rollback(self, snapshot_id: UUID) -> None:
        """Restore tensor and metadata state from an existing snapshot."""
        try:
            state = self._snapshots[snapshot_id]
        except KeyError as error:
            raise KeyError(f"unknown snapshot_id: {snapshot_id}") from error

        self.M_t = state.matrix.detach().clone()
        self.events = list(deepcopy(state.events))
        self.access_count = state.access_count
        self._last_read = deepcopy(state.last_read)

    def save(self, path: str | Path) -> None:
        """Persist current store state using ``torch.save``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "d_model": self.d_model,
            "lambda_decay": self.lambda_decay,
            "temporal_decay_rate": self.temporal_decay_rate,
            "access_decay_rate": self.access_decay_rate,
            "matrix": self.M_t.detach().cpu(),
            "events": [event.model_dump_json() for event in self.events],
            "access_count": self.access_count,
            "store_id": str(self.store_id),
            "last_read": self._last_read.model_dump_json() if self._last_read else None,
        }
        torch.save(payload, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
    ) -> "AkashaStore":
        """Load a previously persisted store."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported or invalid AkashaStore payload")

        matrix = payload.get("matrix")
        if not isinstance(matrix, Tensor) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("saved memory matrix must be square")

        store = cls(
            d_model=int(payload["d_model"]),
            lambda_decay=float(payload["lambda_decay"]),
            temporal_decay_rate=float(payload["temporal_decay_rate"]),
            access_decay_rate=float(payload["access_decay_rate"]),
            dtype=matrix.dtype,
            device=matrix.device,
        )
        if matrix.shape != store.M_t.shape:
            raise ValueError("saved matrix shape does not match d_model")

        store.M_t = matrix.detach().clone()
        store.events = [
            MemoryEvent.model_validate_json(event_json) for event_json in payload["events"]
        ]
        store.access_count = int(payload["access_count"])
        store.store_id = UUID(str(payload["store_id"]))
        last_read_json = payload.get("last_read")
        store._last_read = (
            MemoryReadResult.model_validate_json(last_read_json) if last_read_json else None
        )
        return store

    def explain_last_read(self) -> dict[str, object]:
        """Return the latest read as a JSON-serializable explanation."""
        if self._last_read is None:
            return {
                "status": "not_read",
                "store_id": str(self.store_id),
                "access_count": self.access_count,
            }
        return {
            "status": "ok",
            **self._last_read.model_dump(mode="json"),
        }

    def _validate_vector(self, vector: Tensor, *, name: str) -> Tensor:
        """Validate and align a memory vector with the store tensor."""
        if vector.ndim != 1 or vector.shape[0] != self.d_model:
            raise ValueError(f"{name} must have shape [{self.d_model}]")
        if not vector.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(vector).all():
            raise ValueError(f"{name} must contain only finite values")
        return vector.detach().to(device=self.M_t.device, dtype=self.M_t.dtype)

    def _event_from_metadata(self, metadata: Mapping[str, object]) -> MemoryEvent:
        """Convert write metadata into a strict MemoryEvent."""
        confidence = metadata.get("confidence", 1.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("metadata confidence must be numeric")

        memory_id_value = metadata.get("memory_id", uuid4())
        memory_id = (
            memory_id_value
            if isinstance(memory_id_value, UUID)
            else UUID(str(memory_id_value))
        )
        provenance = self._string_tuple(metadata.get("provenance", ("akasha_store",)))
        evidence = self._string_tuple(metadata.get("evidence", ()))
        content = json.dumps(dict(metadata), sort_keys=True, default=str)
        if not content or content == "{}":
            content = "Akasha memory write"

        return MemoryEvent(
            memory_id=memory_id,
            event_type=MemoryEventType.WRITE,
            content=content,
            confidence=float(confidence),
            provenance=provenance,
            evidence=evidence,
        )

    @staticmethod
    def _string_tuple(value: object) -> tuple[str, ...]:
        """Normalize a metadata collection into a tuple of non-empty strings."""
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, (list, tuple)):
            values = tuple(value)
        else:
            raise TypeError("metadata provenance and evidence must be string collections")
        if not all(isinstance(item, str) and item for item in values):
            raise ValueError("metadata provenance and evidence entries must be non-empty strings")
        return values

    @staticmethod
    def _retrieval_confidence(retrieved: Tensor) -> float:
        """Map retrieval magnitude to a bounded confidence score."""
        magnitude = float(torch.linalg.vector_norm(retrieved).detach().cpu())
        return 1.0 - math.exp(-magnitude)
