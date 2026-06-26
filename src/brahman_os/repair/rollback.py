"""File snapshot and restoration support for repair attempts."""

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class _FileState:
    """Byte-for-byte state of a repository file."""

    path: Path
    existed: bool
    content: bytes | None


class RollbackManager:
    """Create and restore in-memory snapshots of repository files."""

    def __init__(self, repo_root: str | Path) -> None:
        """Initialize snapshot storage for one repository root."""
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"repo_root does not exist: {self.repo_root}")
        self._snapshots: dict[UUID, tuple[_FileState, ...]] = {}
        self._lock = RLock()

    def snapshot(self, paths: tuple[str | Path, ...]) -> UUID:
        """Snapshot existing files and remember which paths did not exist."""
        if not paths:
            raise ValueError("at least one path is required for a snapshot")

        states = tuple(self._capture(path) for path in paths)
        snapshot_id = uuid4()
        with self._lock:
            self._snapshots[snapshot_id] = states
        return snapshot_id

    def create_snapshot(self, paths: tuple[str | Path, ...]) -> UUID:
        """Compatibility alias for snapshot creation."""
        return self.snapshot(paths)

    def explain_snapshot(self, snapshot_id: UUID) -> dict[str, object]:
        """Return JSON-serializable snapshot metadata without file contents."""
        with self._lock:
            try:
                states = self._snapshots[snapshot_id]
            except KeyError as error:
                raise KeyError(f"unknown snapshot_id: {snapshot_id}") from error

            return {
                "snapshot_id": str(snapshot_id),
                "repo_root": str(self.repo_root),
                "file_count": len(states),
                "files": [
                    {
                        "path": state.path.relative_to(self.repo_root).as_posix(),
                        "existed": state.existed,
                        "byte_size": len(state.content) if state.content is not None else 0,
                    }
                    for state in states
                ],
            }

    def rollback(self, snapshot_id: UUID) -> None:
        """Restore all files captured by a snapshot."""
        with self._lock:
            try:
                states = self._snapshots[snapshot_id]
            except KeyError as error:
                raise KeyError(f"unknown snapshot_id: {snapshot_id}") from error

            for state in states:
                if state.existed:
                    state.path.parent.mkdir(parents=True, exist_ok=True)
                    state.path.write_bytes(state.content or b"")
                elif state.path.exists():
                    if not state.path.is_file():
                        raise ValueError(f"rollback target is not a file: {state.path}")
                    state.path.unlink()

    def _capture(self, path: str | Path) -> _FileState:
        """Capture one confined repository path."""
        resolved = self._resolve_repo_path(path)
        if resolved.exists() and not resolved.is_file():
            raise ValueError(f"snapshot path is not a file: {resolved}")
        return _FileState(
            path=resolved,
            existed=resolved.is_file(),
            content=resolved.read_bytes() if resolved.is_file() else None,
        )

    def _resolve_repo_path(self, path: str | Path) -> Path:
        """Resolve a path and reject anything outside the repository."""
        candidate = Path(path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.repo_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(f"path is outside repository: {path}") from error
        return resolved
