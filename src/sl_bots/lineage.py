"""Closed data-purpose lineage for test and production artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

from .contracts import DataPurpose, ensure_purpose


class ProductionLineageError(ValueError):
    """Raised when a test-only ancestor reaches a production artifact."""


@dataclass(frozen=True)
class DatasetManifestV1:
    name: str
    purpose: DataPurpose
    parents: tuple["DatasetManifestV1", ...] = field(default_factory=tuple)
    artifact_type: str = "dataset"
    source_sha256: str | None = None
    parser_version: str | None = None
    projection_version: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("manifest name cannot be empty")
        object.__setattr__(self, "purpose", ensure_purpose(self.purpose))
        parents = tuple(self.parents)
        if any(not isinstance(parent, DatasetManifestV1) for parent in parents):
            raise TypeError("parents must contain DatasetManifestV1 values")
        object.__setattr__(self, "parents", parents)
        if not self.artifact_type:
            raise ValueError("artifact_type cannot be empty")
        if self.source_sha256 is not None and len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a 64-character hex digest")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def source_demo_sha256(self) -> str | None:
        return self.source_sha256

    def effective_purpose(self) -> DataPurpose:
        return merge_lineage((self,))

    def assert_exportable(
        self, target_purpose: DataPurpose | str = DataPurpose.PRODUCTION
    ) -> None:
        target_purpose = ensure_purpose(target_purpose)
        if target_purpose is DataPurpose.PRODUCTION and self.effective_purpose() is DataPurpose.TEST_ONLY:
            raise ProductionLineageError(
                f"{self.name} has a test_only ancestor and cannot be exported"
            )


def _purposes(
    manifest: DatasetManifestV1,
    active: set[int],
    visited: set[int],
) -> Iterable[DataPurpose]:
    identity = id(manifest)
    if identity in active:
        raise ValueError("lineage cycle detected")
    if identity in visited:
        return
    active.add(identity)
    yield manifest.purpose
    for parent in manifest.parents:
        yield from _purposes(parent, active, visited)
    active.remove(identity)
    visited.add(identity)


def merge_lineage(parents: Iterable[DatasetManifestV1]) -> DataPurpose:
    manifests = tuple(parents)
    if any(not isinstance(parent, DatasetManifestV1) for parent in manifests):
        raise TypeError("lineage parents must be DatasetManifestV1 values")
    visited: set[int] = set()
    purposes = (
        purpose
        for manifest in manifests
        for purpose in _purposes(manifest, set(), visited)
    )
    return (
        DataPurpose.TEST_ONLY
        if any(purpose is DataPurpose.TEST_ONLY for purpose in purposes)
        else DataPurpose.PRODUCTION
    )
