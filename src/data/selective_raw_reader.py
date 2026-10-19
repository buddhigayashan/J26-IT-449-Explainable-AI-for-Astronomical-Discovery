"""Fail-closed selective reader used by the duplicate-audit safety harness.

The real MiraBest batches are pickle files whose image-payload layout has not
been approved for row-level extraction.  This module therefore accepts only
the small ``MBRAW1`` streaming fixture format used by synthetic tests.  Any
other format, including an ordinary pickle batch, is rejected before payload
access.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

MAGIC = b"MBRAW1\0"
EXPECTED_SHAPE = (150, 150)
EXPECTED_DTYPE = "uint8"
HEADER_STRUCT = struct.Struct(">I")
ROW_LENGTH_STRUCT = struct.Struct(">I")
CANONICAL_MAGIC = b"AGN-XAI-RAW-IMAGE-SHA256-v1\0"
CANONICAL_DTYPE = b"<u1\0"
CANONICAL_SHAPE_STRUCT = struct.Struct(">II")


class SelectiveReaderError(RuntimeError):
    """Base error for malformed or unsafe selective access."""


class ForbiddenIndexError(SelectiveReaderError):
    """Raised before any payload read for a forbidden index."""


@dataclass(frozen=True)
class RowMetadata:
    dataset_index: int
    label: int
    filename: str
    member: str
    member_row: int


@dataclass(frozen=True)
class RawImage:
    metadata: RowMetadata
    pixels: bytes


@dataclass(frozen=True)
class BatchMetadata:
    member: str
    row_count: int
    labels: tuple[int, ...]
    filenames: tuple[str, ...]
    payload_offsets: tuple[int, ...]
    payload_lengths: tuple[int, ...]


def canonical_bytes(pixels: bytes, shape: tuple[int, int] = EXPECTED_SHAPE,
                    dtype: str = EXPECTED_DTYPE) -> bytes:
    if type(pixels) is not bytes:
        raise ValueError("Pixels must be raw bytes.")
    if shape != EXPECTED_SHAPE or dtype != EXPECTED_DTYPE:
        raise ValueError("Only 150x150 uint8 images are supported.")
    if len(pixels) != shape[0] * shape[1]:
        raise ValueError("Pixel payload length does not match the expected shape.")
    return (CANONICAL_MAGIC + CANONICAL_DTYPE
            + CANONICAL_SHAPE_STRUCT.pack(*shape) + pixels)


def canonical_sha256(pixels: bytes) -> str:
    return hashlib.sha256(canonical_bytes(pixels)).hexdigest()


def _read_exact(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise SelectiveReaderError("Truncated synthetic batch.")
    return value


def _read_header(handle, member: str) -> tuple[int, tuple[int, int], str, tuple[int, ...], tuple[str, ...]]:
    if _read_exact(handle, len(MAGIC)) != MAGIC:
        raise SelectiveReaderError(
            f"Unsupported batch format for {member}; real pickle batches are rejected."
        )
    header_length = HEADER_STRUCT.unpack(_read_exact(handle, HEADER_STRUCT.size))[0]
    if header_length > 1024 * 1024:
        raise SelectiveReaderError("Synthetic header is unreasonably large.")
    try:
        header = json.loads(_read_exact(handle, header_length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectiveReaderError("Malformed synthetic batch header.") from error
    if not isinstance(header, dict):
        raise SelectiveReaderError("Synthetic header must be an object.")
    shape = tuple(header.get("shape", ()))
    dtype = header.get("dtype")
    labels = tuple(header.get("labels", ()))
    filenames = tuple(header.get("filenames", ()))
    row_count = header.get("row_count")
    if (type(row_count) is not int or row_count < 0 or shape != EXPECTED_SHAPE
            or dtype != EXPECTED_DTYPE or len(labels) != row_count
            or len(filenames) != row_count):
        raise SelectiveReaderError("Unsupported synthetic batch shape, dtype, or metadata.")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise SelectiveReaderError("Synthetic labels must be binary integers.")
    if any(type(name) is not str or not name for name in filenames):
        raise SelectiveReaderError("Synthetic filenames must be nonempty strings.")
    return row_count, shape, dtype, labels, filenames


def inspect_batch(path: str | Path, member: str, start_index: int) -> BatchMetadata:
    """Read metadata and row lengths without reading any pixel payload."""
    path = Path(path)
    with path.open("rb") as handle:
        row_count, _, _, labels, filenames = _read_header(handle, member)
        offsets = []
        lengths = []
        for row in range(row_count):
            length_position = handle.tell()
            length = ROW_LENGTH_STRUCT.unpack(_read_exact(handle, ROW_LENGTH_STRUCT.size))[0]
            if length != EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]:
                raise SelectiveReaderError("Unsupported raw row payload length.")
            offsets.append(length_position + ROW_LENGTH_STRUCT.size)
            lengths.append(length)
            handle.seek(length, 1)
        if handle.read(1):
            raise SelectiveReaderError("Unexpected trailing bytes in synthetic batch.")
    return BatchMetadata(member, row_count, labels, filenames,
                         tuple(offsets), tuple(lengths))


class SelectiveRawReader:
    """Read only explicitly allowed rows from synthetic streaming batches."""

    def __init__(self, batches: Mapping[str, str | Path],
                 allowed_indices: Iterable[int], forbidden_indices: Iterable[int]):
        self._batches = {str(member): Path(path) for member, path in batches.items()}
        self.allowed_indices = frozenset(allowed_indices)
        self.forbidden_indices = frozenset(forbidden_indices)
        if self.allowed_indices & self.forbidden_indices:
            raise SelectiveReaderError("Allowed and forbidden index sets overlap.")
        self._metadata: dict[int, tuple[Path, BatchMetadata, int]] = {}
        self.test_payload_reads = 0
        self.payload_reads = 0
        for member, path in self._batches.items():
            metadata = inspect_batch(path, member, 0)
            for row in range(metadata.row_count):
                index = self._index_for_row(member, metadata, row)
                if index in self._metadata:
                    raise SelectiveReaderError("Duplicate dataset index in batch mapping.")
                self._metadata[index] = (path, metadata, row)

    @staticmethod
    def _index_for_row(member: str, metadata: BatchMetadata, row: int) -> int:
        if not member.startswith("data_batch_"):
            raise SelectiveReaderError("Synthetic fixtures require data_batch_N members.")
        try:
            batch_number = int(member.removeprefix("data_batch_"))
        except ValueError as error:
            raise SelectiveReaderError("Malformed batch member name.") from error
        if batch_number < 1:
            raise SelectiveReaderError("Batch number must be positive.")
        return (batch_number - 1) * metadata.row_count + row

    def metadata(self, index: int) -> RowMetadata:
        if type(index) is not int:
            raise ForbiddenIndexError("Dataset index must be an integer.")
        if index in self.forbidden_indices or index not in self.allowed_indices:
            raise ForbiddenIndexError("Index is outside the permitted development scope.")
        try:
            path, batch, row = self._metadata[index]
        except KeyError as error:
            raise SelectiveReaderError("Permitted index is absent from batch metadata.") from error
        return RowMetadata(index, batch.labels[row], batch.filenames[row],
                          batch.member, row)

    def read(self, index: int) -> RawImage:
        metadata = self.metadata(index)
        path, batch, row = self._metadata[index]
        with path.open("rb") as handle:
            handle.seek(batch.payload_offsets[row])
            pixels = _read_exact(handle, batch.payload_lengths[row])
        self.payload_reads += 1
        if index in self.forbidden_indices:
            self.test_payload_reads += 1
            raise ForbiddenIndexError("Forbidden payload read detected.")
        return RawImage(metadata, pixels)
