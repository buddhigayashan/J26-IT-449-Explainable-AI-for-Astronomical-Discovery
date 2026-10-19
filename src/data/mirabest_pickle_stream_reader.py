"""Fail-closed streaming reader for the observed MiraBest pickle layout.

This module deliberately does not use Python's pickle unpickler. It scans the
inert protocol-3 byte stream, validates the observed NumPy-array grammar, and
copies only explicitly permitted image payloads.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping

IMAGE_BYTES = 150 * 150
EXPECTED_ROWS = 88
EXPECTED_SHAPE = (150, 150)
EXPECTED_DTYPE = "u1"
MAX_SELECTED_IMAGES = 8
MAX_SELECTED_BYTES = MAX_SELECTED_IMAGES * IMAGE_BYTES
SPLIT_SHA256 = "c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008"
MAGIC_CANONICAL = b"AGN-XAI-RAW-IMAGE-SHA256-v1\0"


class StreamReaderError(RuntimeError):
    """Raised when the serialized member is malformed or unsafe."""


class ForbiddenPayloadError(StreamReaderError):
    """Raised if a forbidden row would have to be materialized."""


@dataclass(frozen=True)
class StreamRecord:
    dataset_index: int
    member: str
    member_row: int
    label: int
    filename: str
    pixels: bytes


@dataclass
class ReaderInstrumentation:
    permitted_payload_extractions: int = 0
    forbidden_payload_encounters: int = 0
    forbidden_payload_extraction_attempts: int = 0
    forbidden_payloads_materialized: int = 0
    maximum_pixel_buffer_size: int = 0
    unexpected_parser_events: int = 0
    payload_boundaries: int = 0
    payload_lengths: tuple[int, ...] = ()
    label_count: int = 0
    filename_count: int = 0
    allowed_row_count: int = 0
    forbidden_row_count: int = 0


@dataclass(frozen=True)
class MemberStructure:
    member: str
    serialized_size: int
    global_start: int
    global_end: int
    payload_count: int
    payload_lengths: tuple[int, ...]
    label_count: int
    filename_count: int


EXPECTED_MEMBER_NAMES = tuple(
    f"batches/data_batch_{index}" for index in range(1, 9)
) + ("batches/test_batch",)
EXPECTED_METADATA_NAMES = frozenset(
    {"batches/hash.md5", "batches/batches.meta"}
)


def split_sets(path: str | Path, expected_sha256: str = SPLIT_SHA256) -> tuple[frozenset[int], frozenset[int]]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise StreamReaderError("Frozen split SHA-256 mismatch.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StreamReaderError("Frozen split manifest is unreadable.") from error
    expected = {"train": 554, "validation": 119, "test": 119}
    if set(value) != set(expected):
        raise StreamReaderError("Unexpected frozen split keys.")
    sets: dict[str, frozenset[int]] = {}
    for name, count in expected.items():
        values = value[name]
        if (not isinstance(values, list) or len(values) != count
                or any(type(item) is not int or not 0 <= item < 792 for item in values)
                or len(set(values)) != count):
            raise StreamReaderError(f"Invalid {name} split.")
        sets[name] = frozenset(values)
    if (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
            or sets["validation"] & sets["test"]
            or set().union(*sets.values()) != set(range(792))):
        raise StreamReaderError("Frozen split overlap or incomplete coverage.")
    return sets["train"] | sets["validation"], sets["test"]


def _file_digest(path: str | Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_archive_manifest(archive: tarfile.TarFile) -> None:
    try:
        manifest_member = archive.getmember("batches/hash.md5")
    except KeyError as error:
        raise StreamReaderError("Missing archive checksum manifest.") from error
    if not manifest_member.isfile():
        raise StreamReaderError("Archive checksum manifest is not a regular file.")
    source = archive.extractfile(manifest_member)
    if source is None:
        raise StreamReaderError("Cannot read archive checksum manifest.")
    entries: dict[str, str] = {}
    for raw_line in source:
        try:
            checksum, name = raw_line.decode("ascii").split()
        except (UnicodeDecodeError, ValueError) as error:
            raise StreamReaderError("Malformed archive checksum manifest.") from error
        if len(checksum) != 32 or any(char not in "0123456789abcdef" for char in checksum):
            raise StreamReaderError("Unsupported archive checksum format.")
        full_name = name if name.startswith("batches/") else f"batches/{name}"
        if full_name in entries:
            raise StreamReaderError("Duplicate archive checksum entry.")
        entries[full_name] = checksum
    expected = set(EXPECTED_MEMBER_NAMES) | {"batches/batches.meta"}
    if set(entries) != expected:
        raise StreamReaderError("Archive checksum manifest coverage mismatch.")
    for name, expected_md5 in entries.items():
        member = archive.getmember(name)
        source = archive.extractfile(member)
        if source is None:
            raise StreamReaderError(f"Cannot read checksum member: {name}")
        digest = hashlib.md5()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_md5:
            raise StreamReaderError(f"Archive member checksum mismatch: {name}")


def _validate_archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise StreamReaderError("Duplicate archive member names.")
    if any(member.issym() or member.islnk() for member in members):
        raise StreamReaderError("Archive links are not supported.")
    files = {member.name: member for member in members if member.isfile()}
    expected = set(EXPECTED_MEMBER_NAMES) | set(EXPECTED_METADATA_NAMES)
    if set(files) != expected:
        raise StreamReaderError("Archive member set mismatch.")
    for name in EXPECTED_MEMBER_NAMES:
        if files[name].size <= 0:
            raise StreamReaderError(f"Empty archive member: {name}")
    return files


def _member_start(member: str) -> int:
    if member == "batches/test_batch":
        return 704
    prefix = "batches/data_batch_"
    if not member.startswith(prefix):
        raise StreamReaderError(f"Unsupported archive member: {member}")
    try:
        number = int(member[len(prefix):])
    except ValueError as error:
        raise StreamReaderError("Malformed data batch member.") from error
    if not 1 <= number <= 8:
        raise StreamReaderError("Data batch number is outside 1..8.")
    return (number - 1) * EXPECTED_ROWS


class _Cursor:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.position = 0

    def read(self, size: int) -> bytes:
        value = self.handle.read(size)
        if len(value) != size:
            raise StreamReaderError("Truncated pickle member.")
        self.position += size
        return value

    def byte(self) -> int:
        return self.read(1)[0]

    def skip(self, size: int) -> None:
        if not self.handle.seekable():
            raise StreamReaderError("Payload skipping requires a seekable member stream.")
        start = self.handle.tell()
        self.handle.seek(size, 1)
        if self.handle.tell() - start != size:
            raise StreamReaderError("Serialized payload skip failed.")
        self.position += size

    def line(self) -> bytes:
        value = bytearray()
        while True:
            char = self.read(1)
            if char == b"\n":
                return bytes(value)
            value.extend(char)
            if len(value) > 512:
                raise StreamReaderError("GLOBAL argument is too long.")


def _decode_filename(value: bytes) -> str | None:
    if len(value) == 180:
        try:
            text = value.decode("utf-32-le").rstrip("\0")
        except UnicodeDecodeError:
            return None
        return text if text.endswith(".png") else None
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text.endswith(".png") else None


def _read_pickle_bytes_header(cursor: _Cursor, opcode: int) -> int:
    if opcode == 0x42:  # BINBYTES
        length = struct.unpack("<I", cursor.read(4))[0]
    elif opcode == 0x43:  # SHORT_BINBYTES
        length = cursor.byte()
    else:
        raise StreamReaderError("Internal binary-opcode error.")
    if length > 22500 and length != 180:
        raise StreamReaderError("Unsupported serialized byte-string length.")
    return length


def _scan_member(handle: BinaryIO, member: str, allowed: frozenset[int],
                 forbidden: frozenset[int], instrumentation: ReaderInstrumentation,
                 extract_pixels: bool,
                 extract_indices: frozenset[int] | None = None) -> list[StreamRecord]:
    cursor = _Cursor(handle)
    if cursor.byte() != 0x80 or cursor.byte() != 3:
        raise StreamReaderError("Only pickle protocol 3 is supported.")

    records: list[StreamRecord] = []
    payload_positions: list[tuple[int, int]] = []
    labels: list[int] = []
    filenames: list[str] = []
    data_started = False
    labels_started = False
    filenames_started = False
    image_row = 0
    saw_data_key = saw_labels_key = saw_filenames_key = False
    shape_values: list[int] = []
    saw_dtype = False
    saw_stop = False

    while True:
        opcode = cursor.byte()
        if opcode == 0x2E:  # STOP
            saw_stop = True
            break
        if opcode == 0x71:  # BINPUT
            cursor.skip(1)
        elif opcode == 0x68:  # BINGET
            cursor.skip(1)
        elif opcode == 0x72:  # LONG_BINPUT
            cursor.skip(4)
        elif opcode == 0x6A:  # LONG_BINGET
            cursor.skip(4)
        elif opcode in (0x70, 0x94):  # PUT, MEMOIZE
            if opcode == 0x70:
                cursor.line()
        elif opcode in (0x28, 0x5D, 0x7D, 0x4E, 0x88, 0x89, 0x4B, 0x4A):
            if opcode == 0x4B:
                value = cursor.byte()
                if data_started and image_row == 0 and value in (150,):
                    shape_values.append(value)
                if labels_started and not data_started:
                    if value not in (0, 1):
                        raise StreamReaderError("Non-binary label encountered.")
                    labels.append(value)
            elif opcode == 0x4A:
                cursor.skip(4)
        elif opcode in (0x58, 0x55):  # BINUNICODE, SHORT_BINUNICODE
            length = struct.unpack("<I", cursor.read(4))[0] if opcode == 0x58 else cursor.byte()
            if length > 1024:
                raise StreamReaderError("Unicode metadata is too long.")
            value = cursor.read(length).decode("utf-8")
            if value == "data":
                if saw_data_key or not saw_labels_key or len(labels) != EXPECTED_ROWS:
                    raise StreamReaderError("Data section is out of order.")
                saw_data_key = data_started = True
                labels_started = False
            elif value == "labels":
                if saw_labels_key or saw_data_key:
                    raise StreamReaderError("Labels section is out of order.")
                saw_labels_key = labels_started = True
            elif value == "filenames":
                if not saw_data_key or image_row != EXPECTED_ROWS or saw_filenames_key:
                    raise StreamReaderError("Filenames section is out of order.")
                saw_filenames_key = filenames_started = True
                data_started = labels_started = False
            elif value == EXPECTED_DTYPE:
                saw_dtype = True
            elif filenames_started and value.endswith(".png"):
                filenames.append(value)
        elif opcode == 0x63:  # GLOBAL
            argument = cursor.line() + b" " + cursor.line()
            if argument not in {
                b"numpy.core.multiarray _reconstruct",
                b"numpy._core.multiarray _reconstruct",
                b"numpy ndarray",
                b"numpy dtype",
                b"numpy.core.multiarray scalar",
            }:
                raise StreamReaderError("Unsupported GLOBAL reference.")
        elif opcode == 0x42 or opcode == 0x43:
            length = _read_pickle_bytes_header(cursor, opcode)
            if opcode == 0x42 and data_started and length != IMAGE_BYTES:
                raise StreamReaderError("Image payload length is not 22,500 bytes.")
            if opcode == 0x42 and length == IMAGE_BYTES:
                if not data_started:
                    raise StreamReaderError("Image payload outside data section.")
                if image_row >= EXPECTED_ROWS:
                    raise StreamReaderError("Too many image payloads.")
                index = _member_start(member) + image_row
                image_row += 1
                if index in forbidden:
                    instrumentation.forbidden_payload_encounters += 1
                    cursor.skip(length)
                    continue
                if index not in allowed:
                    instrumentation.unexpected_parser_events += 1
                    raise StreamReaderError("Image index is outside the authorized scope.")
                if extract_indices is None or index in extract_indices:
                    payload_positions.append((index, cursor.position))
                cursor.skip(length)
            elif opcode == 0x43:
                payload = cursor.read(length)
                filename = _decode_filename(payload)
                if filenames_started and filename is not None:
                    filenames.append(filename)
        elif opcode in (0x74, 0x85, 0x86, 0x87, 0x81, 0x82, 0x52, 0x62, 0x5C, 0x4C):
            if opcode == 0x4C:  # LONG_BINPUT-like legacy memo variant
                cursor.skip(4)
        elif opcode == 0x53:  # STRING, not part of the observed grammar
            raise StreamReaderError("Unsupported STRING opcode.")
        elif opcode == 0x50 or opcode == 0x51:
            raise StreamReaderError("Unsupported text pickle opcode.")
        elif opcode == 0x80:
            raise StreamReaderError("Unexpected nested PROTO opcode.")
        else:
            # The observed stream contains many stack/memo opcodes. Only the
            # inert one-byte forms are accepted here; unknown bytes fail closed.
            if opcode not in {0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36,
                              0x37, 0x38, 0x39, 0x61, 0x62, 0x65, 0x67,
                              0x68, 0x6A, 0x6B, 0x6C, 0x6D, 0x6F, 0x75,
                              0x76, 0x80, 0x81, 0x82, 0x85, 0x86, 0x87,
                              0x90, 0x91, 0x92, 0x93, 0x95}:
                raise StreamReaderError(
                    f"Unsupported pickle opcode 0x{opcode:02x} at offset {cursor.position - 1}."
                )

    if not saw_stop or not saw_data_key or not saw_labels_key or not saw_filenames_key:
        raise StreamReaderError("Incomplete MiraBest member sections.")
    if image_row != EXPECTED_ROWS or len(labels) != EXPECTED_ROWS or len(filenames) != EXPECTED_ROWS:
        raise StreamReaderError(
            f"MiraBest member row counts invalid: images={image_row}, "
            f"labels={len(labels)}, filenames={len(filenames)}."
        )
    if shape_values != [150, 150] or not saw_dtype:
        raise StreamReaderError("Unsupported image shape or dtype metadata.")
    instrumentation.payload_boundaries = image_row
    instrumentation.payload_lengths = (IMAGE_BYTES,)
    instrumentation.label_count = len(labels)
    instrumentation.filename_count = len(filenames)
    instrumentation.allowed_row_count = len(payload_positions)
    instrumentation.forbidden_row_count = (
        instrumentation.forbidden_payload_encounters
    )
    if instrumentation.forbidden_payloads_materialized:
        raise StreamReaderError("Forbidden payload materialization detected.")
    if extract_pixels:
        start = _member_start(member)
        for index, position in payload_positions:
            if index in forbidden:
                instrumentation.forbidden_payload_extraction_attempts += 1
                raise ForbiddenPayloadError("Forbidden payload reached extraction pass.")
            handle.seek(position)
            payload = handle.read(IMAGE_BYTES)
            if len(payload) != IMAGE_BYTES:
                raise StreamReaderError("Permitted payload was truncated.")
            instrumentation.permitted_payload_extractions += 1
            instrumentation.maximum_pixel_buffer_size = max(
                instrumentation.maximum_pixel_buffer_size, len(payload)
            )
            row = index - start
            records.append(StreamRecord(index, member, row, labels[row], filenames[row], payload))
        records.sort(key=lambda item: item.member_row)
    return records


def read_member(path: str | Path, member: str, allowed_indices: Iterable[int],
                forbidden_indices: Iterable[int], *,
                extract_pixels: bool = True,
                instrumentation: ReaderInstrumentation | None = None) -> list[StreamRecord]:
    allowed = frozenset(allowed_indices)
    forbidden = frozenset(forbidden_indices)
    if allowed & forbidden:
        raise StreamReaderError("Allowed and forbidden sets overlap.")
    if instrumentation is None:
        instrumentation = ReaderInstrumentation()
    with Path(path).open("rb") as handle:
        return _scan_member(handle, member, allowed, forbidden, instrumentation, extract_pixels)


def _validate_selected_indices(selected_indices: Iterable[int],
                               allowed: frozenset[int],
                               forbidden: frozenset[int]) -> frozenset[int]:
    values = list(selected_indices)
    if len(values) > MAX_SELECTED_IMAGES:
        raise StreamReaderError("Selected image count exceeds the pilot limit.")
    if len(values) != len(set(values)):
        raise StreamReaderError("Selected image indices contain duplicates.")
    if any(type(index) is not int or not 0 <= index < 792 for index in values):
        raise StreamReaderError("Selected image index is invalid.")
    selected = frozenset(values)
    if selected & forbidden:
        raise StreamReaderError("Selected image index belongs to frozen TEST.")
    if not selected <= allowed:
        raise StreamReaderError("Selected image index is outside TRAIN/VALIDATION.")
    if len(selected) * IMAGE_BYTES > MAX_SELECTED_BYTES:
        raise StreamReaderError("Selected image payload budget exceeded.")
    return selected


def extract_selected_archive(
    path: str | Path,
    allowed_indices: Iterable[int],
    forbidden_indices: Iterable[int],
    selected_indices: Iterable[int],
    *,
    expected_sha256: str,
    expected_md5: str,
    validate_internal_manifest: bool = True,
    instrumentation: ReaderInstrumentation | None = None,
) -> list[StreamRecord]:
    """Extract only an explicitly bounded subset from a pinned archive."""
    allowed = frozenset(allowed_indices)
    forbidden = frozenset(forbidden_indices)
    if allowed & forbidden:
        raise StreamReaderError("Allowed and forbidden sets overlap.")
    if (len(allowed) != 673 or len(forbidden) != 119
            or allowed | forbidden != frozenset(range(792))):
        raise StreamReaderError("Archive extraction requires the complete frozen research split.")
    selected = _validate_selected_indices(selected_indices, allowed, forbidden)
    if instrumentation is None:
        instrumentation = ReaderInstrumentation()
    if _file_digest(path, "sha256") != expected_sha256:
        raise StreamReaderError("Archive SHA-256 mismatch.")
    if _file_digest(path, "md5") != expected_md5:
        raise StreamReaderError("Archive MD5 mismatch.")
    try:
        archive = tarfile.open(path, mode="r:gz")
    except (OSError, tarfile.TarError) as error:
        raise StreamReaderError("Cannot open gzip tar archive.") from error
    records: list[StreamRecord] = []
    with archive:
        files = _validate_archive_members(archive)
        if validate_internal_manifest:
            _validate_archive_manifest(archive)
        for member_name in EXPECTED_MEMBER_NAMES:
            member = files[member_name]
            stream = archive.extractfile(member)
            if stream is None:
                raise StreamReaderError(f"Cannot open archive member: {member_name}")
            before = ReaderInstrumentation()
            records.extend(_scan_member(
                stream, member_name, allowed, forbidden, before,
                True, selected,
            ))
            instrumentation.permitted_payload_extractions += before.permitted_payload_extractions
            instrumentation.forbidden_payload_encounters += before.forbidden_payload_encounters
            instrumentation.forbidden_payload_extraction_attempts += before.forbidden_payload_extraction_attempts
            instrumentation.forbidden_payloads_materialized += before.forbidden_payloads_materialized
            instrumentation.maximum_pixel_buffer_size = max(
                instrumentation.maximum_pixel_buffer_size,
                before.maximum_pixel_buffer_size,
            )
            instrumentation.unexpected_parser_events += before.unexpected_parser_events
            instrumentation.payload_boundaries += before.payload_boundaries
            instrumentation.label_count += before.label_count
            instrumentation.filename_count += before.filename_count
            instrumentation.allowed_row_count += before.allowed_row_count
            instrumentation.forbidden_row_count += before.forbidden_row_count
            instrumentation.payload_lengths = tuple(sorted(set(
                instrumentation.payload_lengths + before.payload_lengths
            )))
    records.sort(key=lambda record: record.dataset_index)
    if len(records) != len(selected):
        raise StreamReaderError("Selected image extraction count mismatch.")
    return records


def scan_archive(path: str | Path, allowed_indices: Iterable[int],
                 forbidden_indices: Iterable[int], *,
                 expected_sha256: str | None = None,
                 expected_md5: str | None = None,
                 validate_internal_manifest: bool = True,
                 extract_pixels: bool = False,
                 instrumentation: ReaderInstrumentation | None = None) -> list[MemberStructure]:
    """Validate synthetic or separately authorized archives without eager loading."""
    allowed = frozenset(allowed_indices)
    forbidden = frozenset(forbidden_indices)
    if allowed & forbidden:
        raise StreamReaderError("Allowed and forbidden sets overlap.")
    if (len(allowed) != 673 or len(forbidden) != 119
            or allowed | forbidden != frozenset(range(792))):
        raise StreamReaderError("Archive scan requires the complete frozen research split.")
    if instrumentation is None:
        instrumentation = ReaderInstrumentation()
    if expected_sha256 is not None and _file_digest(path, "sha256") != expected_sha256:
        raise StreamReaderError("Archive SHA-256 mismatch.")
    if expected_md5 is not None and _file_digest(path, "md5") != expected_md5:
        raise StreamReaderError("Archive MD5 mismatch.")
    if extract_pixels and (expected_sha256 is None or expected_md5 is None):
        raise StreamReaderError("Pixel extraction requires both archive integrity pins.")
    try:
        archive = tarfile.open(path, mode="r:gz")
    except (OSError, tarfile.TarError) as error:
        raise StreamReaderError("Cannot open gzip tar archive.") from error
    with archive:
        files = _validate_archive_members(archive)
        if validate_internal_manifest:
            _validate_archive_manifest(archive)
        structures: list[MemberStructure] = []
        for member_name in EXPECTED_MEMBER_NAMES:
            member = files[member_name]
            stream = archive.extractfile(member)
            if stream is None:
                raise StreamReaderError(f"Cannot open archive member: {member_name}")
            before = ReaderInstrumentation()
            _scan_member(stream, member_name, allowed, forbidden, before, extract_pixels)
            start = _member_start(member_name)
            structures.append(MemberStructure(
                member_name, member.size, start, start + EXPECTED_ROWS - 1,
                before.payload_boundaries, before.payload_lengths,
                before.label_count, before.filename_count,
            ))
            instrumentation.permitted_payload_extractions += before.permitted_payload_extractions
            instrumentation.forbidden_payload_encounters += before.forbidden_payload_encounters
            instrumentation.forbidden_payload_extraction_attempts += before.forbidden_payload_extraction_attempts
            instrumentation.forbidden_payloads_materialized += before.forbidden_payloads_materialized
            instrumentation.maximum_pixel_buffer_size = max(
                instrumentation.maximum_pixel_buffer_size,
                before.maximum_pixel_buffer_size,
            )
            instrumentation.unexpected_parser_events += before.unexpected_parser_events
            instrumentation.payload_boundaries += before.payload_boundaries
            instrumentation.label_count += before.label_count
            instrumentation.filename_count += before.filename_count
            instrumentation.allowed_row_count += before.allowed_row_count
            instrumentation.forbidden_row_count += before.forbidden_row_count
            instrumentation.payload_lengths = tuple(sorted(set(
                instrumentation.payload_lengths + before.payload_lengths
            )))
    return structures


def canonical_sha256(pixels: bytes) -> str:
    if type(pixels) is not bytes or len(pixels) != IMAGE_BYTES:
        raise ValueError("Expected raw 150x150 uint8 bytes.")
    return hashlib.sha256(
        MAGIC_CANONICAL + b"<u1\0" + struct.pack(">II", *EXPECTED_SHAPE) + pixels
    ).hexdigest()
