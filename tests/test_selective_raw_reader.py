import json
import struct

import pytest

from src.data.selective_raw_reader import (
    MAGIC,
    SelectiveRawReader,
    SelectiveReaderError,
    ForbiddenIndexError,
    canonical_sha256,
    inspect_batch,
)


def write_batch(path, batch_number, rows):
    header = {
        "row_count": len(rows),
        "shape": [150, 150],
        "dtype": "uint8",
        "labels": [label for label, _, _ in rows],
        "filenames": [name for _, name, _ in rows],
    }
    with path.open("wb") as handle:
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        handle.write(MAGIC)
        handle.write(struct.pack(">I", len(raw)))
        handle.write(raw)
        for _, _, pixels in rows:
            handle.write(struct.pack(">I", len(pixels)))
            handle.write(pixels)


def reader_fixture(tmp_path):
    pixels_a = bytes([1]) * (150 * 150)
    pixels_b = bytes([2]) * (150 * 150)
    path = tmp_path / "data_batch_1"
    write_batch(path, 1, [
        (0, "a.png", pixels_a),
        (1, "b.png", pixels_a),
        (1, "c.png", pixels_b),
    ])
    return path, pixels_a, pixels_b


def test_allowed_rows_are_selectively_decoded(tmp_path):
    path, pixels_a, _ = reader_fixture(tmp_path)
    reader = SelectiveRawReader({"data_batch_1": path}, {0, 1}, {2})
    result = reader.read(1)
    assert result.pixels == pixels_a
    assert result.metadata.dataset_index == 1
    assert reader.payload_reads == 1


def test_forbidden_rows_are_rejected_before_payload_read(tmp_path):
    path, _, _ = reader_fixture(tmp_path)
    reader = SelectiveRawReader({"data_batch_1": path}, {0, 1}, {2})
    with pytest.raises(ForbiddenIndexError):
        reader.read(2)
    assert reader.payload_reads == 0
    assert reader.test_payload_reads == 0


def test_malformed_or_pickle_batches_fail_closed(tmp_path):
    path = tmp_path / "bad"
    path.write_bytes(b"\x80\x05ordinary-pickle")
    with pytest.raises(SelectiveReaderError):
        inspect_batch(path, "data_batch_1", 0)


@pytest.mark.parametrize("shape,dtype", [([149, 150], "uint8"), ([150, 150], "uint16")])
def test_unexpected_shape_or_dtype_fails_closed(tmp_path, shape, dtype):
    path = tmp_path / "data_batch_1"
    header = {"row_count": 1, "shape": shape, "dtype": dtype,
              "labels": [0], "filenames": ["a.png"]}
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(MAGIC + struct.pack(">I", len(raw)) + raw)
    with pytest.raises(SelectiveReaderError):
        inspect_batch(path, "data_batch_1", 0)


def test_hash_is_deterministic():
    pixels = bytes([7]) * (150 * 150)
    assert canonical_sha256(pixels) == canonical_sha256(pixels)
    assert canonical_sha256(pixels) != canonical_sha256(bytes([8]) * (150 * 150))
