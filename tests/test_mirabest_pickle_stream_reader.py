import hashlib
import json
import pickle
import struct
import tarfile

import numpy as np
import pytest

from src.data.mirabest_pickle_stream_reader import (
    EXPECTED_ROWS,
    ReaderInstrumentation,
    StreamReaderError,
    canonical_sha256,
    extract_selected_archive,
    read_member,
    scan_archive,
)


def _fixture(path, member="batches/data_batch_1", mutate=None, binary_filenames=False):
    data = []
    labels = []
    filenames = []
    base = np.zeros((150, 150), dtype=np.uint8)
    for row in range(EXPECTED_ROWS):
        pixels = base.copy()
        pixels[0, 0] = row
        data.append(pixels)
        labels.append(row % 2)
        filename = f"BestCat_proc/Cropped/0_{row:03d}_+{row:03d}.png"
        filenames.append(filename.encode("utf-32-le").ljust(180, b"\0")
                         if binary_filenames else filename)
    if mutate:
        mutate(data, labels, filenames)
    payload = pickle.dumps(
        {"batch_label": "synthetic", "labels": labels, "data": data,
         "filenames": filenames},
        protocol=3,
    )
    path.write_bytes(payload)
    return path


def _archive_fixture(path):
    members = {}
    for index in range(1, 9):
        member_path = path.parent / f"data_batch_{index}"
        _fixture(member_path, binary_filenames=(index % 2 == 0))
        members[f"batches/data_batch_{index}"] = member_path.read_bytes()
    test_path = path.parent / "test_batch"
    _fixture(test_path, member="batches/test_batch", binary_filenames=True)
    members["batches/test_batch"] = test_path.read_bytes()
    members["batches/batches.meta"] = b"synthetic metadata"
    lines = []
    for name, payload in members.items():
        lines.append(f"{hashlib.md5(payload).hexdigest()}  {name}\n".encode("ascii"))
    members["batches/hash.md5"] = b"".join(lines)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, __import__("io").BytesIO(payload))


def _archive_pins(path):
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        hashlib.md5(path.read_bytes()).hexdigest(),
    )


def test_valid_protocol_three_fixture_and_index_mapping(tmp_path):
    path = _fixture(tmp_path / "data_batch_1")
    allowed = frozenset(range(88))
    records = read_member(path, "batches/data_batch_1", allowed, frozenset())
    assert len(records) == 88
    assert records[0].dataset_index == 0
    assert records[-1].dataset_index == 87
    assert records[0].pixels[0] == 0


def test_real_dictionary_key_order_regression(tmp_path):
    data = [np.zeros((150, 150), dtype=np.uint8) for _ in range(EXPECTED_ROWS)]
    payload = pickle.dumps(
        {"batch_label": "synthetic", "labels": [0] * EXPECTED_ROWS,
         "data": data,
         "filenames": [f"BestCat_proc/Cropped/0_{i:03d}_+{i:03d}.png"
                       for i in range(EXPECTED_ROWS)]},
        protocol=3,
    )
    path = tmp_path / "real_key_order"
    path.write_bytes(payload)
    records = read_member(path, "batches/data_batch_1", range(88), ())
    assert len(records) == 88


def test_test_batch_global_mapping(tmp_path):
    path = _fixture(tmp_path / "test_batch", binary_filenames=True)
    records = read_member(path, "batches/test_batch", frozenset(range(704, 792)), frozenset())
    assert records[0].dataset_index == 704
    assert records[-1].dataset_index == 791


def test_second_observed_filename_encoding(tmp_path):
    path = _fixture(tmp_path / "test_batch", binary_filenames=True)
    records = read_member(path, "batches/test_batch", {704}, set(range(705, 792)))
    assert records[0].filename.endswith(".png")


def test_forbidden_rows_are_skipped_without_materialization(tmp_path):
    path = _fixture(tmp_path / "data_batch_1")
    instrumentation = ReaderInstrumentation()
    records = read_member(path, "batches/data_batch_1", frozenset(range(1, 88)),
                          frozenset({0}), instrumentation=instrumentation)
    assert len(records) == 87
    assert instrumentation.permitted_payload_extractions == 87
    assert instrumentation.forbidden_payload_encounters == 1
    assert instrumentation.forbidden_payload_extraction_attempts == 0
    assert instrumentation.forbidden_payloads_materialized == 0


def test_extraction_path_fails_closed_for_forbidden_rows(tmp_path):
    path = _fixture(tmp_path / "data_batch_1")
    instrumentation = ReaderInstrumentation()
    records = read_member(path, "batches/data_batch_1", frozenset(range(1, 88)),
                          frozenset({0}), extract_pixels=True,
                          instrumentation=instrumentation)
    assert len(records) == 87
    assert instrumentation.forbidden_payloads_materialized == 0


def test_truncated_and_unknown_opcode_fail_closed(tmp_path):
    path = _fixture(tmp_path / "data_batch_1")
    raw = path.read_bytes()
    path.write_bytes(raw[:-10])
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())
    path.write_bytes(b"\x80\x03\xff")
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())


def test_wrong_shape_dtype_and_length_fail_closed(tmp_path):
    def wrong_shape(data, labels, filenames):
        data[0] = np.zeros((149, 150), dtype=np.uint8)
    path = tmp_path / "wrong_shape"
    _fixture(path, mutate=wrong_shape)
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())

    path = tmp_path / "wrong_length"
    _fixture(path)
    raw = bytearray(path.read_bytes())
    first_payload = raw.index(b"\x42")
    raw[first_payload + 1:first_payload + 5] = struct.pack("<I", 22499)
    path.write_bytes(raw)
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())


def test_invalid_counts_and_section_order_fail_closed(tmp_path):
    path = tmp_path / "labels_short"
    _fixture(path, mutate=lambda data, labels, names: labels.pop())
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())

    path = tmp_path / "filenames_short"
    _fixture(path, mutate=lambda data, labels, names: names.pop())
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())

    path = tmp_path / "wrong_order"
    data = [np.zeros((150, 150), dtype=np.uint8) for _ in range(EXPECTED_ROWS)]
    payload = pickle.dumps(
        {"labels": [0] * EXPECTED_ROWS, "data": data,
         "filenames": ["x.png"] * EXPECTED_ROWS}, protocol=3)
    path.write_bytes(payload)
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())

    path = tmp_path / "wrong_dtype"
    _fixture(path, mutate=lambda data, labels, names: data.__setitem__(
        0, np.zeros((150, 150), dtype=np.uint16)))
    with pytest.raises(StreamReaderError):
        read_member(path, "batches/data_batch_1", range(88), ())


def test_deterministic_hash_and_bounded_buffer(tmp_path):
    path = _fixture(tmp_path / "data_batch_1")
    instrumentation = ReaderInstrumentation()
    records = read_member(path, "batches/data_batch_1", {0}, set(range(1, 88)),
                          instrumentation=instrumentation)
    assert canonical_sha256(records[0].pixels) == canonical_sha256(records[0].pixels)
    assert instrumentation.maximum_pixel_buffer_size == 22500


def test_duplicate_hash_compatibility_and_conflicting_labels(tmp_path):
    def duplicate(data, labels, filenames):
        data[1] = data[0].copy()
        labels[1] = 1 - labels[0]
    path = _fixture(tmp_path / "data_batch_1", mutate=duplicate)
    records = read_member(path, "batches/data_batch_1", {0, 1},
                          set(range(2, 88)))
    assert canonical_sha256(records[0].pixels) == canonical_sha256(records[1].pixels)
    assert records[0].label != records[1].label


def test_no_eager_loader_or_ml_dependencies_are_needed(tmp_path):
    path = _fixture(tmp_path / "data_batch_1")
    records = read_member(path, "batches/data_batch_1", {0}, set(range(1, 88)))
    assert len(records) == 1


def test_bounded_archive_extraction_returns_exactly_eight_in_order(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    sha256, md5 = _archive_pins(archive)
    selected = [615, 0, 672, 119, 527, 176, 88, 671]
    instrumentation = ReaderInstrumentation()
    records = extract_selected_archive(
        archive, range(673), range(673, 792), selected,
        expected_sha256=sha256, expected_md5=md5,
        instrumentation=instrumentation,
    )
    assert [record.dataset_index for record in records] == sorted(selected)
    assert all(record.pixels.__class__ is bytes for record in records)
    assert all(len(record.pixels) == 150 * 150 for record in records)
    assert all(
        np.frombuffer(record.pixels, dtype=np.uint8).reshape(150, 150).shape == (150, 150)
        for record in records
    )
    assert all(np.frombuffer(record.pixels, dtype=np.uint8).dtype == np.uint8
               for record in records)
    assert all(record.label in (0, 1) for record in records)
    assert all(record.filename.endswith(".png") for record in records)
    assert instrumentation.permitted_payload_extractions == 8
    assert instrumentation.maximum_pixel_buffer_size == 22500
    assert instrumentation.forbidden_payload_encounters == 119
    assert instrumentation.forbidden_payload_extraction_attempts == 0
    assert instrumentation.forbidden_payloads_materialized == 0


def test_bounded_archive_extraction_accepts_empty_selection(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    sha256, md5 = _archive_pins(archive)
    instrumentation = ReaderInstrumentation()
    records = extract_selected_archive(
        archive, range(673), range(673, 792), [],
        expected_sha256=sha256, expected_md5=md5,
        instrumentation=instrumentation,
    )
    assert records == []
    assert instrumentation.permitted_payload_extractions == 0
    assert instrumentation.maximum_pixel_buffer_size == 0


@pytest.mark.parametrize(
    "selected",
    [
        list(range(9)),
        [0, 0],
        [-1],
        [792],
        [673],
    ],
)
def test_bounded_archive_extraction_rejects_invalid_selection(tmp_path, selected):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    sha256, md5 = _archive_pins(archive)
    with pytest.raises(StreamReaderError):
        extract_selected_archive(
            archive, range(673), range(673, 792), selected,
            expected_sha256=sha256, expected_md5=md5,
        )


def test_bounded_archive_extraction_preserves_hashable_pixels_and_budget(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    sha256, md5 = _archive_pins(archive)
    selected = list(range(8))
    records = extract_selected_archive(
        archive, range(673), range(673, 792), selected,
        expected_sha256=sha256, expected_md5=md5,
    )
    assert len(records) == 8
    assert sum(len(record.pixels) for record in records) == 180000
    assert canonical_sha256(records[0].pixels) == canonical_sha256(records[0].pixels)
    assert canonical_sha256(records[0].pixels) != canonical_sha256(records[1].pixels)


def test_metadata_only_tar_scan_validates_all_members_and_mixed_split(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    forbidden = frozenset(range(119))
    allowed = frozenset(range(119, 792))
    instrumentation = ReaderInstrumentation()
    structures = scan_archive(
        archive, allowed, forbidden, instrumentation=instrumentation,
        extract_pixels=False,
    )
    assert len(structures) == 9
    assert sum(item.payload_count for item in structures) == 792
    assert all(item.payload_lengths == (22500,) for item in structures)
    assert instrumentation.permitted_payload_extractions == 0
    assert instrumentation.forbidden_payload_encounters == 119
    assert instrumentation.forbidden_payload_extraction_attempts == 0
    assert instrumentation.forbidden_payloads_materialized == 0
    assert instrumentation.payload_boundaries == 792
    assert instrumentation.label_count == 792
    assert instrumentation.filename_count == 792
    assert instrumentation.payload_lengths == (22500,)
    assert instrumentation.allowed_row_count == 673
    assert instrumentation.forbidden_row_count == 119
    assert instrumentation.maximum_pixel_buffer_size == 0
    assert instrumentation.unexpected_parser_events == 0


def test_tar_scan_rejects_checksum_and_member_failures(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    with pytest.raises(StreamReaderError):
        scan_archive(archive, range(119, 792), range(119),
                     expected_sha256="0" * 64)
    broken = tmp_path / "broken.tar.gz"
    with tarfile.open(broken, "w:gz") as output:
        info = tarfile.TarInfo("batches/data_batch_1")
        info.size = 3
        output.addfile(info, __import__("io").BytesIO(b"bad"))
    with pytest.raises(StreamReaderError):
        scan_archive(broken, range(119, 792), range(119))


def test_tar_scan_rejects_manifest_checksum_and_duplicate_members(tmp_path):
    archive = tmp_path / "synthetic.tar.gz"
    _archive_fixture(archive)
    corrupt = tmp_path / "corrupt.tar.gz"
    with tarfile.open(archive, "r:gz") as source, tarfile.open(corrupt, "w:gz") as output:
        for member in source.getmembers():
            payload = source.extractfile(member).read() if member.isfile() else None
            if member.name == "batches/hash.md5":
                payload = payload.replace(b"  batches/", b"  batches/", 1)
                payload = b"0" * 32 + payload[32:]
            if member.isfile():
                output.addfile(member, __import__("io").BytesIO(payload))
    with pytest.raises(StreamReaderError):
        scan_archive(corrupt, range(119, 792), range(119))

    duplicate = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "r:gz") as source, tarfile.open(duplicate, "w:gz") as output:
        for member in source.getmembers():
            if member.isfile():
                output.addfile(member, __import__("io").BytesIO(
                    source.extractfile(member).read()))
        member = source.getmember("batches/batches.meta")
        output.addfile(member, __import__("io").BytesIO(
            source.extractfile(member).read()))
    with pytest.raises(StreamReaderError):
        scan_archive(duplicate, range(119, 792), range(119))


def test_split_validation_enforces_pinned_hash_and_counts(tmp_path):
    from src.data.mirabest_pickle_stream_reader import split_sets
    split = {
        "train": list(range(554)),
        "validation": list(range(554, 673)),
        "test": list(range(673, 792)),
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    allowed, forbidden = split_sets(path, expected)
    assert len(allowed) == 673
    assert len(forbidden) == 119
    with pytest.raises(StreamReaderError):
        split_sets(path, "0" * 64)


def test_split_validation_rejects_duplicate_list_entries_before_set_conversion(tmp_path):
    from src.data.mirabest_pickle_stream_reader import split_sets
    split = {
        "train": list(range(553)) + [0],
        "validation": list(range(554, 673)),
        "test": list(range(673, 792)),
    }
    path = tmp_path / "duplicate_split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(StreamReaderError):
        split_sets(path, expected)
