import json

import pytest

from scripts.run_duplicate_audit_train_val import (
    audit_records,
    classify_duplicate_groups,
    validate_split,
)
from src.data.selective_raw_reader import MAGIC, SelectiveRawReader

from test_selective_raw_reader import reader_fixture, write_batch


def test_duplicate_groups_classify_cross_split_and_label_conflict():
    records = [
        {"dataset_index": 0, "split": "train", "label": 0,
         "canonical_sha256": "a", "confirmed_byte_equal": True},
        {"dataset_index": 1, "split": "validation", "label": 1,
         "canonical_sha256": "a", "confirmed_byte_equal": True},
        {"dataset_index": 2, "split": "train", "label": 1,
         "canonical_sha256": "b", "confirmed_byte_equal": True},
        {"dataset_index": 3, "split": "train", "label": 1,
         "canonical_sha256": "b", "confirmed_byte_equal": True},
    ]
    groups = classify_duplicate_groups(records)
    assert len(groups) == 2
    assert groups[0]["cross_train_validation"] is True
    assert groups[0]["conflicting_labels"] is True
    assert groups[1]["cross_train_validation"] is False


def test_audit_counts_only_allowed_scope(tmp_path):
    path, pixels_a, pixels_b = reader_fixture(tmp_path)
    reader = SelectiveRawReader({"data_batch_1": path}, {0, 1}, {2})
    result = audit_records(reader, {"train": frozenset({0}), "validation": frozenset({1})})
    assert result["authorized_images_processed"] == 2
    assert result["test_images_decoded"] == 0
    assert result["duplicate_group_count"] == 1
    assert result["cross_split_duplicate_group_count"] == 1
    assert result["conflicting_label_group_count"] == 1


def test_split_hash_and_structure_are_fail_closed(tmp_path):
    split = {"train": list(range(554)), "validation": list(range(554, 673)),
             "test": list(range(673, 792))}
    path = tmp_path / "split_indices.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    import hashlib
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    sets = validate_split(path, expected)
    assert len(sets["train"] | sets["validation"]) == 673
    assert len(sets["test"]) == 119


def test_existing_output_namespace_cannot_be_overwritten(tmp_path):
    output = tmp_path / "duplicate_audit_train_val_v1"
    output.mkdir()
    with pytest.raises(FileExistsError):
        output.mkdir(exist_ok=False)
