"""Isolated exact-duplicate audit utilities.

This script intentionally refuses ordinary MiraBest pickle batches.  It is
usable only with synthetic ``MBRAW1`` fixtures until a separately reviewed
real selective decoder is approved.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.data.selective_raw_reader import SelectiveRawReader, canonical_sha256

EXPECTED_SPLIT_SHA256 = "c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008"
EXPECTED_ARCHIVE_SHA256 = "505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8"
EXPECTED_CHECKPOINT_SHA256 = "d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(path: str | Path, expected_sha256: str = EXPECTED_SPLIT_SHA256):
    path = Path(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError("Frozen split SHA-256 mismatch.")
    split = json.loads(path.read_text(encoding="utf-8"))
    expected_counts = {"train": 554, "validation": 119, "test": 119}
    if set(split) != set(expected_counts):
        raise ValueError("Unexpected split keys.")
    sets = {}
    for name, count in expected_counts.items():
        values = split[name]
        if (not isinstance(values, list) or len(values) != count
                or any(type(index) is not int or not 0 <= index < 792 for index in values)
                or len(set(values)) != count):
            raise ValueError(f"Invalid {name} split.")
        sets[name] = frozenset(values)
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("Frozen splits overlap.")
    if set().union(*sets.values()) != set(range(792)):
        raise ValueError("Frozen splits do not cover indices 0..791.")
    return sets


def classify_duplicate_groups(records: Iterable[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record["canonical_sha256"]].append(record)
    result = []
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        splits = sorted({item["split"] for item in members})
        labels = sorted({item["label"] for item in members})
        result.append({
            "canonical_sha256": digest,
            "confirmed_byte_equal": all(item["confirmed_byte_equal"] for item in members),
            "dataset_indices": sorted(item["dataset_index"] for item in members),
            "splits": splits,
            "labels": labels,
            "cross_train_validation": set(splits) == {"train", "validation"},
            "conflicting_labels": len(labels) > 1,
            "members": sorted(members, key=lambda item: item["dataset_index"]),
        })
    return result


def audit_records(reader: SelectiveRawReader, split_sets: dict[str, frozenset]) -> dict:
    records = []
    for split in ("train", "validation"):
        for index in sorted(split_sets[split]):
            image = reader.read(index)
            records.append({
                "dataset_index": index,
                "split": split,
                "label": image.metadata.label,
                "filename": image.metadata.filename,
                "member": image.metadata.member,
                "member_row": image.metadata.member_row,
                "canonical_sha256": canonical_sha256(image.pixels),
                "confirmed_byte_equal": True,
            })
    groups = classify_duplicate_groups(records)
    unique_hashes = len({item["canonical_sha256"] for item in records})
    return {
        "authorized_scope": "TRAIN + VALIDATION",
        "authorized_images_processed": len(records),
        "forbidden_scope": "TEST",
        "test_images_decoded": reader.test_payload_reads,
        "unique_image_hashes": unique_hashes,
        "duplicate_groups": groups,
        "duplicate_group_count": len(groups),
        "affected_unique_images": len({i for group in groups for i in group["dataset_indices"]}),
        "cross_split_duplicate_group_count": sum(group["cross_train_validation"] for group in groups),
        "conflicting_label_group_count": sum(group["conflicting_labels"] for group in groups),
    }


def refuse_existing_mirabest_loader() -> None:
    raise RuntimeError(
        "Real MiraBest pickle batches are not supported by this isolated safety harness."
    )


def main() -> int:
    refuse_existing_mirabest_loader()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
