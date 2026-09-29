"""Tests for Phase 2 preprocessing and reproducible splitting."""

from collections import Counter
from pathlib import Path

import pytest
import torch

from src.data.download_mirabest import MiraBestFull, load_mirabest
from src.data.split_dataset import build_preprocessed_splits, stratified_split_indices


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mirabest"


@pytest.fixture(scope="module")
def dataset():
    if not MiraBestFull.check_integrity(DATASET_ROOT):
        pytest.skip("MiraBest dataset is not available; tests do not trigger downloads.")
    return load_mirabest(DATASET_ROOT, download=False)


@pytest.fixture(scope="module")
def split_indices(dataset):
    labels = [dataset[index][1] for index in range(len(dataset))]
    return stratified_split_indices(labels, seed=42)


def test_all_samples_are_assigned_once(dataset, split_indices):
    split_values = list(split_indices.values())
    flattened = [index for indices in split_values for index in indices]
    assert len(flattened) == len(dataset)
    assert len(set(flattened)) == len(dataset)
    assert set(flattened) == set(range(len(dataset)))


def test_split_is_reproducible_with_same_seed(dataset, split_indices):
    labels = [dataset[index][1] for index in range(len(dataset))]
    assert split_indices == stratified_split_indices(labels, seed=42)


def test_class_proportions_are_preserved(dataset, split_indices):
    overall = Counter(dataset[index][1] for index in range(len(dataset)))
    overall_proportions = {label: count / len(dataset) for label, count in overall.items()}
    for indices in split_indices.values():
        counts = Counter(dataset[index][1] for index in indices)
        for label, overall_proportion in overall_proportions.items():
            split_proportion = counts[label] / len(indices)
            assert abs(split_proportion - overall_proportion) <= 0.02


def test_preprocessed_views_return_float_tensors_without_resizing(dataset, split_indices):
    subsets = build_preprocessed_splits(dataset, split_indices)
    for subset in subsets.values():
        image, label = subset[0]
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
        assert tuple(image.shape) == (1, 150, 150)
        assert label in {0, 1}