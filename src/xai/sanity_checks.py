"""Synthetic-testable parameter sensitivity core; no dataset or artifact I/O.

CPU execution is intentional: standard PyTorch CPU initialization with isolated
RNG state defines reproducibility. Retained learned BatchNorm buffers mean even
the final stage is NOT a wholly independently initialized network. Similarity
does not establish faithfulness, causality, or astronomical correctness.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.xai.gradcam import generate_gradcam
from src.xai.stability import mask_jaccard, topk_mask

SEEDS = (3000, 3001, 3002, 3003, 3004)
STAGES = ("classifier.4", "classifier.1", "blocks.2", "blocks.1", "blocks.0", "stem")


@dataclass(frozen=True)
class Similarity:
    jaccard: float | None
    spearman: float | None
    total_variation: float | None
    status: str
    reasons: tuple[str, ...] = ()


def _map(value, status, prefix):
    if status not in {"valid", "degenerate", "unavailable"}:
        raise ValueError("Unknown explanation status.")
    if value is None:
        return None, prefix + "_unavailable"
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (150, 150) or array.dtype.kind not in "fiu":
        raise ValueError("CAM must be a numeric 150x150 array.")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        return None, prefix + "_nonfinite"
    if (array < 0).any():
        raise ValueError("CAM must be nonnegative.")
    if status != "valid":
        return None, prefix + "_" + status
    tolerance = float(np.finfo(np.float32).eps) * max(float(array.max()), 1.0)
    if array.max() - array.min() <= tolerance:
        return None, prefix + "_degenerate"
    return array, None


def _average_ranks(values):
    """One-based average ranks, including all exact ties (no scipy dependency)."""
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(flat[order]) != 0) + 1]
    ends = np.r_[starts[1:], flat.size]
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


def compare_attributions(reference, candidate, dataset_index, *,
                         reference_status="valid", candidate_status="valid"):
    """Full-support top-10% Jaccard, average-tie Spearman and spatial TV.

    Use unnormalized nonnegative upsampled CAMs, never display-normalized maps.
    Invalid dimensions/negative values raise; unavailable explanations retain
    explicit reasons and have NO numerical similarity values.
    """
    if type(dataset_index) is not int or not 0 <= dataset_index < 792:
        raise ValueError("Invalid dataset index.")
    first, reason1 = _map(reference, reference_status, "reference")
    second, reason2 = _map(candidate, candidate_status, "candidate")
    reasons = tuple(r for r in (reason1, reason2) if r)
    if reasons:
        return Similarity(None, None, None, "unavailable", reasons)
    support = np.ones((150, 150), dtype=bool)
    # Reuse the frozen float32 pixel-ranking policy verbatim for model CAMs.
    # Scale only values exceeding float32 range to avoid conversion overflow.
    p, q = first / first.max(), second / second.max()
    float32_max = float(np.finfo(np.float32).max)
    rank_first = first if first.max() <= float32_max else p
    rank_second = second if second.max() <= float32_max else q
    jaccard = mask_jaccard(topk_mask(rank_first, support, dataset_index),
                           topk_mask(rank_second, support, dataset_index))
    r, s = _average_ranks(first), _average_ranks(second)
    r -= r.mean()
    s -= s.mean()
    rho = float(np.dot(r, s) / np.sqrt(np.dot(r, r) * np.dot(s, s)))
    # Scale before summing to avoid overflow for otherwise finite CAMs.
    tv = float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())
    return Similarity(jaccard, float(np.clip(rho, -1, 1)), tv, "valid")


def _snapshot(model):
    return (
        {n: t.detach().clone() for n, t in model.state_dict().items()},
        {n: (p.requires_grad, None if p.grad is None else p.grad.detach().clone())
         for n, p in model.named_parameters()},
        tuple(m.training for m in model.modules()),
    )


def _preserved(model, snapshot):
    state, params, modes = snapshot
    current = model.state_dict()
    return (current.keys() == state.keys()
            and all(torch.equal(current[n], t) for n, t in state.items())
            and tuple(m.training for m in model.modules()) == modes
            and all(p.requires_grad == params[n][0] and
                    ((p.grad is None and params[n][1] is None) or
                     (p.grad is not None and params[n][1] is not None
                      and torch.equal(p.grad, params[n][1])))
                    for n, p in model.named_parameters()))


def _bn_buffers(model):
    return {f"{name}.{key}": value.detach().clone()
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)}


def _validate_model(model):
    if any(m.training for m in model.modules()):
        raise ValueError("Every module must already be in evaluation mode.")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Sanity-check core requires a CPU model.")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Sanity checks require gradient-enabled execution.")
    for name in STAGES:
        group = model.get_submodule(name)
        leaves = [m for m in group.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        if not leaves:
            raise ValueError(f"Stage {name} has no Conv/Linear parameters.")


@dataclass(frozen=True)
class RandomizedStage:
    seed: int
    stage: int
    modules_randomized: tuple[str, ...]
    parameter_modules: tuple[str, ...]
    model: nn.Module
    batchnorm_buffers_preserved: bool


def iter_randomized_models(model, seed):
    """Yield independent stage snapshots of one cumulative seeded trajectory.

    RNG context exits BEFORE each yield, including early iterator abandonment.
    Consumers should discard each snapshot after scoring. No model is saved.
    """
    _validate_model(model)
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Seed must be one of 3000..3004.")
    original = _snapshot(model)
    expected_bn = _bn_buffers(model)
    copy = deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        local_rng = torch.get_rng_state().clone()
    randomized = []
    try:
        for stage, name in enumerate(STAGES, 1):
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.set_rng_state(local_rng)
                for suffix, module in copy.get_submodule(name).named_modules():
                    path = name + ("." + suffix if suffix else "")
                    if isinstance(module, nn.Conv2d):
                        nn.Conv2d.reset_parameters(module)
                    elif isinstance(module, nn.Linear):
                        nn.Linear.reset_parameters(module)
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        # NEVER call BN.reset_parameters/reset_running_stats.
                        if module.affine:
                            module.weight.fill_(1)
                            module.bias.zero_()
                    else:
                        continue
                    randomized.append(path)
                local_rng = torch.get_rng_state().clone()
            actual_bn = _bn_buffers(copy)
            buffers_ok = (actual_bn.keys() == expected_bn.keys() and
                          all(torch.equal(actual_bn[n], v) for n, v in expected_bn.items()))
            if not buffers_ok or any(m.training for m in copy.modules()):
                raise RuntimeError("Randomized-copy BatchNorm/evaluation state changed.")
            if not _preserved(model, original):
                raise RuntimeError("Original model state changed.")
            yield RandomizedStage(seed, stage, STAGES[:stage], tuple(randomized),
                                  deepcopy(copy), buffers_ok)
    finally:
        if not _preserved(model, original):
            raise RuntimeError("Original model state changed.")


@dataclass(frozen=True)
class SanityRecord:
    seed: int | None
    stage: int
    modules_randomized: tuple[str, ...]
    parameter_modules: tuple[str, ...]
    fixed_target_class: int
    similarity: Similarity
    original_state_preserved: bool
    batchnorm_buffers_preserved: bool
    evaluation_mode_preserved: bool
    scoring_state_preserved: bool


def _explain(model, tensor, target):
    try:
        return generate_gradcam(model, tensor, target), None
    except ValueError as error:
        # Numerical unavailability is recorded; programming/configuration errors
        # propagate rather than silently becoming scientific observations.
        if "non-finite" in str(error).lower():
            return None, "candidate_nonfinite"
        raise


def run_sanity_checks(model, input_tensor, dataset_index):
    """Return identity + 30 stage records for ONE caller-supplied prepared tensor.

    No dataset loading, preprocessing, OOD operations, or experiment runner.
    The original prediction is obtained on an unchanged disposable copy. A
    degenerate reference still records every stage as unavailable. A failed
    original forward raises because its target cannot safely be established.
    """
    _validate_model(model)
    if type(dataset_index) is not int or not 0 <= dataset_index < 792:
        raise ValueError("Invalid dataset index.")
    original = _snapshot(model)
    reference = generate_gradcam(deepcopy(model), input_tensor)
    target = reference.predicted_class
    records = []

    def score(copy, seed, stage, groups, leaves):
        before = _snapshot(copy)
        result, reason = _explain(copy, input_tensor, target)
        if reason:
            similarity = Similarity(None, None, None, "unavailable", (reason,))
        else:
            if result.target_class != target:
                raise RuntimeError("Grad-CAM changed the fixed target class.")
            similarity = compare_attributions(
                reference.upsampled_heatmap, result.upsampled_heatmap, dataset_index,
                reference_status=reference.status, candidate_status=result.status)
        source_ok = _preserved(model, original)
        copy_ok = _preserved(copy, before)
        bn = _bn_buffers(copy)
        source_bn = _bn_buffers(model)
        bn_ok = bn.keys() == source_bn.keys() and all(
            torch.equal(bn[n], v) for n, v in source_bn.items())
        eval_ok = not any(m.training for m in copy.modules())
        if not all((source_ok, copy_ok, bn_ok, eval_ok)):
            raise RuntimeError("Model state changed during sanity-check scoring.")
        records.append(SanityRecord(seed, stage, groups, leaves, target, similarity,
                                    source_ok, bn_ok, eval_ok, copy_ok))

    try:
        score(deepcopy(model), None, 0, (), ())
        for seed in SEEDS:
            for randomized in iter_randomized_models(model, seed):
                score(randomized.model, seed, randomized.stage,
                      randomized.modules_randomized, randomized.parameter_modules)
        return tuple(records)
    finally:
        if not _preserved(model, original):
            raise RuntimeError("Original model state changed.")
