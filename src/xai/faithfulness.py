"""Phase3B-XAI-v1.0.0 primary AOPC and secondary full-deletion core.

No dataset/checkpoint loading, Grad-CAM generation, OOD gating, file writes,
statistics or cohort evaluation occurs here. Intensities are floating arrays in
[0,1], not normalized classifier tensors. Dataset access must remain behind the
Phase 3B.1 allowlist interface in any future orchestrator. Callbacks must be pure;
use ModelScorer (or pass an nn.Module) to isolate a supplied model's state.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.preprocess import PreprocessConfig, build_transforms

PROTOCOL = "Phase3B-XAI-v1.0.0"
FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
# Integer arithmetic avoids floating-point budget underflow at decimal fractions.
BUDGETS = tuple(22500 * k // 20 for k in range(1, 7))
RANDOM_SEEDS = tuple(range(1000, 1020))
REPLACEMENTS = ("border_median", "black", "gaussian")
FULL_FRACTIONS = tuple(k / 20 for k in range(21))
FULL_BUDGETS = tuple(22500 * k // 20 for k in range(21))


def _index(value):
    if type(value) is not int or not 0 <= value < 792:
        raise ValueError("dataset_index must be an integer in [0,791].")
    return value


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fi":
        raise ValueError(f"{name} must be a real numeric array with shape {shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")
    return array


def _intensity(image):
    array = _numeric(image, (150, 150), "Intensity image")
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Intensity image must be in [0,1], before normalization.")
    return array.astype(np.float32, copy=True)


def tile_pixels() -> tuple[np.ndarray, ...]:
    """361 disjoint, row-major flattened pixel lists covering all 22500 pixels."""
    boundaries = tuple(150 * j // 19 for j in range(20))
    grid = np.arange(22500).reshape(150, 150)
    return tuple(grid[boundaries[r]:boundaries[r+1], boundaries[c]:boundaries[c+1]].ravel().copy()
                 for r in range(19) for c in range(19))


def _rank_values(values, dataset_index):
    _index(dataset_index)
    values = _numeric(values, (361,), "Tile scores")
    return tuple(sorted(range(361), key=lambda t: (
        -float(values[t]), hashlib.sha256(
            f"{PROTOCOL}|tile-tie|42|{dataset_index}|{t}".encode("ascii")).hexdigest(), t)))


def rank_gradcam(raw_cam, dataset_index: int, *, status="valid") -> tuple[int, ...]:
    """Rank raw native channels' spatial CAM values, rejecting core-degenerate maps."""
    cam = _numeric(raw_cam, (19, 19), "Raw CAM")
    if status not in {"valid", "degenerate"}:
        raise ValueError("CAM status must be valid or degenerate.")
    if np.any(cam < 0):
        raise ValueError("Raw CAM must be nonnegative.")
    if status == "degenerate":
        raise ValueError("Degenerate Grad-CAM cannot define a deletion ranking.")
    # Match the unchanged core's float32 display-span criterion. This is only
    # a degeneracy check; ranking below still uses native values, not interpolation.
    with np.errstate(over="ignore"):
        native = cam.astype(np.float32, copy=True)
    if not np.isfinite(native).all():
        raise ValueError("Raw CAM cannot be represented as finite float32 values.")
    display = F.interpolate(torch.from_numpy(native)[None, None], size=(150, 150),
                            mode="bilinear", align_corners=False)[0, 0]
    tolerance = torch.finfo(display.dtype).eps * display.max().abs().clamp_min(1.0)
    if not torch.isfinite(display).all() or display.max() - display.min() <= tolerance:
        raise ValueError("Degenerate Grad-CAM cannot define a deletion ranking.")
    return _rank_values(cam.ravel(), dataset_index)


def rank_brightness(image, dataset_index: int) -> tuple[int, ...]:
    """Descending mean original intensity for exactly the same coordinate tiles."""
    flat = _intensity(image).ravel()
    means = np.array([flat[pixels].mean(dtype=np.float64) for pixels in tile_pixels()])
    return _rank_values(means, dataset_index)


def random_rankings(dataset_index: int) -> dict[int, tuple[int, ...]]:
    _index(dataset_index)
    return {seed: tuple(int(t) for t in np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([seed, dataset_index]))).permutation(361)) for seed in RANDOM_SEEDS}


def _permutation(ranking):
    values = tuple(ranking)
    if (len(values) != 361 or any(type(t) not in (int, np.int64, np.int32) for t in values)
            or set(values) != set(range(361))):
        raise ValueError("Ranking must be a complete permutation of tile IDs 0..360.")
    return values


def deletion_masks(ranking) -> np.ndarray:
    ranking = _permutation(ranking)
    pixels = tile_pixels()
    ordered = np.concatenate([pixels[t] for t in ranking])
    masks = np.zeros((6, 22500), dtype=bool)
    for mask, budget in zip(masks, BUDGETS):
        mask[ordered[:budget]] = True
    return masks.reshape(6, 150, 150)


def replacement_image(image, policy: str) -> np.ndarray:
    """Border union counts each corner once; Gaussian is discrete separable reflect."""
    image = _intensity(image)
    if policy == "black":
        return np.zeros_like(image)
    if policy == "border_median":
        border = np.ones((150, 150), dtype=bool)
        border[5:-5, 5:-5] = False
        return np.full_like(image, np.median(image[border]))
    if policy != "gaussian":
        raise ValueError(f"Unknown replacement policy: {policy}")
    positions = np.arange(-9, 10, dtype=np.float64)
    kernel = np.exp(-(positions ** 2) / (2 * 3 ** 2))
    kernel /= kernel.sum()
    padded = np.pad(image.astype(np.float64), ((0, 0), (9, 9)), mode="reflect")
    horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    padded = np.pad(horizontal, ((9, 9), (0, 0)), mode="reflect")
    blurred = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)
    return blurred.astype(np.float32)


def perturb_image(image, mask, replacement) -> np.ndarray:
    image = _intensity(image)
    replacement = _intensity(replacement)
    mask = np.asarray(mask)
    if mask.shape != (150, 150) or mask.dtype != np.bool_:
        raise ValueError("Deletion mask must be boolean with shape (150,150).")
    return np.where(mask, replacement, image).astype(np.float32)


def normalize_intensity(image) -> torch.Tensor:
    """Reuse existing deterministic validation ToTensor/Normalize, unchanged."""
    _, transform = build_transforms(PreprocessConfig())
    return transform(_intensity(image)).unsqueeze(0)


def _logits(values) -> tuple[float, float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.shape == (1, 2):
        values = values[0]
    checked = _numeric(values, (2,), "Binary logits").astype(np.float64)
    return float(checked[0]), float(checked[1])


def _target(target):
    if type(target) is not int or target not in (0, 1):
        raise ValueError("Target class must be integer 0 (FRI) or 1 (FRII).")
    return target


@dataclass(frozen=True)
class Scores:
    probability: float
    logit: float
    margin: float


def target_scores(logits, target_class: int) -> Scores:
    target = _target(target_class)
    z = np.array(_logits(logits), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        exponentials = np.exp(z - z.max())
        margin = z[target] - z[1 - target]
    values = Scores(float(exponentials[target] / exponentials.sum()), float(z[target]), float(margin))
    if not np.isfinite(tuple(values.__dict__.values())).all():
        raise ValueError("Non-finite derived score.")
    return values


@dataclass(frozen=True)
class DeletionCurve:
    target_class: int
    original_logits: tuple[float, float]
    deleted_logits: tuple[tuple[float, float], ...]
    fractions: tuple[float, ...] = FRACTIONS


def aopc(curve: DeletionCurve) -> Scores:
    """Mean original-minus-deleted score over exactly the six fixed fractions."""
    target = _target(curve.target_class)
    original_logits = _logits(curve.original_logits)
    if target != int(np.argmax(original_logits)):
        raise ValueError("Curve target must be the original predicted class.")
    if tuple(curve.fractions) != FRACTIONS or len(curve.deleted_logits) != 6:
        raise ValueError("Incomplete/invalid deletion curve: require all six fixed fractions.")
    original = target_scores(original_logits, target)
    deleted = [target_scores(row, target) for row in curve.deleted_logits]
    result = Scores(*(float(np.mean([getattr(original, field) - getattr(row, field) for row in deleted]))
                      for field in ("probability", "logit", "margin")))
    if not np.isfinite(tuple(result.__dict__.values())).all():
        raise ValueError("Non-finite AOPC.")
    return result


@dataclass(frozen=True)
class Contrasts:
    random_mean: Scores
    cam_minus_random: Scores
    cam_minus_brightness: Scores


def paired_contrasts(cam: Scores, brightness: Scores, random: Mapping[int, Scores]) -> Contrasts:
    if (set(random) != set(RANDOM_SEEDS) or len(random) != 20
            or any(type(seed) is not int for seed in random)):
        raise ValueError("Require exactly the 20 random controls keyed by seeds 1000..1019.")
    for score in (cam, brightness, *random.values()):
        if not isinstance(score, Scores) or not np.isfinite((score.probability, score.logit, score.margin)).all():
            raise ValueError("All paired AOPC scores must be finite.")
    mean = Scores(*(float(np.mean([getattr(random[s], field) for s in RANDOM_SEEDS]))
                    for field in ("probability", "logit", "margin")))
    def difference(other):
        return Scores(*(getattr(cam, f) - getattr(other, f) for f in ("probability", "logit", "margin")))
    result = Contrasts(mean, difference(mean), difference(brightness))
    if not all(np.isfinite((s.probability, s.logit, s.margin)).all()
               for s in (result.random_mean, result.cam_minus_random, result.cam_minus_brightness)):
        raise ValueError("Non-finite paired contrast.")
    return result


@dataclass(frozen=True)
class Rankings:
    dataset_index: int
    gradcam: tuple[int, ...]
    brightness: tuple[int, ...]
    random: tuple[tuple[int, tuple[int, ...]], ...]


def prepare_rankings(image, raw_cam, dataset_index, *, cam_status="valid") -> Rankings:
    """Compute all rankings once from original arrays; reuse for all replacements."""
    return Rankings(_index(dataset_index), rank_gradcam(raw_cam, dataset_index, status=cam_status),
                    rank_brightness(image, dataset_index), tuple(random_rankings(dataset_index).items()))


class ModelScorer:
    """Evaluate a disposable copy, preserving the supplied model including gradients.

    Requires all source modules in evaluation mode. The copy preserves weights,
    buffers and architecture; no .eval(), .train(), optimizer or backward is called.
    """

    def __init__(self, model: nn.Module):
        if any(module.training for module in model.modules()):
            raise ValueError("ModelScorer requires evaluation mode in every module.")
        self._model = deepcopy(model)
        tensor = next(iter(self._model.parameters()), None)
        if tensor is None:
            tensor = next(iter(self._model.buffers()), torch.empty(0))
        self._device = tensor.device

    def __call__(self, normalized: torch.Tensor):
        with torch.no_grad():
            return self._model(normalized.detach().clone().to(self._device)).detach().cpu().clone()


@dataclass(frozen=True)
class FaithfulnessResult:
    rankings: Rankings
    replacement: str
    gradcam: DeletionCurve
    brightness: DeletionCurve
    random: tuple[tuple[int, DeletionCurve], ...]
    contrasts: Contrasts


def evaluate_rankings(image, rankings: Rankings, scorer: Callable | nn.Module,
                      *, replacement="border_median") -> FaithfulnessResult:
    """Single supplied-array calculation only; no ranking recomputation or cohort run.

    Returns raw logits for audit. A changed perturbed prediction never changes the
    original target. The scorer is called once for the original and 22*6 times
    for deletions. Callers can reuse Rankings with another replacement policy.
    """
    image = _intensity(image)
    _index(rankings.dataset_index)
    if tuple(seed for seed, _ in rankings.random) != RANDOM_SEEDS:
        raise ValueError("Rankings require exactly seeds 1000..1019 in order.")
    orders = (rankings.gradcam, rankings.brightness, *(order for _, order in rankings.random))
    for order in orders:
        _permutation(order)
    replaced = replacement_image(image, replacement)
    if isinstance(scorer, nn.Module):
        scorer = ModelScorer(scorer)

    def score(intensity):
        # Fresh tensors isolate original arrays even from a mutating callback.
        with torch.no_grad():
            return _logits(scorer(normalize_intensity(intensity)))

    original = score(image)
    target = int(np.argmax(original))

    def curve(order):
        deleted = tuple(score(perturb_image(image, mask, replaced)) for mask in deletion_masks(order))
        return DeletionCurve(target, original, deleted)

    cam, bright, *random_curves = (curve(order) for order in orders)
    random = tuple(zip(RANDOM_SEEDS, random_curves))
    contrasts = paired_contrasts(aopc(cam), aopc(bright), {s: aopc(c) for s, c in random})
    return FaithfulnessResult(rankings, replacement, cam, bright, random, contrasts)


def full_deletion_masks(ranking) -> np.ndarray:
    """Separate 21-point grid: zero pixels at 0%, every pixel at 100%.

    Integer budgets implement floor(22500*k/20) without decimal underflow.
    Tile and within-tile order are identical to the primary six-step masks.
    """
    ranking = _permutation(ranking)
    pixels = tile_pixels()
    ordered = np.concatenate([pixels[t] for t in ranking])
    masks = np.zeros((21, 22500), dtype=bool)
    for mask, budget in zip(masks, FULL_BUDGETS):
        mask[ordered[:budget]] = True
    return masks.reshape(21, 150, 150)


@dataclass(frozen=True)
class FullDeletionCurve:
    """Full logits INCLUDING the original 0% observation, not just deletions."""

    target_class: int
    original_logits: tuple[float, float]
    logits: tuple[tuple[float, float], ...]
    fractions: tuple[float, ...] = FULL_FRACTIONS


def full_curve_scores(curve: FullDeletionCurve) -> tuple[Scores, ...]:
    """Validate all endpoints and score one fixed original class, even on flips."""
    if not isinstance(curve, FullDeletionCurve):
        raise ValueError("Expected a FullDeletionCurve.")
    target = _target(curve.target_class)
    original = _logits(curve.original_logits)
    if target != int(np.argmax(original)):
        raise ValueError("Full curve target must be the original predicted class.")
    if tuple(curve.fractions) != FULL_FRACTIONS or len(curve.logits) != 21:
        raise ValueError("Incomplete/invalid full deletion curve: require all 21 fixed fractions.")
    logits = tuple(_logits(row) for row in curve.logits)
    if logits[0] != original:
        raise ValueError("Full curve 0% logits must equal original logits exactly.")
    return tuple(target_scores(row, target) for row in logits)


def full_curve_effects(curve: FullDeletionCurve) -> tuple[Scores, ...]:
    """Signed original-minus-deleted effects, including exactly zero at 0%.

    These diagnostics are NOT the ordinate used for the frozen deletion AUC.
    Negative effects indicate increased original-class scores and are retained.
    """
    scores = full_curve_scores(curve)
    original = scores[0]
    effects = tuple(Scores(*(getattr(original, field)-getattr(row, field)
                            for field in ("probability", "logit", "margin"))) for row in scores)
    if not np.isfinite([(s.probability, s.logit, s.margin) for s in effects]).all():
        raise ValueError("Non-finite full-curve deletion effects.")
    return effects


def deletion_auc(curve: FullDeletionCurve) -> Scores:
    """Frozen YAML: trapezoidal integral of SCORE over all deletion fractions.

    Sum k=1..20: (score[k-1]+score[k])/2 * (f[k]-f[k-1]). No baseline
    subtraction, endpoint omission, clipping, extrapolation or extra scaling.
    Probability is the original-class score; logit and margin are diagnostics.
    """
    scores = full_curve_scores(curve)
    values = np.array([(s.probability, s.logit, s.margin) for s in scores], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        # Halve before addition to avoid overflow in finite endpoint averages.
        areas = ((values[:-1] / 2 + values[1:] / 2) * np.diff(FULL_FRACTIONS)[:, None]).sum(axis=0)
    if not np.isfinite(areas).all():
        raise ValueError("Non-finite deletion AUC.")
    return Scores(*(float(value) for value in areas))


@dataclass(frozen=True)
class FullFaithfulnessResult:
    rankings: Rankings
    replacement: str
    gradcam: FullDeletionCurve
    brightness: FullDeletionCurve
    random: tuple[tuple[int, FullDeletionCurve], ...]
    gradcam_auc: Scores
    brightness_auc: Scores
    random_auc: tuple[tuple[int, Scores], ...]


def evaluate_full_rankings(image, rankings: Rankings, scorer: Callable | nn.Module,
                           *, replacement="border_median") -> FullFaithfulnessResult:
    """Secondary-only full curves; reuse original rankings across replacements.

    One original forward plus 22*20 deletion forwards. Reuse original logits for
    every 0% endpoint. At 100%, every pixel is the existing replacement image;
    no synthetic zero logit/probability is imposed. Any invalid callback output
    aborts without returning a partial result. A supplied nn.Module is copied.
    No primary AOPC API or mathematical definition is changed.
    """
    image = _intensity(image)
    if not isinstance(rankings, Rankings):
        raise ValueError("Expected validated Rankings from prepare_rankings.")
    _index(rankings.dataset_index)
    if tuple(seed for seed, _ in rankings.random) != RANDOM_SEEDS:
        raise ValueError("Rankings require exactly seeds 1000..1019 in order.")
    orders = (rankings.gradcam, rankings.brightness, *(order for _, order in rankings.random))
    for order in orders:
        _permutation(order)
    replaced = replacement_image(image, replacement)
    if isinstance(scorer, nn.Module):
        scorer = ModelScorer(scorer)

    def score(intensity):
        with torch.no_grad():
            return _logits(scorer(normalize_intensity(intensity)))

    original = score(image)
    target = int(np.argmax(original))

    def curve(order):
        deleted = tuple(score(perturb_image(image, mask, replaced))
                        for mask in full_deletion_masks(order)[1:])
        return FullDeletionCurve(target, original, (original, *deleted))

    cam, bright, *random_curves = (curve(order) for order in orders)
    random = tuple(zip(RANDOM_SEEDS, random_curves))
    return FullFaithfulnessResult(rankings, replacement, cam, bright, random,
                                  deletion_auc(cam), deletion_auc(bright),
                                  tuple((seed, deletion_auc(c)) for seed, c in random))
