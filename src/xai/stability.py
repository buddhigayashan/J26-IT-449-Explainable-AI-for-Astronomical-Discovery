"""Frozen Phase 3B stability/robustness core for supplied synthetic/image arrays.

No datasets, checkpoints, files, OOD gates, statistics, frontend or evaluation
runner are accessed. Future data access must use the Phase 3B.1 allowlists.
150x150 comparisons are implemented here; no native-grid pooling convention is
invented for the separately specified secondary 19x19 analysis.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math

import numpy as np
import torch
from torch import nn

from src.xai.faithfulness import normalize_intensity, replacement_image, target_scores
from src.xai.gradcam import TARGET_LAYER, GradCAMResult, generate_gradcam

PROTOCOL = "Phase3B-XAI-v1.0.0"
ROTATIONS = (-5, -2, 2, 5)
SENSITIVITY_ROTATIONS = (-15, 15)
NOISE_SIGMAS = (0.005, 0.01)
NOISE_SEEDS = tuple(range(2000, 2010))
BRIGHTNESS_GAINS = (0.98, 1.02)
TOP_FRACTIONS = (0.10, 0.05, 0.20)  # Primary first.


def _index(index):
    if type(index) is not int or not 0 <= index < 792:
        raise ValueError("dataset_index must be an integer in [0,791].")


def _array(value, name, *, finite=True):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (150, 150) or array.dtype.kind not in "fi":
        raise ValueError(f"{name} must be a real numeric 150x150 array.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32, copy=True)
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float32 values.")
    return array


def _intensity(image):
    image = _array(image, "Intensity image")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Intensity image must be in [0,1], before normalization.")
    return image


def _support(value):
    array = np.asarray(value)
    if array.shape != (150, 150) or array.dtype != np.bool_:
        raise ValueError("Support must be a boolean 150x150 mask.")
    return array.copy()


@dataclass(frozen=True)
class TransformSpec:
    kind: str
    value: float | None = None
    seed: int | None = None

    def __post_init__(self):
        if not isinstance(self.kind, str):
            raise ValueError("Transformation kind must be a protocol name.")
        if self.kind in {"identity", "horizontal_flip", "vertical_flip"}:
            valid = self.value is None and self.seed is None
        elif self.kind == "rotation":
            valid = (type(self.value) in (int, float) and self.value in ROTATIONS + SENSITIVITY_ROTATIONS
                     and self.seed is None)
        elif self.kind == "noise":
            valid = (type(self.value) in (int, float) and self.value in NOISE_SIGMAS
                     and type(self.seed) is int and self.seed in NOISE_SEEDS)
        elif self.kind == "brightness":
            valid = type(self.value) in (int, float) and self.value in BRIGHTNESS_GAINS and self.seed is None
        else:
            valid = False
        if not valid:
            raise ValueError("Transformation is outside the frozen protocol.")

    @property
    def secondary(self):
        return self.kind == "brightness" or (self.kind == "rotation" and self.value in SENSITIVITY_ROTATIONS)


def protocol_transformations() -> tuple[TransformSpec, ...]:
    """Exactly 30 transformations, no cumulative composition or identity replicate."""
    return (tuple(TransformSpec("rotation", a) for a in ROTATIONS + SENSITIVITY_ROTATIONS)
            + (TransformSpec("horizontal_flip"), TransformSpec("vertical_flip"))
            + tuple(TransformSpec("noise", sigma, seed) for sigma in NOISE_SIGMAS for seed in NOISE_SEEDS)
            + tuple(TransformSpec("brightness", gain) for gain in BRIGHTNESS_GAINS))


def _rotate(array, angle, fill, source_support):
    """Pixel-center bilinear rotation, positive angles counterclockwise.

    Coordinates are centered at (74.5,74.5). Every nonzero interpolation weight
    must come from an in-bounds, valid source pixel. Zero-weight neighbors do not
    invalidate exact pixel-center samples. No nearest-neighbor validity warp.
    """
    y, x = np.mgrid[:150, :150].astype(np.float64)
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    sx = cosine * (x - 74.5) - sine * (y - 74.5) + 74.5
    sy = sine * (x - 74.5) + cosine * (y - 74.5) + 74.5
    # Remove trigonometric roundoff at exact integer locations, not genuine fractions.
    sx = np.where(np.abs(sx - np.rint(sx)) < 1e-12, np.rint(sx), sx)
    sy = np.where(np.abs(sy - np.rint(sy)) < 1e-12, np.rint(sy), sy)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    dx, dy = sx - x0, sy - y0
    result = np.zeros((150, 150), dtype=np.float64)
    valid = np.ones((150, 150), dtype=bool)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                            (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        xx, yy = x0 + ox, y0 + oy
        inside = (xx >= 0) & (xx < 150) & (yy >= 0) & (yy < 150)
        cx, cy = xx.clip(0, 149), yy.clip(0, 149)
        values = np.where(inside, array[cy, cx], fill)
        result += weight * values
        valid &= (weight == 0) | (inside & source_support[cy, cx])
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite interpolated map.")
    return result, valid


@dataclass(frozen=True)
class TransformedImage:
    dataset_index: int
    spec: TransformSpec
    image: np.ndarray
    valid_support: np.ndarray
    clipped_fraction: float


def _validate_transformation(transformation):
    if not isinstance(transformation, TransformedImage) or not isinstance(transformation.spec, TransformSpec):
        raise ValueError("Expected a TransformedImage with a frozen TransformSpec.")
    _index(transformation.dataset_index)
    if (type(transformation.clipped_fraction) not in (int, float)
            or not math.isfinite(transformation.clipped_fraction)
            or not 0 <= transformation.clipped_fraction <= 1):
        raise ValueError("Clipped fraction must be finite and in [0,1].")


def apply_transform(image, spec: TransformSpec, dataset_index: int) -> TransformedImage:
    _index(dataset_index)
    image = _intensity(image)
    if not isinstance(spec, TransformSpec):
        raise ValueError("Expected a frozen TransformSpec.")
    support = np.ones((150, 150), dtype=bool)
    clipped = 0.0
    if spec.kind == "rotation":
        fill = float(replacement_image(image, "border_median")[0, 0])
        result, support = _rotate(image, spec.value, fill, support)
    elif spec.kind == "horizontal_flip":
        result = np.fliplr(image).copy()
    elif spec.kind == "vertical_flip":
        result = np.flipud(image).copy()
    elif spec.kind in {"noise", "brightness"}:
        if spec.kind == "noise":
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([spec.seed, dataset_index])))
            unbounded = image.astype(np.float64) + rng.normal(0, spec.value, image.shape)
        else:
            unbounded = image.astype(np.float64) * spec.value
        clipped = float(np.mean((unbounded < 0) | (unbounded > 1)))
        result = unbounded.clip(0, 1).astype(np.float32)
    else:
        result = image.copy()  # Identity is a verification control only.
    return TransformedImage(dataset_index, spec, result, support, clipped)


def apply_independent_transformations(image, dataset_index: int) -> tuple[TransformedImage, ...]:
    """Array helper only: every transform receives the same original intensities."""
    original = _intensity(image)
    return tuple(apply_transform(original, spec, dataset_index) for spec in protocol_transformations())


def inverse_align(continuous_cam, transformation: TransformedImage) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous aligned CAM and complete two-warp valid support."""
    _validate_transformation(transformation)
    cam = _array(continuous_cam, "Continuous CAM")
    support = _support(transformation.valid_support)
    spec = transformation.spec
    if spec.kind == "rotation":
        return _rotate(cam, -spec.value, 0.0, support)
    if spec.kind == "horizontal_flip":
        return np.fliplr(cam).copy(), np.fliplr(support).copy()
    if spec.kind == "vertical_flip":
        return np.flipud(cam).copy(), np.flipud(support).copy()
    return cam.copy(), support


@lru_cache(maxsize=8)
def _pixel_ties(dataset_index):
    _index(dataset_index)
    return tuple(hashlib.sha256(
        f"{PROTOCOL}|pixel-tie|42|{dataset_index}|{pixel_id}".encode("ascii")).digest()
                 for pixel_id in range(22500))


def _cam_status(cam, requested_status, support):
    if requested_status not in {"valid", "degenerate"}:
        raise ValueError("CAM status must be valid or degenerate.")
    if not np.isfinite(cam).all():
        return "nonfinite"
    if np.any(cam < 0):
        raise ValueError("CAM must be nonnegative.")
    if requested_status == "degenerate":
        return "degenerate"
    values = cam[support]
    if not len(values):
        return "degenerate"
    tolerance = np.finfo(np.float32).eps * max(abs(float(values.max())), 1.0)
    return "degenerate" if float(values.max()) - float(values.min()) <= tolerance else "valid"


def _pixel_order(cam, support, dataset_index):
    ties = _pixel_ties(dataset_index)
    flat = cam.ravel()
    return tuple(sorted(np.flatnonzero(support), key=lambda i: (-float(flat[i]), ties[i], int(i))))


def _mask_at_fraction(cam, order, fraction):
    if type(fraction) not in (int, float) or fraction not in TOP_FRACTIONS:
        raise ValueError("Top fraction must be 0.05, 0.10 or 0.20.")
    percent = {0.05: 5, 0.10: 10, 0.20: 20}[fraction]
    count = (percent * len(order) + 99) // 100
    mask = np.zeros(22500, dtype=bool)
    mask[list(order[:count])] = True
    cutoff = cam.ravel()[order[count - 1]]
    boundary_count = int(np.count_nonzero(cam.ravel()[list(order)] == cutoff))
    return mask.reshape(150, 150), boundary_count if boundary_count > 1 else 0


def topk_mask(continuous_cam, support, dataset_index, fraction=0.10) -> np.ndarray:
    _index(dataset_index)
    cam = _array(continuous_cam, "CAM")
    support = _support(support)
    if support.sum() < 18000:
        raise ValueError("Unavailable: less than 80% valid support.")
    if _cam_status(cam, "valid", support) != "valid":
        raise ValueError("Degenerate CAM cannot define a top-k attribution mask.")
    return _mask_at_fraction(cam, _pixel_order(cam, support, dataset_index), fraction)[0]


def mask_jaccard(first, second) -> float | None:
    first, second = _support(first), _support(second)
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else None


@dataclass(frozen=True)
class JaccardResult:
    fraction: float
    similarity: float | None
    selected_pixels_per_map: int | None
    original_boundary_ties: int | None
    transformed_boundary_ties: int | None


@dataclass(frozen=True)
class StabilityResult:
    dataset_index: int
    transformation: TransformSpec
    original_predicted_class: int
    transformed_predicted_class: int
    target_class: int
    class_flip: bool
    case_group: str
    original_class_probability_change: float
    original_class_margin_change: float
    jaccard: tuple[JaccardResult, ...]
    valid_support_fraction: float
    clipped_fraction: float
    original_cam_status: str
    transformed_cam_status: str
    status: str
    reason: str


def _prediction(explanation):
    logits = explanation.logits
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)
    if logits.shape != (1, 2) or logits.dtype.kind not in "fi" or not np.isfinite(logits).all():
        raise ValueError("Explanation requires finite binary logits with shape (1,2).")
    predicted = int(logits.argmax(axis=1)[0])
    if type(explanation.predicted_class) is not int or explanation.predicted_class != predicted:
        raise ValueError("Explanation predicted class disagrees with its logits.")
    return predicted, logits


def compare_explanations(original: GradCAMResult, transformed: GradCAMResult,
                         transformation: TransformedImage, dataset_index: int) -> StabilityResult:
    """Compare supplied CAMs; fixed-target provenance is checked, not inferred anew.

    Degenerate/nonfinite maps return unavailable Jaccards with model diagnostics
    retained. Invalid shapes, targets or negative maps raise explicit errors.
    """
    _index(dataset_index)
    _validate_transformation(transformation)
    if transformation.dataset_index != dataset_index:
        raise ValueError("Transformation belongs to a different dataset index.")
    if original.layer_name != TARGET_LAYER or transformed.layer_name != TARGET_LAYER:
        raise ValueError("Both explanations must use the frozen Grad-CAM target layer.")
    original_class, original_logits = _prediction(original)
    transformed_class, transformed_logits = _prediction(transformed)
    if (type(original.target_class) is not int or type(transformed.target_class) is not int
            or original.target_class != original_class or transformed.target_class != original_class):
        raise ValueError("Both CAM targets must be the fixed original predicted class.")
    first = _array(original.upsampled_heatmap, "Original CAM", finite=False)
    second = _array(transformed.upsampled_heatmap, "Transformed CAM", finite=False)
    ones = np.ones((150, 150), dtype=bool)
    first_status = _cam_status(first, original.status, ones)
    second_status = _cam_status(second, transformed.status, ones)
    # Geometry remains meaningful even when transformed attribution is nonfinite.
    aligned, common = inverse_align(second if second_status != "nonfinite" else np.zeros_like(second), transformation)
    valid_fraction = float(common.mean())
    reasons = []
    if valid_fraction < 0.8:
        reasons.append("insufficient_valid_support")
    if first_status == "valid":
        first_status = _cam_status(first, "valid", common)
    if second_status == "valid":
        second_status = _cam_status(aligned, "valid", common)
    for name, status in (("original", first_status), ("transformed", second_status)):
        if status != "valid":
            reasons.append(f"{name}_{status}_cam")
    if reasons:
        comparisons = tuple(JaccardResult(q, None, None, None, None) for q in TOP_FRACTIONS)
    else:
        first_order = _pixel_order(first, common, dataset_index)
        second_order = _pixel_order(aligned, common, dataset_index)
        comparisons = []
        for fraction in TOP_FRACTIONS:
            first_mask, first_ties = _mask_at_fraction(first, first_order, fraction)
            second_mask, second_ties = _mask_at_fraction(aligned, second_order, fraction)
            comparisons.append(JaccardResult(fraction, mask_jaccard(first_mask, second_mask),
                                              int(first_mask.sum()), first_ties, second_ties))
        comparisons = tuple(comparisons)
    before = target_scores(original_logits, original_class)
    after = target_scores(transformed_logits, original_class)
    probability_change = after.probability - before.probability
    margin_change = after.margin - before.margin
    if not np.isfinite((probability_change, margin_change)).all():
        raise ValueError("Non-finite original-class score change.")
    flip = original_class != transformed_class
    return StabilityResult(dataset_index, transformation.spec, original_class, transformed_class,
                           original_class, flip, "class_flipped" if flip else "prediction_preserving",
                           probability_change, margin_change,
                           comparisons, valid_fraction, transformation.clipped_fraction,
                           first_status, second_status, "unavailable" if reasons else "valid",
                           ";".join(reasons) if reasons else "accepted")


def partition_cases(results) -> dict[str, tuple[StabilityResult, ...]]:
    """Keep both groups, including unavailable comparisons; no favorable-case filter."""
    values = tuple(results)
    return {group: tuple(r for r in values if r.case_group == group)
            for group in ("prediction_preserving", "class_flipped")}


def compare_with_model(model: nn.Module, image, spec: TransformSpec, dataset_index: int) -> StabilityResult:
    """Optional single-array adapter on a disposable model, not a dataset runner."""
    if any(module.training for module in model.modules()):
        raise ValueError("Stability attribution requires evaluation mode in every module.")
    original_image = _intensity(image)
    transformation = apply_transform(original_image, spec, dataset_index)
    isolated = deepcopy(model)
    tensor = next(iter(isolated.parameters()), None)
    device = tensor.device if tensor is not None else torch.device("cpu")
    original = generate_gradcam(isolated, normalize_intensity(original_image).to(device))
    transformed = generate_gradcam(isolated, normalize_intensity(transformation.image).to(device),
                                   target_class=original.predicted_class)
    return compare_explanations(original, transformed, transformation, dataset_index)
