"""Intensity-proxy alignment only, not verified astrophysical localization.

Phase3B-XAI-v1.0.0 numerical core: supplied arrays only; no model/dataset access,
image loading, file writes, preprocessing, OOD decisions or statistical tests.
Pass GradCAMResult.upsampled_heatmap, NOT its min-max display heatmap. Values
alone cannot establish that provenance; future callers must enforce it.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.xai.stability import mask_jaccard, topk_mask

VARIANTS = (("inclusive", 2.0, 0.02), ("primary", 3.0, 0.05), ("restrictive", 5.0, 0.10))
QUANTIZATION_FLOOR = 1.0 / 255.0
RANDOM_SEEDS = tuple(range(1000, 1020))
NOTICE = "Intensity-derived emission proxy; not verified jets, lobes, cores or hotspots."


class AlignmentInputError(ValueError):
    """Invalid input cannot produce meaningful diagnostics; stable reason_code attached."""

    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__(reason_code)


def _array(value, name, *, finite=True):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (150, 150) or array.dtype.kind not in "fiu":
        raise AlignmentInputError(f"invalid_{name}_shape_or_dtype")
    array = array.astype(np.float64, copy=True)
    if finite and not np.isfinite(array).all():
        raise AlignmentInputError(f"nonfinite_{name}")
    return array


def _intensity(image):
    image = _array(image, "image")
    if image.min() < 0 or image.max() > 1:
        raise AlignmentInputError("invalid_intensity_range")
    return image


def _index(index):
    if type(index) is not int or not 0 <= index < 792:
        raise AlignmentInputError("invalid_dataset_index")


def _mask(mask):
    mask = np.asarray(mask)
    if mask.shape != (150, 150) or mask.dtype != np.bool_:
        raise AlignmentInputError("invalid_emission_mask")
    return mask.copy()


def _border_mask():
    border = np.ones((150, 150), dtype=bool)
    border[5:-5, 5:-5] = False
    return border


def border_values(image) -> np.ndarray:
    """Exact outer-five-pixel union, row-major, each corner counted once (2900)."""
    return _intensity(image)[_border_mask()].copy()


@dataclass(frozen=True)
class EmissionProxy:
    variant: str
    sigma_factor: float
    dynamic_fraction: float
    background_median: float
    border_mad: float
    sigma_b: float
    maximum_intensity: float
    sigma_component: float
    dynamic_component: float
    quantization_floor: float
    threshold_offset: float
    absolute_threshold: float
    emission_mask: np.ndarray
    emission_pixels: int
    emission_area_fraction: float
    zero_border_dispersion: bool
    border_touching_emission: bool
    emission_in_border_band: bool
    status: str
    reasons: tuple[str, ...]
    notice: str = NOTICE


def emission_proxies(image) -> tuple[EmissionProxy, ...]:
    """Construct all three masks from intensity alone, without any CAM argument.

    Border-touching means the outermost image perimeter; band occupancy is also
    recorded separately. Zero MAD and border contact are diagnostics, not mask
    repairs/exclusions. Thresholds above the image maximum are not clipped.
    """
    image = _intensity(image)
    border = image[_border_mask()]
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    sigma = 1.4826 * mad
    maximum = float(image.max())
    results = []
    for variant, factor, dynamic_fraction in VARIANTS:
        sigma_component = factor * sigma
        dynamic_component = dynamic_fraction * (maximum - background)
        threshold = max(sigma_component, dynamic_component, QUANTIZATION_FLOOR)
        mask = image > background + threshold
        count = int(mask.sum())
        reasons = ("empty_emission_mask",) if count == 0 else (("full_emission_mask",) if count == 22500 else ())
        touching = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        band = bool(mask[_border_mask()].any())
        mask.setflags(write=False)
        results.append(EmissionProxy(variant, factor, dynamic_fraction, background, mad, sigma,
                                     maximum, sigma_component, dynamic_component, QUANTIZATION_FLOOR,
                                     threshold, background + threshold, mask, count, count / 22500,
                                     sigma == 0, touching, band,
                                     "unavailable" if reasons else "valid", reasons))
    return tuple(results)


@dataclass(frozen=True)
class AttributionOverlap:
    attribution_kind: str
    emission_area_fraction: float
    attribution_total: float | None
    emission_attribution_mass: float | None
    background_attribution_mass: float | None
    area_adjusted_enrichment: float | None
    status: str
    reasons: tuple[str, ...]
    top10_emission_iou: float | None
    top10_pixels: int | None
    top10_status: str
    top10_reasons: tuple[str, ...]


def attribution_overlap(attribution, emission_mask, dataset_index, *, attribution_kind="gradcam",
                        cam_status="valid") -> AttributionOverlap:
    """Measure raw positive attribution; unavailable fields are None, never zero-imputed.

    Constant positive uniform attribution is an intentional analytic control: its
    mass remains meaningful, while its top-k metric is unavailable. Constant or
    declared-degenerate Grad-CAM never acquires a fabricated spatial ranking.
    """
    _index(dataset_index)
    if attribution_kind not in {"gradcam", "brightness", "uniform"}:
        raise AlignmentInputError("invalid_attribution_kind")
    if cam_status not in {"valid", "degenerate"}:
        raise AlignmentInputError("invalid_cam_status")
    values = _array(attribution, "attribution", finite=False)
    mask = _mask(emission_mask)
    count = int(mask.sum())
    area = count / 22500
    reasons = []
    if count == 0:
        reasons.append("empty_emission_mask")
    elif count == 22500:
        reasons.append("full_emission_mask")
    total = None
    degenerate = False
    if not np.isfinite(values).all():
        reasons.append("nonfinite_attribution")
    elif np.any(values < 0):
        raise AlignmentInputError("negative_attribution")
    else:
        with np.errstate(over="ignore"):
            total = float(values.sum(dtype=np.float64))
            representable = values.astype(np.float32)
        if not np.isfinite(total):
            reasons.append("nonfinite_attribution_total")
            total = None
        elif total == 0:
            reasons.append("zero_attribution")
        elif not np.isfinite(representable).all():
            reasons.append("attribution_float32_overflow")
        else:
            tolerance = np.finfo(np.float32).eps * max(float(representable.max()), 1.0)
            degenerate = (float(representable.max()) - float(representable.min())) <= tolerance
    if attribution_kind == "gradcam" and (cam_status == "degenerate" or degenerate):
        reasons.append("degenerate_gradcam")
    if attribution_kind == "uniform" and np.isfinite(values).all() and values.min() != values.max():
        raise AlignmentInputError("nonuniform_uniform_control")
    if reasons:
        reason_tuple = tuple(reasons)
        return AttributionOverlap(attribution_kind, area, total, None, None, None,
                                  "unavailable", reason_tuple, None, None, "unavailable", reason_tuple)
    emission_mass = float(values[mask].sum(dtype=np.float64) / total)
    background_mass = 1.0 - emission_mass
    enrichment = emission_mass - area
    if not np.isfinite((emission_mass, background_mass, enrichment)).all():
        raise AlignmentInputError("nonfinite_overlap_arithmetic")
    if degenerate:
        iou, pixels = None, None
        top_reasons = ("constant_control_has_no_spatial_ranking",)
    else:
        top = topk_mask(values, np.ones((150, 150), dtype=bool), dataset_index, fraction=0.10)
        iou, pixels = mask_jaccard(top, mask), int(top.sum())
        top_reasons = ()
    return AttributionOverlap(attribution_kind, area, total, emission_mass, background_mass, enrichment,
                              "valid", (), iou, pixels, "unavailable" if top_reasons else "valid", top_reasons)


def brightness_attribution(image) -> np.ndarray:
    image = _intensity(image)
    background = float(np.median(image[_border_mask()]))
    return np.maximum(image - background, 0)


@dataclass(frozen=True)
class RandomControlAvailability:
    status: str = "unavailable"
    reason: str = "random_native_to_input_projection_unspecified"
    seeds: tuple[int, ...] = RANDOM_SEEDS
    native_shape: tuple[int, int] = (19, 19)
    generator: str = "NumPy Generator(PCG64(SeedSequence([seed,dataset_index,1])))"


def random_control_availability() -> RandomControlAvailability:
    """Frozen policy supplies native maps/seeds but no explicit 19->150 projection.

    Do not silently reuse Grad-CAM interpolation or invent pixel-grid randomness.
    No RNG is consumed. Completing this optional control requires clarification.
    """
    return RandomControlAvailability()


@dataclass(frozen=True)
class ProxyAlignment:
    proxy: EmissionProxy
    gradcam: AttributionOverlap
    brightness: AttributionOverlap
    uniform: AttributionOverlap
    random: RandomControlAvailability


@dataclass(frozen=True)
class EmissionAlignmentResult:
    dataset_index: int
    variants: tuple[ProxyAlignment, ...]
    notice: str = NOTICE


def audit_emission_alignment(image, upsampled_cam, dataset_index, *, cam_status="valid") -> EmissionAlignmentResult:
    """Supplied-array core only; preserves every variant and unavailable reason.

    Masks and baseline attribution use no CAM values. This function neither
    generates explanations nor validates astrophysical structures.
    """
    _index(dataset_index)
    image = _intensity(image)
    values = _array(upsampled_cam, "attribution", finite=False)
    proxies = emission_proxies(image)
    brightness = brightness_attribution(image)
    uniform = np.ones((150, 150), dtype=np.float64)
    results = tuple(ProxyAlignment(
        proxy,
        attribution_overlap(values, proxy.emission_mask, dataset_index, cam_status=cam_status),
        attribution_overlap(brightness, proxy.emission_mask, dataset_index, attribution_kind="brightness"),
        attribution_overlap(uniform, proxy.emission_mask, dataset_index, attribution_kind="uniform"),
        random_control_availability()) for proxy in proxies)
    return EmissionAlignmentResult(dataset_index, results)
