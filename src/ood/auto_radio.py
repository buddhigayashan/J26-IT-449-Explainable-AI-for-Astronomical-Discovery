"""Deterministic, pixel-preserving candidates for conservative web-map support."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from src.ood.input_guard import GuardedImage

PREPROCESSING_VERSION = "auto-radio-v1-pixel-preserving"
PREPROCESSING_CONFIGURATION = {
    "image_size": [150, 150], "grayscale": "PIL L",
    "padding": "symmetric; rounded median of all four source borders; no crop",
    "resize": "PIL bilinear only if size differs", "intensity_remapping": False,
    "tensor": "torchvision ToTensor float32; mean=0.5; std=0.5",
    "alpha": "all pixels must be fully opaque", "max_frames": 1,
    "dimension_limits": [16, 4096], "accepted_modes": ["L", "RGB", "RGBA", "P", "1"],
}
DOMAIN_RULES = {
    "relative_std_min": 0.001, "signal_range_min": 0.001,
    "foreground_fraction_min": 0.005, "foreground_fraction_max": 0.20,
    "foreground_level": 0.2, "border_foreground_max": 0.05,
    "negative_fraction_max": 0.10, "texture_residual_max": 0.35,
    "color_affine_residual_max": 0.05,
    "achromatic_channel_spread_max": 2 / 255.0,
    "active_channel_range_min": 0.01, "color_reference_variance_min": 1e-6,
    "color_residual_region": "union of channel/reference emission above 20% dynamic range",
}


@dataclass(frozen=True)
class RadioCandidate:
    representation: str
    guarded: GuardedImage


@dataclass(frozen=True)
class AutoCandidates:
    candidates: tuple[RadioCandidate, ...]
    already_compatible: bool


def validate_upload(image: Any) -> Image.Image:
    """Decode once; never hide NaNs, alpha, frame changes, or unknown scaling."""
    if not isinstance(image, Image.Image):
        raise ValueError("Please upload one valid radio-galaxy image.")
    try:
        if min(image.size) < 16 or max(image.size) > 4096:
            raise ValueError("Image dimensions must be between 16 and 4096 pixels.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated or multi-frame images are unsupported.")
        image.load()
        if not np.isfinite(np.asarray(image)).all():
            raise ValueError("Image contains non-finite pixel values.")
        if image.mode not in {"L", "RGB", "RGBA", "P", "1"}:
            raise ValueError("Numeric maps need a known 8-bit display scale; automatic intensity stretching is unsupported.")
        if image.mode in {"RGBA", "P"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            if not np.all(np.asarray(rgba)[..., 3] == 255):
                raise ValueError("Transparent images are unsupported; supply an opaque radio map.")
            return rgba.convert("RGB")
        return image.convert("L") if image.mode == "1" else image.copy()
    except (OSError, TypeError, SyntaxError) as error:
        raise ValueError("The uploaded file is malformed or corrupted.") from error


def _border(pixels: np.ndarray) -> np.ndarray:
    return np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))


def check_radio_structure(grayscale: Image.Image) -> None:
    """Representation rules only; neither semantic radio detection nor OOD."""
    pixels = np.asarray(grayscale, dtype=np.float64) / 255.0
    if float(pixels.max()) == float(pixels.min()):
        raise ValueError("Blank or uniform images are unsupported.")
    if float(pixels.std()) < DOMAIN_RULES["relative_std_min"]:
        raise ValueError("Near-uniform images are unsupported.")
    background = float(np.median(pixels))
    span = float(np.percentile(pixels, 99.5)) - background
    if span <= DOMAIN_RULES["signal_range_min"]:
        raise ValueError("Image has insufficient resolved positive emission.")
    diagnostic = np.clip((pixels - background) / span, 0, 1)
    fraction = float(np.mean(diagnostic > DOMAIN_RULES["foreground_level"]))
    if not DOMAIN_RULES["foreground_fraction_min"] <= fraction <= DOMAIN_RULES["foreground_fraction_max"]:
        raise ValueError("Image does not contain a sufficiently compatible radio-morphology representation (dense or insufficient emission).")
    if float(np.mean(_border(diagnostic) > DOMAIN_RULES["foreground_level"])) > DOMAIN_RULES["border_foreground_max"]:
        raise ValueError("Full-frame border structure is unsupported; upload a source map without annotations or a photographic frame.")
    if float(np.mean(pixels < background - 0.2 * span)) > DOMAIN_RULES["negative_fraction_max"]:
        raise ValueError("Structured background is incompatible with a supported positive-emission map.")
    # Diagnostic blur only: never feed a smoothed image to the classifier.
    diagnostic_image = Image.fromarray(np.rint(diagnostic * 255).astype(np.uint8))
    blurred = np.asarray(diagnostic_image.filter(ImageFilter.GaussianBlur(1)), dtype=np.float64) / 255
    strong = diagnostic > DOMAIN_RULES["foreground_level"]
    if float(np.mean(np.abs(diagnostic - blurred)[strong])) > DOMAIN_RULES["texture_residual_max"]:
        raise ValueError("High-frequency texture is incompatible with a supported radio map.")


def standardize_grayscale(image: Image.Image) -> GuardedImage:
    """Keep intensities; estimate padding from the median of the full border."""
    preview = image.convert("L")
    if preview.width != preview.height:
        side = max(preview.size)
        background = int(np.rint(np.median(_border(np.asarray(preview)))))
        square = Image.new("L", (side, side), background)
        square.paste(preview, ((side - preview.width) // 2, (side - preview.height) // 2))
        preview = square
    if preview.size != (150, 150):
        preview = preview.resize((150, 150), Image.Resampling.BILINEAR)
    tensor = transforms.Normalize((0.5,), (0.5,))(transforms.ToTensor()(preview)).unsqueeze(0)
    return GuardedImage(preview, tensor)


def extract_auto_candidates(image: Any) -> AutoCandidates:
    """Require whole-image compatibility before trying any individual channel.

    Supported color encoding is approximately affine single-structure RGB.
    Independently textured/composite channels fail closed. No candidate-specific
    stretching, source extraction, cropping, inversion, or confidence selection.
    """
    source = validate_upload(image)
    grayscale = source.convert("L")
    check_radio_structure(grayscale)
    if source.mode == "L":
        return AutoCandidates((RadioCandidate("grayscale", standardize_grayscale(grayscale)),),
                              source.size == (150, 150))
    rgb = np.asarray(source.convert("RGB"), dtype=np.float64) / 255.0
    achromatic = float(np.max(np.ptp(rgb, axis=-1))) <= DOMAIN_RULES["achromatic_channel_spread_max"]
    if achromatic:
        return AutoCandidates((RadioCandidate("achromatic_luminance", standardize_grayscale(grayscale)),),
                              source.size == (150, 150))
    channels = [source.convert("RGB").getchannel(c) for c in "RGB"]
    spreads = np.ptp(rgb.reshape(-1, 3), axis=0)
    reference = rgb[..., int(np.argmax(spreads))].reshape(-1)
    centered = reference - reference.mean()
    variance = float(np.mean(centered ** 2))
    if variance <= DOMAIN_RULES["color_reference_variance_min"]:
        raise ValueError("Color image contains insufficient coherent radio-map structure.")
    # A sparse channel cannot rescue unrelated dense or independently textured
    # content in another channel. Every varying channel must share the structure.
    active_channels = []
    for index, channel in enumerate(channels):
        if spreads[index] < DOMAIN_RULES["active_channel_range_min"]:
            continue
        check_radio_structure(channel)
        values = rgb[..., index].reshape(-1)
        slope = float(np.mean(centered * (values - values.mean()))) / variance
        fitted = values.mean() + slope * centered
        emission = ((values - values.min()) > 0.2 * spreads[index]) | (
            (reference - reference.min()) > 0.2 * float(np.ptp(reference)))
        # Measure coherence where the emission actually is. A mostly empty
        # frame must not dilute a mismatch between independent sparse objects.
        residual = float(np.sqrt(np.mean((values - fitted)[emission] ** 2))) / spreads[index]
        if slope <= 0 or residual > DOMAIN_RULES["color_affine_residual_max"]:
            raise ValueError("Color channels contain inconsistent structures; optical/composite imagery is unsupported.")
        active_channels.append(("RGB"[index] + "_channel", channel))
    maximum = Image.fromarray(np.max(np.asarray(source.convert("RGB")), axis=-1))
    check_radio_structure(maximum)
    representations = [("luminance", grayscale), *active_channels, ("maximum_channel", maximum)]
    candidates = []
    seen = set()
    for name, channel in representations:
        guarded = standardize_grayscale(channel)
        # Exact deduplication avoids counting identical channels as extra votes.
        identity = guarded.image.tobytes()
        if identity not in seen:
            seen.add(identity)
            candidates.append(RadioCandidate(name, guarded))
    return AutoCandidates(tuple(candidates), False)
