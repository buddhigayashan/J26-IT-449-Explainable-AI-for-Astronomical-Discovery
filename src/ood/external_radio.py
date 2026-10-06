"""Conservative display-map compatibility checks, not radio provenance detection.

No learned statistics or calibration are fitted here. The frozen OOD detector
remains mandatory after this preparation. Only positive, dark-background,
achromatic maps are supported; uncertain representations fail closed.
"""

from typing import Any

import numpy as np
from PIL import Image
from torchvision import transforms

from src.ood.input_guard import GuardedImage


def prepare_external_radio_image(image: Any) -> GuardedImage:
    """Validate display-map pixels without remapping classifier intensities.

    Median/percentile contrast is used ONLY for rejection checks. Preserve
    original 8-bit intensities, pad to square with zero and resize only when
    needed using PIL bilinear. Already compatible 150x150 grayscale inputs
    receive exactly the original ToTensor/Normalize operations. Numeric maps
    with unspecified physical/display scaling fail closed.
    Heuristic constants are policy limits, not validated survey thresholds.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Please upload one valid PIL radio galaxy image.")
    try:
        image.load()
        if min(image.size) < 16 or max(image.size) > 4096:
            raise ValueError("External image dimensions must be between 16 and 4096 pixels.")
        if image.mode not in {"L", "RGB", "RGBA", "F", "I", "I;16"}:
            raise ValueError("Unsupported external image pixel mode; use a grayscale or RGB radio map.")
        raw = np.asarray(image)
        if not np.isfinite(raw).all():
            raise ValueError("Image contains non-finite pixel values.")
        if image.mode not in {"L", "RGB", "RGBA"}:
            raise ValueError("Numeric radio maps have an unspecified intensity scale; supply a known 8-bit grayscale display map.")
        if image.mode == "RGBA" and not np.all(raw[..., 3] == 255):
            raise ValueError("Transparent radio maps are ambiguous; upload an opaque map.")
        if image.mode in {"RGB", "RGBA"}:
            rgb = raw[..., :3].astype(np.float32) / 255.0
            if float(np.mean(np.ptp(rgb, axis=-1) > 0.08)) > 0.01:
                raise ValueError("Colored or composite imagery is unsupported; supply an achromatic radio map.")
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        else:
            pixels = raw.astype(np.float64)
            pixels /= 255.0
        if not np.isfinite(pixels).all():
            raise ValueError("Image contains non-finite pixel values.")
        low, high = float(pixels.min()), float(pixels.max())
        if high == low:
            raise ValueError("Blank or uniform images are unsupported.")
        # Check original intensities so numerical noise cannot become fake emission.
        scale = max(abs(low), abs(high), 1.0)
        if float(pixels.std()) / scale < 0.001:
            raise ValueError("Near-uniform images are unsupported.")
        background = float(np.median(pixels))
        upper = float(np.percentile(pixels, 99.5))
        if upper - background <= 0.001 * scale:
            raise ValueError("Insufficient resolved positive radio-map emission.")
        domain_signal = np.clip((pixels - background) / (upper - background), 0, 1)
        # Require sparse resolved signal on a quiet background. These checks
        # reject many photographs/screenshots, but cannot identify radio emission.
        foreground = domain_signal > 0.2
        occupancy = float(foreground.mean())
        border = np.concatenate((domain_signal[0], domain_signal[-1],
                                 domain_signal[:, 0], domain_signal[:, -1]))
        if not 0.005 <= occupancy <= 0.20 or float(np.mean(border > 0.2)) > 0.05:
            raise ValueError("Image lacks sparse emission on a quiet border; optical photos, plots, and screenshots are unsupported.")
        if float(np.mean(pixels < background - 0.2 * (upper - background))) > 0.10:
            raise ValueError("Background structure is inconsistent with a supported positive-emission radio map.")
        # Do not feed domain_signal to the classifier: stretching/clipping
        # changes brightness relationships learned from original MiraBest pixels.
        preview = image.convert("L")
        if image.width != image.height:
            side = max(image.size)
            square = Image.new("L", (side, side), 0)
            square.paste(preview, ((side - image.width) // 2, (side - image.height) // 2))
            preview = square
        if preview.size != (150, 150):
            preview = preview.resize((150, 150), Image.Resampling.BILINEAR)
        tensor = transforms.ToTensor()(preview)
        tensor = transforms.Normalize((0.5,), (0.5,))(tensor).unsqueeze(0)
        return GuardedImage(preview, tensor)
    except (OSError, TypeError, SyntaxError) as error:
        raise ValueError("The uploaded file is not a valid image.") from error
