"""Class-logit Grad-CAM for the frozen lightweight ResNet, without state updates.

This numerical API accepts the exact prepared classifier tensor. It does not
preprocess images, make OOD decisions, load checkpoints, or write artifacts.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

TARGET_LAYER = "blocks.2.convolution.3"
CLASS_NAMES = ("FRI", "FRII")


@dataclass(frozen=True)
class GradCAMResult:
    """Detached attribution outputs; classes use the classifier's integer mapping.

    ``raw_cam`` is the nonnegative 19x19 map. ``upsampled_heatmap`` retains
    its magnitude at 150x150; ``heatmap`` is its display normalization [0, 1].
    Degenerate maps have no spatial contrast and receive a zero display map.
    """

    raw_cam: torch.Tensor
    upsampled_heatmap: torch.Tensor
    heatmap: torch.Tensor
    target_class: int
    predicted_class: int
    logits: torch.Tensor
    layer_name: str
    status: Literal["valid", "degenerate"]


def _cam_from_activations(activations, gradients):
    """Spatially average logit gradients, weight channels, sum, then ReLU."""
    if (activations.ndim != 4 or activations.shape != gradients.shape
            or activations.shape[0] != 1):
        raise ValueError("Grad-CAM requires matching single-image NCHW activations and gradients.")
    if not torch.isfinite(activations).all() or not torch.isfinite(gradients).all():
        raise ValueError("Non-finite Grad-CAM activations or gradients.")
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).relu()[0]
    if not torch.isfinite(cam).all():
        raise ValueError("Non-finite Grad-CAM channel sum.")
    return cam


def generate_gradcam(model: nn.Module, input_tensor: torch.Tensor,
                     target_class: int | None = None) -> GradCAMResult:
    """Explain one class logit at ``model.blocks[2].convolution[3]``.

    Requires an entirely evaluation-mode model and a finite float32 input of
    shape (1, 1, 150, 150). No parameter ``.grad`` fields are populated or
    cleared. Parameter requires-grad flags, buffers and caller inputs are not
    changed. Calls on the same model must be serialized: hooks are temporary
    but attached to the supplied module while its forward pass runs.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Grad-CAM cannot run inside torch.inference_mode().")
    if any(module.training for module in model.modules()):
        raise ValueError("Grad-CAM requires evaluation mode for every model module.")
    if not isinstance(input_tensor, torch.Tensor):
        raise ValueError("Grad-CAM input must be a float32 tensor.")
    if input_tensor.shape != (1, 1, 150, 150) or input_tensor.dtype != torch.float32:
        raise ValueError("Grad-CAM input must be float32 with shape (1, 1, 150, 150).")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("Grad-CAM input must contain only finite values.")
    if target_class is not None and (type(target_class) is not int or target_class not in (0, 1)):
        raise ValueError("Grad-CAM target_class must be 0 (FRI), 1 (FRII), or None.")
    try:
        layer = model.blocks[2].convolution[3]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError(f"Grad-CAM target layer {TARGET_LAYER} is unavailable.") from error
    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"Grad-CAM target layer {TARGET_LAYER} must be a convolution.")
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            # Fresh ordinary tensor also supports an input created in inference mode.
            inputs = input_tensor.detach().clone().requires_grad_(True)
            logits = model(inputs)
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError("Grad-CAM requires finite classifier logits with shape (1, 2).")
            if len(captured) != 1 or captured[0].shape != (1, 128, 19, 19):
                raise ValueError("Expected one target-layer activation with shape (1, 128, 19, 19).")
            activations = captured[0]
            if not activations.requires_grad:
                raise RuntimeError("Grad-CAM target activations are disconnected from autograd.")
            predicted = int(logits.detach().argmax(dim=1).item())
            target = predicted if target_class is None else target_class
            gradients, = torch.autograd.grad(logits[0, target], activations)
            raw = _cam_from_activations(activations.detach(), gradients.detach())
            upsampled = F.interpolate(raw[None, None], size=(150, 150),
                                      mode="bilinear", align_corners=False)[0, 0]
            if not torch.isfinite(upsampled).all():
                raise ValueError("Non-finite upsampled Grad-CAM.")
            minimum, maximum = upsampled.min(), upsampled.max()
            span = maximum - minimum
            # Relative float32 resolution; tiny maps are not arbitrarily amplified.
            tolerance = torch.finfo(upsampled.dtype).eps * maximum.abs().clamp_min(1.0)
            degenerate = bool(span <= tolerance)
            heatmap = (torch.zeros_like(upsampled) if degenerate
                       else ((upsampled - minimum) / span).clamp(0, 1))
            return GradCAMResult(raw, upsampled, heatmap, target, predicted,
                                 logits.detach().clone(), TARGET_LAYER,
                                 "degenerate" if degenerate else "valid")
    finally:
        handle.remove()
