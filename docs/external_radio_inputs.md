# External radio-map compatibility

The UI now recommends a separate [AUTO / WEB RADIO IMAGE path](auto_web_radio_inputs.md).
This document describes the retained strict `external_radio` mode, whose OOD
requirements remain unchanged.

Select **External radio map (experimental)** in the demo, or call
`predict_image(pil_image, input_mode="external_radio")`. The default
`mirabest` path retains its original preprocessing and decisions. Internal
class keys remain `FRI`, `FRII`, and `UNSUPPORTED` for compatibility.

This mode is intended for opaque achromatic radio intensity-map cutouts with
positive emission against a quiet dark background. It does not authenticate
radio data or make arbitrary web images scientifically suitable. Color maps,
optical composites, annotated figures, screenshots, inverted maps, and ambiguous
inputs should be supplied as original intensity maps instead. RGB and opaque
RGBA representations of grayscale maps are supported.

## Exact deterministic preparation

1. Require a decodable PIL image, dimensions from 16 through 4096 pixels per
   axis, and L, RGB, or RGBA pixel mode. Check original pixel values for NaN/Inf
   before conversions. Reject non-opaque alpha. Numeric F/I/I;16 maps fail closed
   because their physical/display intensity scale is unspecified.
2. For RGB/RGBA reject if more than 1% of pixels have channel spread greater
   than 0.08 of the 8-bit range; convert accepted color representations with
   PIL `convert("L")`. Preserve the original grayscale intensities.
3. Reject constant images and relative standard deviation below 0.001, using
   `max(abs(min), abs(max), 1)` as scale, on pixels divided by 255.
4. For **rejection checks only**, subtract the median background and divide by
   the difference between the 99.5th percentile and median. Reject if that
   difference is at most 0.001 times the scale. Clip this diagnostic array to
   [0, 1]. This array is never encoded into the classifier image or tensor.
   No background subtraction, contrast stretching, percentile clipping, or
   polarity inversion is applied to the classifier input.
5. Require 0.5–20% of pixels above normalized intensity 0.2, at most 5% of border
   pixels above 0.2, and at most 10% of original pixels below median minus 0.2
   times the normalization range. These are conservative representation rules,
   not thresholds fitted to a survey or the final MiraBest test set.
6. Preserve the original 8-bit grayscale pixels; symmetrically pad the shorter
   dimension to square with zero (the extra pixel goes on the right/bottom).
   Preserve the complete frame. Resize to 150x150 using **PIL bilinear** only
   when required. A 150x150 grayscale input retains its original pixels exactly.
7. Convert to float32, divide by 255, then apply `(pixel - 0.5) / 0.5` to produce
   a `(1, 1, 150, 150)` tensor in [-1, 1], using the same torchvision ToTensor
   and Normalize as the original MiraBest path.
8. Require the existing frozen energy **and** feature-distance OOD checks on
   exactly that tensor. No relaxed thresholds, runtime calibration, alternate
   model, or radio-looking exception. Only then report classifier probabilities.
   Any quality, loading, OOD, or inference failure returns UNSUPPORTED with a reason.

## Scientific limits

The weights, frozen calibration, training pipeline, reported final metrics,
and original MiraBest input path are unchanged. The final test split is not
used for choosing any compatibility rule. The two explicitly requested held-out
PNG exports are used only for consistency regression: identical classifier
tensors, labels, probabilities, and OOD outcomes in both modes. They are not
used to select thresholds or preprocessing parameters. Synthetic rejection
tests establish software behavior, not semantic rejection rates or survey accuracy.

Radio morphology depends on resolution, sensitivity, observing frequency,
dynamic range, and source selection. Resizing loses information even though
padding avoids cropping lobes. Intensity conventions different from the training
data remain unsupported or unvalidated; no automatic remapping solves this.
High background offsets or negative-emission maps can be rejected.
Real maps with emission at the boundary, large diffuse emission, colored
palettes, or weak signal may fail closed. Frozen MiraBest OOD thresholds can
remain too strict for other surveys; this implementation does not relax them.

Sparse grayscale drawings, isolated optical objects, or dark screenshots can
imitate these representation rules. Passing the rules and OOD is not proof of
radio provenance; no deterministic pixel heuristic guarantees rejection of all
people, cars, dogs, landscapes, or optical images. Supply known radio data and
treat external predictions as exploratory, without a validated accuracy claim.
Dedicated independent external-survey validation would be needed for such claims.

## Intensity consistency correction

The initial experimental implementation fed the diagnostic percentile stretch
into the classifier. This boosted faint emission and saturated bright structure,
changing the learned input representation and potentially the predicted class.
The correction separates unchanged domain rejection checks from pixel-preserving
classifier preparation for all supported 8-bit maps, regardless of source,
filename, label, or dimensions. There is no sample-specific exception and no
OOD bypass. Numeric physical-intensity maps require a scientifically established
display representation outside this experimental path.
