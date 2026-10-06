# AUTO / WEB RADIO IMAGE (EXPERIMENTAL)

AUTO is the recommended Gradio choice. Upload an opaque, single-frame image
containing genuine radio morphology. Web origin alone does not establish radio
provenance. The original MiraBest and stricter external-radio modes remain
available; `predict_image` still defaults to `mirabest` for API reproducibility.
Use `predict_image(image, input_mode="auto_web")` to request AUTO explicitly.

## Architecture

1. Decode PIL input; reject corruption, unknown numeric scaling, NaN/Inf,
   transparency, animations, and dimensions outside 16–4096 pixels per axis.
   Palette PNGs are decoded with their palette and alpha before processing.
2. Apply whole-image radio-map representation checks. Quiet-background, sparse
   positive-emission structure is required, without requiring symmetry or a
   centered source. Reject blank/uniform, dense or photographic structure,
   substantial boundary content, negative-background structure, and excessive
   high-frequency diagnostic texture. These are deterministic policy limits,
   not semantic identification of radio galaxies. All numeric policy values
   are stored in the separate calibration metadata.
3. For grayscale or effectively achromatic inputs, preserve the original
   grayscale pixel intensities. For 150x150 inputs, geometry is untouched.
   If the original frozen OOD checks pass, AUTO uses exactly the original
   classifier tensor, probabilities, and label. No confidence rule or new gate
   changes this accepted original route. If original OOD rejects, AUTO requires
   the separate external gate, just like other external candidates; original
   mode itself continues to reject under its original rules.
4. For color images, require approximately affine, positive shared spatial
   structure across **every varying RGB channel**. The source luminance, each
   varying channel, and maximum-channel image must each pass the representation
   checks. A sparse channel cannot rescue dense/independently textured channels.
   Channel coherence is measured over the union of their emission regions so
   a large empty background cannot dilute disagreement between sparse structures.
   Candidate images are luminance, individual R/G/B intensities, and maximum
   channel intensity. Constant channels are omitted; exact duplicate tensors
   are counted once. This supports simple shared-morphology color encodings,
   not arbitrary nonlinear palettes or arbitrary optical/radio composites.
5. Symmetrically pad non-square candidates using the rounded median of all four
   source borders, preserving all original content. Resize only when needed
   using PIL bilinear to 150x150. No crop, source selection, inversion, smoothing,
   or intensity stretching enters the classifier. Convert via torchvision
   ToTensor float32 and Normalize(mean=0.5, std=0.5).
6. Require separately calibrated energy **and** feature-distance gates for all
   plausible external candidates, never heuristic-only fallback. Classify all
   candidates with the locked ResNet. Any label disagreement returns AMBIGUOUS,
   even if one candidate is more confident or another fails its gate. If they
   agree but any candidate fails a gate or has confidence below the fixed 0.60
   policy minimum, return UNSUPPORTED. Otherwise report the unanimous label
   with mean candidate probabilities, never highest-confidence selection.
7. UI reports class/confidence, handling route, candidate names, and the
   experimental limitation. Rejection/ambiguity has no classification scores.
   Numeric gate diagnostics are available in the structured API result, not
   the ordinary UI.

The strict `external_radio` mode is unchanged by AUTO and still requires the
original frozen OOD checks. See [external_radio_inputs.md](external_radio_inputs.md).

## Separate calibration

Run `python scripts/calibrate_external_compatibility.py`. This writes only:

- `outputs/ood/external_compatibility_calibration.pt`
- `outputs/ood/external_compatibility_calibration.json`

These generated local artifacts are ignored by the repository's existing
`outputs/ood/` rule; they must be generated/transferred explicitly for deployment.
If absent, stale, incompatible, or corrupted, the external candidate path fails
closed. Compatible original-route images can still use the original calibration.

The verified local loader is used with `download=False`. Only the existing
554 TRAIN and 119 VALIDATION indices are selected, without reading labels to
select positives. The final test split is inspected as **indices only** to
enforce disjointness, and is never selected for calibration. The two exported
held-out demo PNGs are used only in explicit consistency regressions. The
seven reserved external OOD images are neither read nor used to select policy.

Each allowed sample produces ten fixed variant families: original; centered
180x150 and 150x180 canvases; bilinear 150→96→150 degradation; contrast ×0.9
and ×1.1; gamma 0.9 and 1.1; affine RGB encoding `(p, 0.75p, 0.5p)`; and
grayscale JPEG quality 95. No crops, rotation, label-dependent choices, or
automatic brightness optimization are used. Mild clipping/quantization occurs
only in explicitly defined **calibration** variants, never as runtime intensity
remapping. The policy is fixed before evaluating the requested test regressions.

The same candidate extraction/guards apply to variants. Rejected variants are
excluded with per-family counts recorded. TRAIN candidates define the 512-element
feature mean and diagonal variance (floor 1e-6) from the locked ResNet spatial
pool. VALIDATION candidates independently define upper energy and standardized
feature-distance thresholds at the predeclared 99th percentile. No optimizer,
weight update, runtime fitting, or final test threshold selection occurs.

The artifact includes exact TRAIN/VALIDATION indices, counts, split-file path
and SHA256, checkpoint path and SHA256, verified dataset manifest SHA256,
feature dimension/layer, transformation definitions, seed 42, preprocessing
version/configuration, representation rules, consistency policy, and accepted/
rejected variant counts. Runtime validates the model, policies, statistics, and
current split bytes against this provenance. Original OOD artifacts are never
written; no original threshold is changed.

## Scientific limits

Positive-only synthetic calibration measures compatibility with transformed
MiraBest, **not** a false-accept rate for natural/optical images and not
external-survey accuracy. Candidate agreement is correlated evidence from one
image, not independent confirmation of correctness. A shared optical structure
or drawing can resemble a radio map and pass these checks. Conversely genuine
radio maps with extended/full-frame emission, different backgrounds, weak signal,
nonlinear color palettes, annotations, or different survey resolution/sensitivity
can be rejected. Arbitrary web images remain unsuitable for scientific FR claims.

Tests use synthetic unrelated styles for rejection and sentinel TEST indices for
calibration discipline. They do not establish semantic rejection of every person,
car, dog, landscape, optical galaxy, screenshot, or drawing. Independent external
radio positives and unrelated negatives are required for a descriptive evaluation
before making stronger reliability claims. Reported final MiraBest metrics are
unchanged and do not describe AUTO performance.
