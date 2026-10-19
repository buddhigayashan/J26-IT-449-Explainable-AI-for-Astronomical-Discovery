# Phase3B-XAI-v1.0.0: frozen exploratory evaluation protocol

Project: J26-IT-449-IT23308602-AGN-XAI. Research: AGN-Aware Radio Source
Structure Classification using CNN + XAI.

Status: methodology approved for staged implementation. Phase 3B.1 implements
documentation, configuration, metadata provenance and isolation only. This file
does not authorize metric execution, training, test evaluation, or UI integration.
Phase 3B.2 requires separate authorization. The YAML is the machine-readable
counterpart; its canonical parsed-content SHA-256 is pinned independently in
`src/xai/evaluation_data.py`. Any semantic configuration change must fail closed
until a reviewed version/pin update. No outcome-driven amendment is permitted.

## Research questions and population

Separately evaluate deletion sensitivity, transformation stability, intensity
alignment and dependence on learned parameters. Do not combine them into an
overall XAI quality score. Hypotheses: CAM-ranked deletion reduces original-class
probability more than random deletion; it provides information beyond brightness;
aligned maps remain similar under small transformations when outputs are stable;
attribution is enriched within an intensity-derived emission proxy; explanations
depend on learned parameters. These are hypotheses, not guaranteed findings.

Frozen payload: 792 binary MiraBest images. Saved splits: 554 TRAIN, 119 VALIDATION,
119 locked TEST. Label mapping: 0=FRI, 1=FRII. All 119 validation images are included,
including errors and degeneracies. No confidence/correctness/visual filtering.
Validation contains 53 FRI and 66 FRII. Use all observations to maximize available
precision; no unsupported statistical-power claim. Validation has already informed
model development; indices 31,35,50,51 were visually inspected in Phase 3A.

Engineering cohort: exactly 20 TRAIN examples per class. Within each class sort
the hexadecimal SHA-256 of ASCII
`Phase3B-XAI-v1.0.0|engineering|42|<dataset_index>` ascending, with index collision
tie-breaker, and take the first 20. Group FRI then FRII in the manifest. Validation
manifest order is ascending saved index. The archive's original train/test batches
are NOT the frozen research split: all saved validation indices remain eligible.
Locked research TEST pixels must never be retrieved, decoded, scored or visualized.
Test labels/filename metadata and indices may be inspected solely for provenance
and isolation. No train/validation/test split regeneration.

## Provenance, manifests and interface boundary

The YAML pins checkpoint and split SHA-256, archive SHA-256 and MD5, source record,
payload ordering/counts and protocol. Read/hash bytes before parsing. Verify all ten
batch/meta members against the archive's own manifest, duplicate/path/layout checks,
88 label/filename pairs per batch, binary labels and filename format. Inspect pickle
opcodes only: never run pickle GLOBAL/REDUCE instructions, instantiate NumPy arrays,
decode pixel payloads, or create images for metadata operations. Unsupported metadata
encodings fail closed. No automatic extraction, repair or download.

Preserve filenames such as `BestCat_proc/Cropped/100_159.613_+041.815.png`.
Candidate key: `filename-coordinate-fields:159.613|+041.815`. Remove directory,
extension and class-code prefix. Do not infer units, perform angular matching,
or claim verified astronomical identity. Metadata review found 792 unique exact
keys and no exact cross-split overlaps; this does not establish physical independence.
The current public loader drops filenames; the Phase 3B metadata reader preserves
them without modifying that loader.

Frozen dataclasses/tuples hold manifests. Each entry records protocol, index,
split, label, filename, candidate key, selection rule, split hash and dataset
provenance, including member/row mapping. Canonical serialization: JSON with sorted
keys, compact separators, ensure_ascii=true, allow_nan=false, UTF-8 and one terminal
newline. No timestamps, mutable nested dictionaries, or implicit manifest writes.
Identical checked inputs produce identical bytes and manifest digest. Serialization
is immutable-in-content, not a digital signature or OS permission setting.

`load_evaluation_data(root)` validates pins and returns 40-image engineering and
119-image validation manifests. `authorize(index, cohort)` checks TEST exclusion,
cohort membership and current provenance. `dispatch(index, cohort, operation, handler)`
authorizes BEFORE calling any retrieve/decode/score/visualize handler. Phase 3B.1
contains no image decoder or evaluation handlers. Consumers must use this boundary;
it cannot police unrelated Python code or malicious callbacks bypassing the interface.
Revalidate files on authorized access to detect drift. Downstream implementations
must consume checked provenance and avoid check/use races; they must not expose a
combined underlying dataset through this API.

## Explanation and preprocessing

Frozen ResNet and existing validation preprocessing: grayscale 150x150 float32,
ToTensor and Normalize(mean=0.5,std=0.5), no augmentation, cropping or contrast
remapping. No public-upload guard applied to trusted samples. Preserve public OOD
behavior and calibration. Grad-CAM target `blocks.2.convolution.3`, pre-softmax
logit gradients, native 19x19 positive CAM; bilinear 150x150 interpolation with
align_corners=false. Retain existing core normalization/degeneracy policy unchanged.
For every image freeze c=argmax(z(x)); use c for ALL interventions even after flips.

## Deletion and faithfulness

Scores: p_c=softmax(z)_c (primary), z_c and m_c=z_c-z_(1-c) (secondary).
For ranking r and deletion step k:

    AOPC_s,K(r) = (1/K) sum[k=1..K] (s(x)-s(x_deleted(r,k))).

Fractions f_k=k/20, k=0..20; PRIMARY K=6 (through 30%). Primary contrast per image:
CAM probability AOPC minus mean of exactly 20 random probability AOPCs. Second
contrast: CAM minus brightness. Full 0..100% curves and trapezoidal deletion AUC
are secondary: sum[k=1..20] ((D[k-1]+D[k])/2)*(f[k]-f[k-1]). Retain negative drops.

Tile boundaries floor(150*j/19), j=0..19, define 361 disjoint image-coordinate
tiles, not exact receptive fields. Rank tiles by raw native CAM, once before
deletion. Brightness ranks mean original tile intensity. Pixel budget=floor(22500*f),
so 5%=1125 and 30%=6750. Use nested masks, row-major pixels within each tile,
and truncate the last tile to meet the exact budget. Same policies for all controls.

Exactly 20 random permutations/image; seeds:
1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,
1010,1011,1012,1013,1014,1015,1016,1017,1018,1019.
RNG=NumPy Generator(PCG64(SeedSequence([seed,dataset_index]))). Reuse each permutation
for all replacement policies and scores. One CAM and one brightness ranking only;
NO extra 20-fold CAM/brightness tie averaging. Equal scores sort by ascending SHA-256
of `Phase3B-XAI-v1.0.0|tile-tie|42|<dataset_index>|<tile_id>`, then tile ID. Record ties.

Replace original intensities u in [0,1], then use unchanged normalization.
Primary replacement: median outer five-pixel border. Predefined sensitivities:
black u=0 (normalized -1), and blurred original image, Gaussian sigma=3 pixels,
radius=9, reflect padding. No choosing favorable baselines. Positive AOPC indicates
score reduction, negative indicates increase; zero may reflect redundancy/saturation
or baseline sensitivity, not proof of failure. Matched controls matter.

Deletion creates OOD artifacts: emphasize <=30%, compare matched budgets and all
predefined replacements, record frozen OOD diagnostics without filtering rejected
perturbations. Research classifier curves do not imply public OOD acceptance.
No thresholds change; no ROAR/retraining-equivalence claim.

## Stability and robustness

PRIMARY Jaccard uses top 10%; sensitivities 5%,20%. Inverse-align continuous CAM
before ranking; restrict both maps to common valid support V, excluding padding
and incomplete interpolation footprints. Select exactly ceil(q*|V|) pixels each.
J=|M_original intersect M_transformed|/|M_original union M_transformed|.
Primary grid 150x150; secondary aligned/pool grid 19x19. Exclude comparisons with
<80% valid support. Pixel tie key is SHA-256 of
`Phase3B-XAI-v1.0.0|pixel-tie|42|<dataset_index>|<original_coordinate_pixel_id>`,
then pixel ID. Record boundary ties. Pixels are not independent observations.

Each transformation starts from the original image: rotations -5,-2,+2,+5 degrees;
sensitivity -15,+15; separate exact horizontal and vertical flips; Gaussian noise
sigma=0.005,0.01 with seeds 2000..2009 (inclusive) per level, NumPy PCG64 with
SeedSequence([seed,index]); brightness gains 0.98,1.02 secondary only. Rotations are
bilinear, centered, 150x150 output, border-median fill. Clip intensity perturbations
to [0,1], record clipped fractions, preserve normalization. Brightness is display
sensitivity, NOT calibrated radio-flux robustness. No accumulated transforms.

Report delta p_c, delta m_c and class-flip indicator. Report all valid pairs,
prediction-preserving and flipped strata, ground-truth classes and correctness.
Low similarity with stable outputs differs from model instability. Stable maps
alone can be uninformative and do not imply faithfulness.

## Intensity-based emission proxy

b=median(border); sigma=1.4826*median(abs(border-b)). Define
E=1[u>b+max(a*sigma,d*(max(u)-b),1/255)]. Primary (a,d)=(3,0.05);
sensitivities (2,0.02),(5,0.10). This is a processed-intensity proxy, not instrumental
noise estimation. No CAM-guided dilation, manual edits or component deletion.
Flags: empty/full masks, zero border dispersion, border-touching emission.

Use unnormalized positive upsampled CAM U:
emission mass=sum(U*E)/sum(U); background mass=1-emission mass;
area=sum(E)/22500; enrichment=emission mass-area. Secondary: top-10% CAM/emission
IoU. Controls: B=max(u-b,0), uniform attribution, and 20 seeded uniform random native
maps (deletion seed list; PCG64(SeedSequence([seed,dataset_index,1])); never consumed by 3B.1).
Report all masses and area together. Zero/full masks and degenerate CAMs yield
unavailable alignment. No claim of validated jet/lobe/core/hotspot localization.
Stronger claims require independent expert annotations, preferably two astronomers
blinded to CAMs, source context and appropriate calibrated imaging.

## Model parameter randomization

Disposable in-memory copies only. Cumulative order: classifier.4, classifier.1,
blocks.2 including shortcut, blocks.1, blocks.0, stem. Seeds 3000..3004 inclusive;
reset Conv/Linear weights/biases with original initialization and included BatchNorm
affine scale=1,bias=0. Preserve ALL frozen BatchNorm running_mean, running_var and
num_batches_tracked in every copy/stage. Never call a reset that changes buffers.
Evaluation mode, fixed original class, no optimizer or randomized checkpoint writes.
No optional buffer-reset experiment; this supersedes the earlier draft.

Compare top-10% Jaccard, Spearman correlation with average tie ranks, and spatial
TV=0.5*sum(abs(U/sum(U)-V/sum(V))). Constant/degenerate maps make comparisons
unavailable. Controls: unchanged copy, repeated frozen CAM, brightness and random
maps. No required monotonic decrease. With learned BN buffers retained, the final
stage is NOT a wholly fresh random network. Parameter sensitivity is necessary
evidence, not proof of faithfulness; no label-randomization training.

## Statistical analysis and failures

Unit=image; present exact candidate-source groups are singleton, not proof of
physical independence. Average repeated random/noise/randomization measurements
within image and setting first; retain raw measurements. Summaries: mean, median,
IQR, valid/failure counts; FRI/FRII separately; sample-weighted overall and equal-class
macro means; correct/incorrect strata. Small subgroups receive cautious descriptions.

95% intervals: 10000 paired class-stratified image percentile bootstrap replicates,
seed=42, percentiles 2.5,97.5. Keep all measurements for a sampled image together.
Metric-specific complete-case intervals disclose conditional denominator/exclusions.
No inferential subgroup result if a class is absent or fewer than two independent
units exist. Verified physical duplicates require a reviewed source-cluster amendment,
not regenerated splits. Intervals do not cover retraining/survey variation.

Only two inferential hypotheses: CAM-minus-random and CAM-minus-brightness primary
probability AOPC30 contrasts. Two-sided paired image sign-flip, 10000 permutations,
seed=43, equal-class macro-mean statistic, plus-one Monte Carlo p=(1+# as/extreme)/(1+B).
Assumes paired-null sign exchangeability/symmetry. Holm across these TWO hypotheses,
familywise alpha=0.05. All other comparisons descriptive. Curve intervals pointwise,
not simultaneous; no independent significance tests at every fraction/subgroup.

Degenerate original CAM: retain image/manifest, attribution metrics unavailable,
never substitute random ranking. One/both transformed maps degenerate: J unavailable,
never assign 1 to both-empty; count transitions. Empty/full emission mask: unavailable
with reason. Flips retain original target. Perturbed OOD inputs remain in research
classifier curves with diagnostics. Nonfinite outputs invalidate affected metrics;
missing deletion step invalidates entire affected AOPC. Unexpected data error stops
run, never substitutes images. Provenance/state drift aborts and invalidates run.
All 119 contribute coverage reporting; no zero imputation or favorable-case removal.

## Cost, storage, limitations and future execution

Planning only: primary deletion 119*22*6=15708 forwards; full three-replacement
curves 119*22*20*3=157080 before caching. Original+30 transformed CAMs/image=3689;
six-stage/five-seed randomizations=3570; about 7259 CAM calls plus controls.
OOD diagnostics may add forwards. Runtime unmeasured. Stream images, start inference
batch=8, CAM individually, one disposable model; starting RAM budget 2 GB, verify
only during authorized engineering. All full maps ~1.3 GB; all deletion images >14 GB,
so retain native maps/scalars/seeds and reconstruct images. Do not execute cost probes now.

Future outputs isolated under outputs/xai/phase3b_v1.0.0; no overwrites; include
protocol/config/software versions, hashes, selection manifests, seeds, failures and
per-image measurements. No metric code is implemented in 3B.1.

Limitations: coarse positive-only main-branch CAMs; interpolation creates no evidence;
softmax saturation/common logit shifts; replacement distribution shift; padding,
clipping and ties; threshold dependence; uncalibrated intensity and uncertain source
identity; validation reuse; small error groups; preserved BN-statistic conditioning.
Do not assert causality or physical correctness from any metric or attractive map.

Future order (separate authorization required): data safeguards/tests; deletion and
analytic tests; alignment/robustness; emission proxies; disposable randomization;
paired statistics; orchestration; fixed 40-image engineering; separately authorized
119-image validation run; complete report. Any later locked-test XAI reporting needs
separate approval and does not change already reported classification test metrics.

## Primary references

- Grad-CAM: https://arxiv.org/abs/1610.02391
- Region perturbation/AOPC: https://arxiv.org/abs/1509.06321
- Deletion/insertion: https://arxiv.org/abs/1806.07421
- Distribution-shift/ROAR distinction: https://arxiv.org/abs/1806.10758
- Stability: https://arxiv.org/abs/1806.08049
- Input invariance: https://arxiv.org/abs/1711.00867
- Parameter sanity checks: https://arxiv.org/abs/1810.03292
- Broader MiraBest context (not a replacement for local payload): https://arxiv.org/abs/2305.11108
- Original FR morphology: https://doi.org/10.1093/mnras/167.1.31P
