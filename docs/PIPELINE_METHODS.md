# Methods — An end-to-end pipeline for atomic-resolution phase analysis of LC-TEM nanoparticle videos

*Paper-ready description of the pipeline implemented in this repository.*
Every number below is taken from the code and from the run logs under
`/shared/jingchl6/material/lc-research/`; the source file implementing each step is
named in the margin so the text can be checked against the implementation.

**Figures**
- Figure 1 — [full pipeline](figures/fig1_pipeline.svg) ([PNG](figures/fig1_pipeline.png))
- Figure 2 — [classifier architecture](figures/fig2_classifier.svg) ([PNG](figures/fig2_classifier.png))

**Bibliography** — [`references.bib`](references.bib); per-stage citation guide in §8.

---

## 0. Overview

We analyse in-situ / liquid-cell transmission electron microscopy (LC-TEM) movies of
metal nanoparticles and produce, for every frame of the movie, a crystallographic phase
label (decahedral, FCC, icosahedral) together with the time windows over which the
particle occupies each phase. The raw data are far too noisy for this to be done
directly: at the dose rates required for millisecond time resolution the lattice fringes
are close to the noise floor, and the particle simultaneously translates across the field
of view because of stage and beam-induced drift.

The pipeline therefore consists of four stages, applied in sequence
(**Fig. 1**):

| # | Stage | Purpose | Method | Implementation |
|---|-------|---------|--------|----------------|
| 1 | Drift correction | Remove stage/beam drift so the temporal axis carries only physical change | TurboReg translation registration on band-pass–filtered frames, with moving-average registration and two-stage trajectory smoothing | `drift_correction.py`, `track_generalized.py` |
| 2 | Denoising | Recover lattice fringes from a low-dose movie without any clean reference | UMVD — self-supervised **blind-frame** temporal-interpolation denoiser | `UMVD/train_lc.py`, `UMVD/inference_lc.py` (UDVD-MF alternative: `denoise_mf.py`) |
| 3 | Segmentation | Produce a tight, scale-normalised, particle-filling crop | SAM 3 with a central box prompt, mask-centroid square crop | `seg_crop_speedup.py` |
| 4 | Classification | Assign a crystallographic phase per frame and build a phase timeline | Swin-Tiny transformer + 8-fold dihedral TTA + temporal median smoothing | `classifier/train.py`, `classifier/classify_video.py` |

Stages 1–3 are chained by `pipeline.py` for a single isolated particle; stage 4 is run on
the resulting crop stack.

**Input data.** Movies are 8-bit AVI, 2048 × 2048 px. Two acquisition families were used:
three room-temperature PtRu/TiO₂ series at 5 fps (866, 959 and 1364 frames) and a
temperature series (100 °C, 200 °C, 300 °C) at 20 fps (1489–1980 frames). The
classifier's training corpus is a separate set of 12 116 hand-labelled 512 × 512 Ag
particle images (§5.1).

---

## 1. Stage 1 — Drift correction

Two variants are implemented; which one is appropriate is determined by the content of
the movie, not by a quality trade-off.

### 1.1 Whole-field registration (default; `drift_correction.py`)

Used for in-situ heating series and for any field of view containing several particles or
an aggregate that restructures, where no single object persists across the movie.

**Band-pass pre-filter.** Registration is *not* performed on the raw frames. Raw
cross-correlation on HRTEM data is dominated by the low-frequency illumination gradient
and by shot noise, and fails (registration collapses to a random walk). Each frame is
instead reduced to a difference-of-Gaussians band,

  `B(I) = normalize( G_{σ=1.5} * I − G_{σ=15} * I )`,

implemented with `cv2.GaussianBlur` at kernel sizes `6σ|1`, then min–max normalised to
[0, 255]. This is the band-pass step used by Gatan DigitalMicrograph's alignment
routines; it suppresses the illumination ramp and the high-frequency noise and leaves the
lattice and particle edges, which is what carries the registration signal.

**Registration.** Frames are optionally downscaled to `--proc-res` (1024 × 1024 in
production; a 4× reduction in registration cost with no measurable loss of translation
accuracy) and registered with pystackreg (a Python port of the TurboReg pyramidal
intensity-based registration algorithm) restricted to the **TRANSLATION** transform —
rotation and scaling are not physically meaningful for stage drift and only add
degrees of freedom for the optimiser to overfit noise with. Registration uses
`reference='first'`: every frame is aligned to frame 0. The alternative
(`reference='previous'`) accumulates registration error monotonically over ~10³ frames
and is not used. A `--moving-average N` option (N = 9 in production) averages N
consecutive band-passed frames before registering, which raises the effective SNR of the
registration target; empirically this is what removes the residual shake in the first
~20 s of a heating ramp, where the per-frame SNR is lowest.

**Two-stage trajectory smoothing.** The registration returns a shift trajectory
`(t_x(f), t_y(f))`. Two distinct failure modes appear in it and are treated separately:

1. *Spikes* — isolated frames (beam blanking, sudden defocus, a bubble crossing the
   field) where registration fails outright and reports a jump of 100–800 px. These are
   removed with a **median filter** of width 11 (clean data) to 31 (very noisy data).
2. *Jitter* — a few-pixel per-frame random error superposed on the true slow drift.
   This is removed with a **Savitzky–Golay filter** (window 51–71, polynomial order 2),
   which is a local least-squares polynomial fit and therefore preserves the slow
   monotonic drift trend that the median filter and a plain box filter would both
   distort.

The smoothed transforms are written back into the `StackReg` object and applied to the
**raw** (not band-passed) frames, so the band-pass affects only *where* to shift, never
the output pixel values.

**Effect.** On the noisiest video in the set (200 °C, 20 fps), the maximum single-frame
shift step falls from 250–870 px to 3–5 px. Genuinely fast sample motion — the first
~20 s of a heating ramp, when the specimen is physically moving — is real signal and is
deliberately retained by the choice of smoothing windows; the filters are tuned to remove
sub-second jitter, not second-scale motion.

Outputs: an aligned float32 TIFF stack, a per-frame shift table, a three-panel drift
diagnostic plot (x/y trajectory, drift magnitude, and the 2-D drift path), and a raw vs.
aligned side-by-side montage.

### 1.2 Single-particle template tracking (`track_generalized.py`)

Used when a single isolated particle is to be followed and cropped, as in `pipeline.py`.
A 900 × 900 px band-passed template is taken from frame 0 of a reference movie, and each
frame of the target movie is matched against it with normalised cross-correlation
(`cv2.TM_CCOEFF_NORMED`, i.e. zero-mean NCC, which is invariant to per-frame brightness
and contrast changes). The particle centre is the matched template centre plus a fixed
user offset, and a 512 × 512 crop is taken about it. Before tracking, blurred frames are
detected from the variance of the Laplacian: a frame is rejected when its sharpness drops
by more than 3 σ relative to the frame-to-frame sharpness difference distribution, which
removes defocus excursions and beam-instability frames from the output stack entirely.

This tracker is faster and yields a particle-centred crop directly, but it presupposes
that the particle resembles the reference template; on morphologically dissimilar samples
NCC → 0 and the track degenerates. `pipeline.py` auto-detects the template location in
frame 0 of a new movie by NCC-matching the reference template and reports the match score,
so a failed auto-detection is visible in the log rather than silent.

---

## 2. Stage 2 — Self-supervised denoising

Two self-supervised video denoisers are implemented. **UMVD is the production choice**;
UDVD-MF is retained as a baseline and as the upstream code base of this repository.
Neither requires a clean reference image, which is the essential property here: no
ground-truth noiseless HRTEM image of a fluxional nanoparticle surface exists, because
the object changes on the same timescale as the acquisition.

### 2.1 Why self-supervision works here

Both denoisers rest on the same statistical argument. If the noise is independent
across the masked coordinate and zero-mean conditioned on the signal, then a network
trained to predict a *withheld* noisy measurement from the *surrounding* measurements
minimises its mean-squared error at the conditional expectation of the withheld
measurement — that is, at the clean signal, since the noise contributes an additive
constant to the loss that the network cannot reduce. This is the Noise2Noise /
blind-spot argument. The two denoisers differ in which coordinate is masked.

- **UDVD-MF** masks a *spatial* coordinate: the receptive field of every output pixel
  excludes the pixel itself (and, by default in this fork, its immediate neighbours).
- **UMVD** masks a *temporal* coordinate: the network sees a window of frames with the
  centre frame zeroed out, and must interpolate it.

The temporal mask is the better fit for this data. The spatial blind spot discards
information at exactly the length scale of interest (a single lattice fringe is 2–3 px),
whereas the temporal mask discards a whole frame but keeps full spatial resolution in the
neighbouring frames. On our LC-TEM movies UMVD is visibly cleaner and retains fringe
contrast that UDVD-MF smooths away.

### 2.2 UMVD architecture and blind-frame mechanism (production)

*Implementation: `UMVD/model.py` (`Denoiser`), driven by `UMVD/train_lc.py` and
`UMVD/inference_lc.py`.*

The input is a temporal window of `n_frames = 7` grayscale frames
`{I_{t−3}, …, I_t, …, I_{t+3}}`, stacked along the batch axis as (7, 1, H, W); at the
sequence boundaries the index is clamped, so the first and last three frames are handled
by replication rather than being dropped.

1. **Per-frame feature extraction.** Three depthwise 3 × 3 convolution blocks
   (`groups = in_channels`, replication padding, ReLU) lift each frame independently to
   `filters = 21` channels. Because the convolutions are depthwise and applied per frame,
   no cross-frame information mixes at this stage — which is what makes the subsequent
   masking exact.

2. **Blind-frame temporal filter.** A fixed, non-learned weight vector
   `w ∈ R^7` multiplies the feature maps frame-by-frame. The weights come from
   `temporalfilter(n_frames, mid, level, minv)`, evaluated with `level = 64`, which gives

   `w = [1, 1, 1, 0, 1, 1, 1]`,

   i.e. a **hard mask that zeroes the centre frame** while passing the six neighbours
   unattenuated. (`level = 1` gives the soft triangular ramp `[1, ⅔, ⅓, 0, ⅓, ⅔, 1]` of
   the original formulation; the production configuration uses the hard mask.) The centre
   frame therefore contributes nothing to the prediction of itself — the network is blind
   to it — and the self-supervised argument of §2.1 applies exactly.

3. **U-Net.** The masked per-frame features are reshaped into a single tensor of
   `7 × 21 = 147` channels and passed through a small U-Net: three encoder blocks
   (147→48, 48→48, 48→48 with the last not down-sampling; each block is five 3 × 3
   conv+ReLU layers, 2 × 2 max-pool between blocks), then two decoder blocks with nearest-
   neighbour up-sampling, size-matched padding, and skip concatenation from the
   corresponding encoder level (and from the 147-channel input at the top level).

4. **Output head.** Three 1 × 1 convolutions, 96 → 384 → 96 → 1, with ReLU between,
   produce the denoised centre frame.

The model has **1 786 191 parameters** (≈1.8 M) — small enough to fine-tune on a single
movie in minutes.

**Training objective.** The target is the *noisy* centre frame of the same window:

  `L = ‖ f_θ(I_{t−3..t+3} ⊙ w) − I_t ‖² / 2`   (summed over pixels, divided by batch size)

No clean data are used anywhere. Adam, lr = 1e-3, halved every 10 epochs; 128 × 128
patches with stride 64 (stride 128, i.e. non-overlapping, for 1024² whole-field frames to
keep the patch count tractable); batch size 4–8; up to 25 epochs with early stopping
(patience 5) on a validation loss computed on full frames of the same movie. Random seed
44 throughout.

**Two deployment modes.**
- *Transfer / frozen weights* — a model trained on one movie is applied unchanged to a new
  one (`inference_lc.py`). ≈1 min per 1000 frames.
- *Fine-tuning* — 5 epochs starting from those weights, at lr = 5e-4 (`train_lc.py
  --init-weights`). ≈30 min.

On this data transfer, fine-tuning, and training from scratch give near-identical output
quality; the transferred model is therefore the default and fine-tuning is an opt-in
(`pipeline.py --finetune`).

Inference is per-frame with the same 7-frame sliding window and boundary clamping. The
stack is normalised by its global maximum before the network and rescaled afterwards, so
the output is in the original intensity units. The result is saved as float32 `.npy`.

*Known failure mode.* Fine-tuning occasionally diverges and saves a degenerate all-zero
model, producing a black output. This is a training instability, not a data problem; the
run should be repeated. A non-zero check on `denoised.npy` is the appropriate guard.

### 2.3 UDVD-MF baseline

*Implementation: `denoise_mf.py`, `models/blind_spot_net.py`, `models/blind-video-net-4.py`.*

UDVD-MF is the unsupervised deep video denoiser of Sheth et al., as extended by the
Crozier group for multi-frame TEM data. The blind spot is constructed geometrically
rather than by masking: every convolution is preceded by a one-row zero-pad
(`ZeroPad2d((0,0,1,0))`) and followed by a one-row crop, so each layer's receptive field
is strictly *above* the output pixel; the input is then processed in four 90° rotations
and the four half-plane receptive fields are recombined by 1 × 1 convolutions
(384 → 384 → 96 → n_out). The result is a network whose receptive field excludes the
centre pixel exactly, by construction rather than by architecture-specific masking.
This fork additionally blinds the immediate neighbours by default (`--include-neighbor`
disables this), which matters when the noise has short-range spatial correlation, and
supports 4-D datasets and `.npy` input. Default training is 50 epochs on 256 × 256
patches with batch size 1.

We report UDVD-MF as the baseline; UMVD's temporal masking is what we use for the
results in this work.

---

## 3. Stage 3 — Segmentation and tight cropping

*Implementation: `seg_crop_speedup.py`.*

Single-particle classification requires the particle to fill a consistent fraction of the
frame; otherwise the classifier learns particle *size in pixels*, which is an artefact of
the crop, rather than the lattice signature.

Each denoised frame is min–max normalised to 8-bit, replicated to RGB, and passed to the
**SAM 3** image model with a single geometric box prompt at the frame centre,
`[cx, cy, w, h] = [0.5, 0.5, 0.4, 0.4]` in normalised coordinates. Of the returned mask
candidates we keep the highest-scoring one whose area falls inside
`[--min-area, --max-area]` (5 000–600 000 px in production) and whose score exceeds
`--conf-thresh = 0.3`; frames with no admissible mask are dropped from the output stack.

For a kept frame, the mask bounding box is converted to a square of side
`max(Δx, Δy) × --shrink` (0.55–0.6), centred on the **mask centroid** rather than on the
bounding-box centre — the centroid is far more stable frame-to-frame when the mask
boundary flickers. The square is clamped to the frame, the crop is taken from the
denoised (not normalised) stack, and resized to a uniform 256 × 256 with area
interpolation. Outputs are the crop stack (TIFF), a 5 fps native-speed MP4, a 4× sped-up
H.264 MP4 for viewing, and a per-frame metadata CSV recording keep/drop, mask area, score
and centroid.

*Scope.* The fixed central box prompt is correct for a single centred particle and
wanders on aggregates, where there is no single persistent object to lock onto. For
aggregate movies the segmentation stage is skipped, or the prompt is replaced by a point
or text prompt.

---

## 4. Stage 4 — Crystallographic phase classification

### 4.1 Class definition

Ag nanoparticles in this size range occupy three structural motifs: **decahedral (Dh)**,
**face-centred cubic (FCC)**, and **icosahedral (Ih)**. The raw annotation additionally
contains an **Ih→Dh** transition class.

Single frames of `Ih` and `Ih→Dh` are **not separable**. Both show a multiply-twinned
projection whose difference is a partial rearrangement that is only identifiable in
context — i.e. from the *trajectory*, not from one image. Training a 4-class model
confirms this quantitatively: 89.3 % overall, with essentially all of the residual error
concentrated on the Ih ↔ Ih→Dh boundary. We therefore **merge Ih and Ih→Dh into a single
`Ih`-family class** for the frame-level model (3 classes) and recover the transition
afterwards from the temporal structure (§4.4). This is a modelling decision about what a
single frame can carry, not a relabelling of the physics.

### 4.2 Backbone and preprocessing

*Implementation: `classifier/train.py`, `classifier/dataset.py`; architecture in **Fig. 2**.*

The classifier is **Swin-Tiny** (`swin_tiny_patch4_window7_224`, timm), ImageNet-1k
pre-trained, with the head replaced by a 3-way linear layer: **27.5 M parameters**,
patch size 4, window size 7, embedding dimension 96, depths (2, 2, 6, 2), stage
dimensions (96, 192, 384, 768), heads (3, 6, 12, 24), input 224 × 224.

A shifted-window transformer is a good fit for this problem: attention is computed inside
7 × 7 windows and the windows are shifted between consecutive blocks, so the model builds
long-range relations across the particle (needed to see a five-fold twin) while retaining
the locality that lattice-fringe evidence lives at. Swin-Base was also evaluated and is
*not* better here (95.55–95.72 % vs. 95.88 %); the dataset is too small to support the
larger model.

Grayscale input is replicated to three channels and ImageNet-normalised
(mean 0.485/0.456/0.406, std 0.229/0.224/0.225), which lets the pre-trained weights be
used unmodified.

*Training augmentation* — `RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1))`,
random horizontal and vertical flips, a random choice of exact 0°/90°/180°/270° rotation,
and colour jitter (brightness 0.2, contrast 0.2). The rotations are exact multiples of
90° rather than arbitrary angles, because arbitrary rotation requires interpolation and
interpolation destroys lattice fringes. A `strong` preset adds RandAugment(2, 9), heavier
jitter and RandomErasing(p = 0.25).

*Evaluation transform* — `Resize(256) → CenterCrop(224)`, no augmentation.

### 4.3 Training protocol

**Particle-level splits.** Each labelled particle contributes ~2 consecutive frames
(file indices pair up as (1,2), (3,4), (19,20), …). A per-image random split would place
the two frames of the same particle on opposite sides and leak particle identity into
validation. `dataset.py:build_splits(by_particle=True)` parses the trailing index out of
each filename, groups frames into particles, and splits **by particle**, stratified within
each class, 90/10, seed 44.

**Class imbalance.** The corpus is strongly imbalanced (§5.1: Ih outnumbers FCC 11:1).
Two mechanisms are combined (`--balance sampler+loss`): a `WeightedRandomSampler` with
inverse-frequency weights (power 1.0, fully equalising classes within an epoch) and an
inverse-frequency-weighted cross-entropy with label smoothing 0.05. A `sampler-sqrt`
(power 0.5) variant is available for gentler rebalancing.

**Warm start.** Training the final configuration directly from ImageNet weights falls
into the trivial imbalance minimum (predict the majority class). The production run is
warm-started from an earlier 3-class checkpoint (`--init-weights`), which loads all
backbone tensors whose names and shapes match and re-initialises the head; this is
essential to reach the reported accuracy.

**Optimisation.** AdamW, lr 1e-4 for the warm-started runs (3e-4 from scratch), weight
decay 1e-4, cosine annealing over the full run, batch size 32, 20–25 epochs, seed 44,
optional AMP. The checkpoint with the best validation accuracy is kept.

### 4.4 Inference: TTA, temporal smoothing, and transition recovery

*Implementation: `classifier/classify_video.py`.*

**8-fold dihedral test-time augmentation.** Each frame is evaluated under all eight
elements of the dihedral group D₄ (four 90° rotations × optional horizontal flip) and the
softmax outputs are averaged. This is principled rather than a generic trick: the
crystallographic phase of a particle is invariant under these transforms — a decahedron
rotated by 90° is still a decahedron — so the eight views are eight genuinely equivalent
observations of the same label, and averaging reduces variance without introducing bias.
Measured gain on the real particle-level validation set: **95.88 % → 96.62 %** (+0.74 pt),
for the cost of 8× inference and no retraining.

**Temporal median smoothing.** Each class-probability stream is median-filtered along
time with a window of 11 frames. The median (rather than a mean) suppresses isolated
single-frame flips — which arise from momentary defocus or a bad segmentation crop —
without blurring a genuine phase change, which is a step and which the median preserves.

**Segmentation of the timeline.** Consecutive frames sharing an argmax label are grouped
into segments; segments shorter than `--min-segment` (10 frames) are absorbed into their
neighbour, and adjacent same-label segments are then merged.

**Recovering the Ih→Dh transition.** The transition class that was deliberately merged out
at the frame level is recovered from two models run in parallel: the merged 3-class model
(Dh / FCC / Ih-family) supplies the segmentation, and a second model trained with the
transition frames *dropped* (`--mode drop-ihdh`, Dh / FCC / Ih) is queried inside each
`Ih`-family segment. Within such a segment a frame is called `Ih` if
`p(Ih) > 0.6 ∧ p(Ih) > p(Dh) + 0.2`, `Dh` under the mirror-image condition, and
**`Ih→Dh` otherwise** — i.e. the transition is defined as the interval in which the
Ih-vs-Dh evidence is genuinely mixed, which is exactly the physical content of the
transition. This uses the temporal context that a single frame lacks, and so recovers the
class that §4.1 showed to be undecidable per-frame.

Outputs: a per-frame probability CSV, a segment table with durations in seconds, a
three-panel timeline figure (merged-model probabilities, drop-model probabilities, final
segment timeline), and an overlay MP4 with the per-frame prediction and confidences burnt
into a header bar.

---

## 5. Data

### 5.1 Real labelled corpus

12 116 single-particle 512 × 512 TIFF images of Ag nanoparticles under varying sputtering
and gas conditions, hand-labelled into four classes:

| Class | Images | Train | Val |
|-------|-------:|------:|----:|
| Dh | 2 396 | 2 156 | 240 |
| FCC | 680 | 609 | 71 |
| Ih | 7 766 | 6 995 | 771 |
| Ih→Dh | 1 274 | 1 142 | 132 |
| **Total** | **12 116** | **10 902** | **1 214** |

Split by particle, stratified per class, 90/10, seed 44 (§4.3). In merged 3-class mode
the Ih and Ih→Dh rows are summed.

### 5.2 Physically simulated training data

*Implementation: `classifier/generate_synthetic.py`, atomic models in
`three_struct_tilt_series.py`.*

Real labels are scarce and expensive, so we additionally simulate HRTEM images from first
principles with **abTEM**, giving 5 000 training + 1 000 validation images per class
(18 000 total).

*Atomic models* (ASE cluster builders, element **Pt** — chosen so the lattice constant,
3.92 Å, brackets the PtRu test particles, whereas Au is 4.08 Å):
- `ih` — Icosahedron, 7 shells, aligned so a five-fold axis is along z;
- `deca` — Decahedron(5, 3, 2), aligned along its five-fold axis (found as the
  minimum-variance eigenvector of the position covariance);
- `fcc` — truncated Octahedron(length = 12, cutoff = 4), rotated 45° about x into the
  [110] zone axis.

*Randomisation per image* — independent rotations about x, y, z uniform on [0°, 90°];
defocus uniform on [50, 90] Å; dose uniform on [1500, 3500] e⁻/Å²; real-space sampling
uniform on [0.095, 0.105] Å/px (a ±5 % effective particle-size variation); and vacuum
padding uniform on [0.5, 25] Å, which makes the particle occupy roughly **40–95 % of the
frame**. That last term is the one that matters most for transfer: it reproduces the
variation in how much of the frame the particle fills in a real SAM 3 crop, and without it
the model learns the synthetic crop scale.

*Image formation* — multislice through an `abtem.Potential`
(`slice_thickness = 2 Å`, `projection = 'infinite'`, Kirkland parametrization) of a
300 kV plane wave, followed by a CTF with semi-angle cutoff 30 mrad, `Cs = −13 µm`,
focal spread 40 Å, and the sampled defocus. The intensity is Gaussian-blurred (σ = 0.6 px)
to represent the detector MTF, converted to electron counts via `I · dose · pixel_area`,
corrupted with **Poisson** shot noise, and given an additive Gaussian term (σ = 0.02) for
read noise. The result is area-resized to 256 × 256 and saved as 8-bit PNG.

Class indices are aligned with the real 3-class schema (`deca`→Dh = 0, `fcc`→FCC = 1,
`ih`→Ih = 2), so a synthetic-trained model can be applied to a real video with no
remapping.

*Use.* `--data synthetic` trains on simulation alone; `--data combined`
(`dataset_combined.py`) trains on real merged-3-class + synthetic (10 902 + 15 000 =
25 902 images) while **validating on real data only**, so the reported number is always an
honest measurement on real images. A `--real-boost` factor can up-weight real samples in
the sampler.

---

## 6. Results

All accuracies are on the **real** particle-level validation set (1 214 images) unless
stated otherwise.

| Configuration | Model | Data | Val accuracy |
|---|---|---|---|
| **Best — 3-class merged + 8-fold TTA** | Swin-Tiny, warm-started | real | **96.62 %** |
| 3-class merged, no TTA | Swin-Tiny, warm-started | real | 95.88 % |
| 3-class, Ih→Dh dropped | Swin-Tiny, warm-started | real | 96.58 % |
| 3-class merged | Swin-Base, warm-started | real | 95.55 % |
| 3-class merged, 50 epochs | Swin-Base | real | 95.72 % |
| 3-class merged, 50 epochs | Swin-Tiny | real | 95.72 % |
| **4-class (Ih→Dh separate)** | Swin-Base | real | **89.29 %** |
| Combined real + synthetic | Swin-Base | combined | 94.81 % (95.39 % + TTA) |
| Combined real + synthetic | Swin-Tiny | combined | 94.40 % (94.73 % + TTA) |
| Synthetic only, synthetic val | Swin-Base / Swin-Tiny | synthetic | 100 % / 99.97 % |

**Confusion matrix**, best merged 3-class model (rows = true, columns = predicted):

|  | Dh | FCC | Ih-family |
|---|---:|---:|---:|
| **Dh** (240) | 222 | 12 | 6 |
| **FCC** (71) | 4 | 67 | 0 |
| **Ih-family** (903 = 771 Ih + 132 Ih→Dh) | 24 | 4 | 875 |

Per-class recall 0.925 (Dh) / 0.944 (FCC) / 0.969 (Ih-family).

**Ensembling was tested and rejected.** Averaging the softmax outputs of the best model
with the two combined-data models gives at best 95.88 % — below the 96.62 % of the single
best model with TTA. The weaker members drag the mean down; there is no diversity benefit
to extract here.

**Synthetic-only training reaches 100 % on synthetic validation but does not transfer**
— the 100 % figure measures only that the simulation is internally separable. Mixing
synthetic with real (`combined`) lands at 94.4–94.8 %, below real-only training. The
useful conclusion is that the simulation is a valid *pre-training* signal (it produced
the warm-start checkpoints) but not a substitute for real labels.

**Denoising.** UMVD produces visibly cleaner output than UDVD-MF on this data, and
transfer ≈ fine-tune ≈ from-scratch within UMVD, so the frozen-weights transfer mode is
used by default.

**Drift correction.** On the noisiest heating video, the maximum single-frame shift step
falls from 250–870 px to 3–5 px under the full recipe (band-pass + moving-average 9 +
median 11–31 + Savitzky–Golay 51–71). The ablation script is
`drift_experiment_ma.py`, which measures residual consecutive-frame motion of the aligned
stack by phase cross-correlation on 150 sampled frame pairs, reported separately for the
early (unstable) and late (stable) parts of the movie.

---

## 7. Reproduction

```bash
# Stage 1 — whole-field drift correction
python drift_correction.py --input in.avi --output aligned.tif \
    --reference first --bandpass --proc-res 1024 \
    --moving-average 9 --median-window 11 --smooth-shifts 51

# Stage 2 — UMVD denoising (transfer)
python ~/.local/UMVD/inference_lc.py \
    --data aligned.tif --weights umvd_ts900/best_model.pth --output denoised.npy

# Stage 2' — UMVD fine-tuning (5 epochs)
python ~/.local/UMVD/train_lc.py --data aligned.tif --output umvd_finetune \
    --init-weights umvd_ts900/best_model.pth \
    --num-epochs 5 --batch-size 8 --image-size 128 --stride 128 --patience 0

# Stage 3 — SAM 3 segmentation + tight crop
python seg_crop_speedup.py --input denoised.npy --output-dir out --name segcrop \
    --min-area 5000 --max-area 600000 --shrink 0.6

# Stage 4 — train the classifier
python classifier/train.py --arch swin_tiny_patch4_window7_224 \
    --epochs 25 --batch-size 32 --lr 1e-4 --balance sampler+loss --mode merge-ihdh \
    --splits-json splits_particle/splits.json --init-weights <warmstart.pt> \
    --output runs/swin_merged

# Stage 4' — classify a video (8-fold TTA + temporal smoothing + transition recovery)
python classifier/classify_video.py --video segcrop.tif --output-dir out --gpu 0

# Stages 1-3 chained for a single isolated particle
python pipeline.py --video in.avi [--finetune]
```

Random seed 44 is fixed in the denoiser, the splits, and the classifier training.

---

## 8. What to cite, stage by stage

Full BibTeX entries: [`references.bib`](references.bib).

### Denoising stage — the citations that matter most

| Cite | For |
|---|---|
| **Aiyetigbo et al., CVPRW 2024** — *Unsupervised Microscopy Video Denoising* (arXiv:2404.12163) | **The denoiser actually used (UMVD).** Cite this for the blind-frame / deep temporal interpolation architecture, the temporal signal filter, and the self-supervised objective of §2.2. |
| **Sheth et al., ICCV 2021** — *Unsupervised Deep Video Denoising* (arXiv:2011.15045) | **UDVD**, the baseline denoiser and the origin of this repository's `models/` and `utils/` code. Cite for the rotated half-plane blind-spot construction of §2.3. |
| **Crozier et al., Science 387, 949–954 (2025)** — *Visualizing nanoparticle surface dynamics and instabilities enabled by deep denoising* (doi:10.1126/science.ads2688) | The application paper for UDVD-MF on TEM: unsupervised denoising enabling millisecond-resolution observation of fluxional nanoparticle surfaces. Cite as the scientific precedent for denoising low-dose TEM movies of nanoparticle surfaces. |
| **Marcos Morales et al., ICML 2023** — *Evaluating Unsupervised Denoising Requires Unsupervised Metrics* (arXiv:2210.05553) | How to evaluate a denoiser with no ground truth. Cite if you report any quantitative denoising metric. |
| **Lehtinen et al., ICML 2018** — *Noise2Noise* | The statistical argument of §2.1 (training against a noisy target recovers the conditional mean). |
| **Krull et al., CVPR 2019** — *Noise2Void*; **Laine et al., NeurIPS 2019** — *High-Quality Self-Supervised Deep Image Denoising* | Blind-spot self-supervision; Laine et al. is the source of the four-rotation half-plane receptive-field trick used in UDVD's `BlindSpotNet`. |

> **Short version for a methods paragraph:** the denoising stage uses UMVD
> (Aiyetigbo et al., CVPRW 2024), a self-supervised blind-frame video denoiser; UDVD
> (Sheth et al., ICCV 2021) is the baseline, and Crozier et al. (Science 2025) is the
> precedent for applying this class of denoiser to low-dose TEM movies of nanoparticles.

### Other stages

| Stage | Cite | For |
|---|---|---|
| Drift correction | **Thévenaz, Ruttimann & Unser, IEEE TIP 7, 27–41 (1998)** | TurboReg — the pyramidal intensity-based registration algorithm behind `pystackreg`. |
| Drift correction | **Savitzky & Golay, Anal. Chem. 36, 1627 (1964)** | The trajectory-smoothing filter. |
| Drift correction | Gatan DigitalMicrograph | The band-pass-then-cross-correlate alignment convention (software attribution, not a paper). |
| Segmentation | **Carion et al., arXiv:2511.16719 (2025)** — *SAM 3: Segment Anything with Concepts* | The segmentation model. Optionally also Kirillov et al., ICCV 2023 (SAM) and Ravi et al. (SAM 2) for lineage. |
| Classifier | **Liu et al., ICCV 2021** — *Swin Transformer* | The backbone. |
| Classifier | **Wightman, `timm` (2019)** | The implementation and the ImageNet-1k pre-trained weights. |
| Classifier | **Deng et al., CVPR 2009** — ImageNet | The pre-training corpus. |
| Classifier | **Loshchilov & Hutter, ICLR 2019 / ICLR 2017** | AdamW; SGDR cosine annealing. |
| Classifier | **Szegedy et al., CVPR 2016** | Label smoothing. |
| Classifier (optional aug) | **Cubuk et al., CVPRW 2020** | RandAugment, used in the `strong` preset. |
| Synthetic data | **Madsen & Susi, Open Res. Europe 1, 24 (2021)** — *The abTEM code* | HRTEM multislice simulation. |
| Synthetic data | **Kirkland, *Advanced Computing in Electron Microscopy*** | The Kirkland scattering-factor parametrization used by the potential. |
| Synthetic data | **Larsen et al., J. Phys. Condens. Matter 29, 273002 (2017)** — ASE | The cluster builders for the icosahedral / decahedral / octahedral models. |
| Framework | **Paszke et al., NeurIPS 2019** — PyTorch; **Virtanen et al., Nat. Methods 2020** — SciPy; **Bradski, 2000** — OpenCV | Standard tooling. |

---

## 9. Limitations

1. **Ih vs. Ih→Dh is not decidable from a single frame.** The 4-class model tops out at
   89.3 %; the transition class is recovered only through the temporal rule of §4.4, which
   inherits that rule's threshold choices (0.6 and 0.2).
2. **The template tracker requires morphological similarity to its reference.** On
   dissimilar particles the NCC score collapses and the track becomes a random walk. The
   auto-detect step reports the score, so this is detectable but not automatic.
3. **The SAM 3 central box prompt assumes one centred object.** It wanders on aggregates.
4. **Synthetic data does not substitute for real labels** (§6); it is useful for
   pre-training only.
5. **Fine-tuning the denoiser can diverge** to an all-zero model (§2.2).
6. **The classifier corpus is Ag** while the pipeline test movies are PtRu/TiO₂; the
   cross-material transfer has not been quantitatively validated on labelled PtRu data.
7. **Drift correction is translation-only.** Rotation and specimen distortion are not
   modelled.
