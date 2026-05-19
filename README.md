# Liquid-Cell TEM Particle Pipeline

End-to-end automatic analysis pipeline for liquid-cell transmission electron microscopy (LC-TEM) videos:

```
raw AVI → drift correction → denoising → segmentation → tight particle crop
```

This repository started as a fork of [UDVD-MF-Denoising](https://github.com/sreyas-mohan/udvd) (Science 2025) but has grown into a full pipeline that supports two self-supervised video denoisers — **[UDVD-MF](https://github.com/sreyas-mohan/udvd)** and **[UMVD](https://github.com/maryaiyetigbo/UMVD)** — and adds Gatan-DigitalMicrograph-inspired drift correction plus SAM 3 segmentation.

In our experiments **UMVD produces visually cleaner results on LC-TEM data**, so it is the default in the end-to-end pipeline.

---

## Quick start

```bash
# One command — drift correct + denoise (transferred weights) + segment + 4× crop
python pipeline.py --video /path/to/video.avi
```

Outputs land in `/shared/jingchl6/material/lc-research/test/<video_name>/`:

| File | Description |
|------|-------------|
| `tracked_*.tif/.mp4` | Drift-corrected, particle-centered 512×512 stack |
| `denoised_*.npy/_comparison.mp4` | UMVD-denoised stack + raw\|denoised side-by-side |
| `segcrop_*.tif/.mp4/_4x.mp4` | SAM 3 mask + tight 256×256 crop + 4× sped-up MP4 |
| `pipeline.log` | Every command executed |

Optional flags:

| Flag | Purpose |
|------|---------|
| `--finetune` | Fine-tune UMVD 5 epochs on this video before denoising (~30 min, better quality) |
| `--gpu <N>` | Pick a CUDA device (default 0) |
| `--tcx --tcy` | Manually specify template center (override auto-detect) |
| `--shrink <0–1>` | Crop tightness (default 0.6) |

---

## How it works (4 stages)

### 1. Drift correction — `track_generalized.py`
- Auto-detects the particle in frame 0 by bandpass-NCC against a reference template extracted from a known good video (defaults to the 053243 chip).
- Tracks the particle frame-by-frame using NCC on bandpass-filtered images — this preprocessing is the key trick from Gatan DigitalMicrograph's `ImageAlignment.dll` (cross-correlation alone fails; bandpass surfaces the nanoparticle lattice while suppressing noise).
- Skips blurry transition frames (detected by Laplacian-variance drop).
- Outputs a 512×512 crop centered on the tracked particle each frame.

### 2. Denoising — two options
**Default (recommended, fast):** transfer-learned inference with frozen UMVD weights trained on a related video.
```bash
python /home/jingchl6/.local/UMVD/inference_lc.py \
    --data tracked.tif --weights umvd_ts900/best_model.pth --output denoised.npy
```
Takes about 1 minute for ~1000 frames.

**Fine-tune (better, slower):** 5 epochs starting from the same weights — usually matches a from-scratch 25-epoch training in quality.
```bash
python /home/jingchl6/.local/UMVD/train_lc.py \
    --data tracked.tif --output umvd_finetune \
    --init-weights umvd_ts900/best_model.pth \
    --num-epochs 5 --batch-size 8 --image-size 128 --patience 0
```

UDVD-MF is also available via `denoise_mf.py` / `denoise_inference.py` but produces visibly more over-smoothing on this data.

### 3. Segmentation — `seg_crop_speedup.py`
- Runs [SAM 3](https://github.com/facebookresearch/sam3) in image mode with a central box prompt `[0.5, 0.5, 0.4, 0.4]` on each denoised frame.
- Keeps only frames with a plausible mask (area 2k–80k px, score ≥ 0.3) — drops blurry / mis-tracked frames automatically.
- Crops a square around the mask centroid (shrink factor, default 0.6) and resizes every kept frame to 256×256 — the particle fills the output.
- Writes a 4× sped-up MP4 with libx264 (ffmpeg from `imageio_ffmpeg`).

### 4. Outputs
Everything saved alongside the input video, plus a `pipeline.log` that reproduces the run.

---

## Repository layout

### Pipeline orchestration
| Script | Purpose |
|--------|---------|
| `pipeline.py` | End-to-end runner — calls steps 1→4 automatically |
| `track_generalized.py` | Stage 1: bandpass-NCC drift correction (template + offset) |
| `denoise_inference.py` | Stage 2: UDVD-MF inference with frozen weights |
| `seg_crop_speedup.py` | Stage 3+4: SAM 3 segmentation + tight crop + 4× speedup |

### Stage-specific helpers
| Script | Purpose |
|--------|---------|
| `drift_correction.py` | Standalone pystackreg-based drift correction (used early in development; superseded by `track_generalized.py`) |
| `track_ts400.py`, `track_ts900.py` | Hard-coded template sizes used during method development |
| `segment_sam3.py` | SAM 3 only (overlay video) — older, kept for reference |
| `crop_mask.py`, `crop_mask_tight.py` | Cropping helpers used before `seg_crop_speedup.py` consolidated them |
| `denoise_mf.py` | UDVD-MF training (from upstream Science 2025 repo) |
| `denoised_to_video.py` | Convert any denoised .npy to a side-by-side comparison MP4 |
| `visualize_denoise.py` | Static side-by-side plot helper |

### Presentation / reference
| File | Purpose |
|------|---------|
| `build_ppt.py` | Builds the slide deck from extracted frames |
| `pipeline_presentation.pptx` | 11-slide deck explaining the pipeline |
| `ImageAlignment.dll` | Reverse-engineering reference for the bandpass-NCC approach |
| `check_fig1.png`, `check_fig2.png` | Paper reproduction sanity checks |

### Model / utility code (from upstream UDVD-MF)
| Folder | Purpose |
|--------|---------|
| `models/` | UDVD-MF model definitions |
| `utils/` | Loss, metrics, progress bars |
| `data.py` | Dataset loaders |
| `calculate_corr.py` | Temporal correlation analysis |

---

## Environment

Two conda environments are used. Both are needed for the end-to-end pipeline.

```bash
# UDVD-MF environment (also hosts our pipeline scripts)
conda env create -n denoise-HDR -f environment.yaml

# UMVD environment — clone https://github.com/maryaiyetigbo/UMVD
git clone https://github.com/maryaiyetigbo/UMVD ~/UMVD
conda create -n umvd python=3.11 -y
conda activate umvd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tifffile pytorch-msssim scikit-image tqdm matplotlib h5py imageio pandas opencv-python-headless

# SAM 3 environment (Python ≥3.12, PyTorch 2.6+)
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/facebookresearch/sam3 ~/sam3
cd ~/sam3 && pip install -e .
pip install einops decord pycocotools psutil "numpy<2" tifffile matplotlib opencv-python-headless scikit-image
```

`pipeline.py` hardcodes the conda environment Python paths at the top of the file; edit those constants if your install differs.

---

## Citations

- **UDVD-MF** (`denoise_mf.py`, `models/`): Mohan et al., "Visualizing nanoparticle surface dynamics and instabilities enabled by deep denoising", Science 2025.
- **UMVD**: Aiyetigbo et al., "Unsupervised microscopy video denoising", CVPR 2024 Workshop. https://github.com/maryaiyetigbo/UMVD
- **SAM 3**: Meta AI, 2025. https://github.com/facebookresearch/sam3
- **DigitalMicrograph / Gatan ImageAlignment** — bandpass-NCC drift correction approach inspired by reverse-engineering `ImageAlignment.dll` (commercial software, not redistributed).
