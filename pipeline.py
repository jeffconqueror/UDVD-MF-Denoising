"""
End-to-end LC-TEM particle pipeline.

Input:  raw .avi video (any name)
Output: a folder per input video with:
  - tracked_<name>.tif/.mp4         (drift-corrected, particle-centered 512x512)
  - denoised_<name>.npy/.mp4        (UMVD-denoised, transferred or fine-tuned)
  - segcrop_<name>.tif/.mp4/.4x.mp4 (SAM 3 mask + tight 256x256 crop)
  - meta.csv                        (per-frame status)
  - pipeline.log                    (this run's log)

Steps (all automatic):
  1. Auto-detect particle in frame 0 by NCC-matching the reference template
     extracted from a known good video (default: 053243).
  2. Drift correction with bandpass-NCC template tracking + blur frame skip.
  3. Denoising with the previously trained UMVD weights.
     If --finetune passed, fine-tunes 5 epochs on this video first.
  4. SAM 3 segmentation with central box prompt; tight square crop around
     mask centroid; resize to 256x256; 4x sped-up MP4.

Usage:
  python pipeline.py --video /path/to/video.avi
  python pipeline.py --video /path/to/video.avi --finetune
  python pipeline.py --video /path/to/video.avi --output-dir /elsewhere
"""
import argparse, os, sys, subprocess, time, shutil
from pathlib import Path

# Defaults — points to existing assets in this repo / shared storage
DEFAULT_REF_VIDEO   = '/shared/jingchl6/material/lc-research/PtRu TiO2 chip1 FS30202_20211106_053243.avi'
DEFAULT_REF_TCX     = 885   # template center in reference frame 0
DEFAULT_REF_TCY     = 630
DEFAULT_TEMPLATE_SZ = 900
DEFAULT_OFF_X       = 200   # particle = template-center + offset
DEFAULT_OFF_Y       = 250
DEFAULT_UMVD_WEIGHTS = '/shared/jingchl6/material/lc-research/test/053243/umvd_ts900/best_model.pth'
DEFAULT_OUT_ROOT    = '/shared/jingchl6/material/lc-research/test'

PY_DENOISE = '/home/jingchl6/miniconda3/envs/denoise-HDR/bin/python3'
PY_UMVD    = '/home/jingchl6/miniconda3/envs/umvd/bin/python'
PY_SAM3    = '/home/jingchl6/miniconda3/envs/sam3/bin/python'
SCRIPT_DIR = '/home/jingchl6/.local/UDVD-MF-Denoising'
UMVD_DIR   = '/home/jingchl6/.local/UMVD'


def run(cmd, log):
    print(f'[pipeline] $ {cmd}', flush=True)
    log.write(f'$ {cmd}\n'); log.flush()
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(r.stdout); log.flush()
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError(f'Step failed (rc={r.returncode}): {cmd}')
    return r.stdout


def auto_detect_particle(ref_video, ref_tcx, ref_tcy, ts, off_x, off_y, target_video):
    """Use bandpass-NCC of the reference template to locate the particle in target frame 0."""
    import cv2, numpy as np
    half = ts // 2

    def bandpass(img, low=1.5, high=15.0):
        img = img.astype(np.float32)
        kl = max(3, int(6*low) | 1); kh = max(3, int(6*high) | 1)
        return cv2.normalize(cv2.GaussianBlur(img, (kl, kl), low) - cv2.GaussianBlur(img, (kh, kh), high),
                             None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    cap = cv2.VideoCapture(ref_video); ret, f = cap.read(); cap.release()
    g = bandpass(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    ref = g[ref_tcy-half:ref_tcy+half, ref_tcx-half:ref_tcx+half]

    cap = cv2.VideoCapture(target_video); ret, f = cap.read(); cap.release()
    g2 = bandpass(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    res = cv2.matchTemplate(g2, ref, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    tcx = loc[0] + half
    tcy = loc[1] + half
    print(f'[pipeline] auto-detect: template center=({tcx},{tcy}) particle=({tcx+off_x},{tcy+off_y}) NCC={score:.3f}')
    return tcx, tcy, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='input AVI video')
    ap.add_argument('--output-dir', default=None,
                    help='output directory (default: {OUT_ROOT}/<video_basename>)')
    ap.add_argument('--name', default=None, help='label used in output filenames (default: video basename)')
    ap.add_argument('--finetune', action='store_true', help='fine-tune UMVD 5 epochs on this video before denoising')
    ap.add_argument('--ref-video', default=DEFAULT_REF_VIDEO)
    ap.add_argument('--ref-tcx', type=int, default=DEFAULT_REF_TCX)
    ap.add_argument('--ref-tcy', type=int, default=DEFAULT_REF_TCY)
    ap.add_argument('--template-size', type=int, default=DEFAULT_TEMPLATE_SZ)
    ap.add_argument('--off-x', type=int, default=DEFAULT_OFF_X)
    ap.add_argument('--off-y', type=int, default=DEFAULT_OFF_Y)
    ap.add_argument('--tcx', type=int, default=None, help='override auto-detected template x')
    ap.add_argument('--tcy', type=int, default=None, help='override auto-detected template y')
    ap.add_argument('--umvd-weights', default=DEFAULT_UMVD_WEIGHTS)
    ap.add_argument('--shrink', type=float, default=0.6)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    # Derive paths
    base = args.name or Path(args.video).stem.replace(' ', '_')
    out_dir = args.output_dir or os.path.join(DEFAULT_OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'pipeline.log')
    log = open(log_path, 'w')
    t0 = time.time()
    print(f'[pipeline] output dir: {out_dir}')
    log.write(f'pipeline started for {args.video}\n  output: {out_dir}\n  GPU: {args.gpu}\n')

    # ---------- Step 1: auto-detect particle ----------
    if args.tcx is not None and args.tcy is not None:
        tcx, tcy = args.tcx, args.tcy
        print(f'[pipeline] using user-specified template center ({tcx},{tcy})')
    else:
        tcx, tcy, score = auto_detect_particle(args.ref_video, args.ref_tcx, args.ref_tcy,
                                                args.template_size, args.off_x, args.off_y, args.video)
        log.write(f'auto-detected template ({tcx},{tcy}) NCC={score:.3f}\n')

    # ---------- Step 2: drift correction ----------
    tracked_path = os.path.join(out_dir, f'tracked_{base}')
    cmd = (f'CUDA_VISIBLE_DEVICES={args.gpu} {PY_DENOISE} -u {SCRIPT_DIR}/track_generalized.py '
           f'--video "{args.video}" --tcx {tcx} --tcy {tcy} '
           f'--off-x {args.off_x} --off-y {args.off_y} '
           f'--template-size {args.template_size} '
           f'--out {tracked_path}')
    run(cmd, log)
    tracked_tif = f'{tracked_path}.tif'

    # ---------- Step 3: denoising ----------
    if args.finetune:
        ft_dir = os.path.join(out_dir, f'umvd_finetune_{base}')
        cmd = (f'CUDA_VISIBLE_DEVICES={args.gpu} {PY_UMVD} -u {UMVD_DIR}/train_lc.py '
               f'--data {tracked_tif} --output {ft_dir} '
               f'--n-frames 7 --num-epochs 5 --batch-size 8 --image-size 128 --patience 0 --lr 5e-4 '
               f'--init-weights {args.umvd_weights}')
        run(cmd, log)
        denoised_npy = os.path.join(ft_dir, 'denoised.npy')
    else:
        denoised_npy = os.path.join(out_dir, f'denoised_{base}.npy')
        cmd = (f'CUDA_VISIBLE_DEVICES={args.gpu} {PY_UMVD} -u {UMVD_DIR}/inference_lc.py '
               f'--data {tracked_tif} --weights {args.umvd_weights} --output {denoised_npy}')
        run(cmd, log)

    # ---------- Step 4: segmentation + tight crop + 4x speedup ----------
    cmd = (f'CUDA_VISIBLE_DEVICES={args.gpu} {PY_SAM3} -u {SCRIPT_DIR}/seg_crop_speedup.py '
           f'--input {denoised_npy} --output-dir {out_dir} '
           f'--name segcrop_{base} --shrink {args.shrink}')
    run(cmd, log)

    elapsed = time.time() - t0
    msg = f'\n[pipeline] DONE in {elapsed/60:.1f} min — output dir: {out_dir}'
    print(msg); log.write(msg + '\n')
    log.close()

    print('\nKey outputs:')
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(('.mp4', '.tif', '.npy', '.csv', '.log')):
            sz = os.path.getsize(os.path.join(out_dir, f))
            print(f'  {f}  ({sz/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
