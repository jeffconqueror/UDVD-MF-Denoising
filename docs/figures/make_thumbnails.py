#!/usr/bin/env python3
"""
Extract the real image thumbnails used in Figures 1 and 2.

Pulls one representative frame out of each pipeline artifact for the 053243
video so the figures show actual data at each stage rather than placeholder
glyphs. Writes small PNGs into docs/figures/assets/ (committed, so the figure
generator has no dependency on /shared).

    python make_thumbnails.py            # uses the defaults below
    python make_thumbnails.py --frame 500

Requires an environment with tifffile + opencv (e.g. the `umvd` env).
"""
import argparse
import os

import cv2
import numpy as np
import tifffile

SHARED = '/shared/jingchl6/material/lc-research'
V = f'{SHARED}/test/053243'
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def norm_u8(a, lo=0.5, hi=99.5):
    """Percentile-stretch to 8 bit — matches how these frames are viewed."""
    a = np.asarray(a, dtype=np.float32)
    p0, p1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p0) / max(p1 - p0, 1e-6) * 255, 0, 255).astype(np.uint8)


def save(name, img, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    path = os.path.join(ASSETS, name)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f'  {name:28s} {img.shape}  {os.path.getsize(path)/1024:.0f} KB')
    return path


def tif_frame(path, idx):
    """Read a single page out of a large TIFF stack without loading the rest."""
    with tifffile.TiffFile(path) as tf:
        idx = min(idx, len(tf.pages) - 1)
        return tf.pages[idx].asarray()


def npy_frame(path, idx):
    a = np.load(path, mmap_mode='r')
    idx = min(idx, len(a) - 1)
    return np.array(a[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', type=int, default=500)
    ap.add_argument('--size', type=int, default=190)
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    f = args.frame
    S = args.size

    # 1. raw AVI frame (2048², full field)
    cap = cv2.VideoCapture(f'{SHARED}/PtRu TiO2 chip1 FS30202_20211106_053243.avi')
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, fr = cap.read()
    cap.release()
    if ok:
        save('01_raw_full.png', norm_u8(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)), S)

    # 2. drift-corrected / tracked particle crop (512²)
    tracked = tif_frame(f'{V}/tracked_ts900_particle.tif', f)
    save('02_aligned.png', norm_u8(tracked), S)

    # 3. UMVD-denoised (512²)
    den = npy_frame(f'{V}/umvd_ts900/denoised.npy', f)
    save('03_denoised_umvd.png', norm_u8(den), S)

    # 3b. UDVD-MF for the side-by-side denoiser comparison
    if os.path.exists(f'{V}/udvd_ts900_pass2.npy'):
        save('03b_denoised_udvd.png',
             norm_u8(npy_frame(f'{V}/udvd_ts900_pass2.npy', f)), S)

    # 4. SAM 3 tight crop (256²)
    seg = tif_frame(f'{V}/cropped_umvd_tight.tif', f)
    save('04_segcrop.png', norm_u8(seg), S)

    # 5. one labelled example per phase class, for Figure 2
    import glob
    for cls, out in [('Dh', '05_class_Dh.png'), ('Ih', '05_class_Ih.png'),
                     ('Ih_to_Dh', '05_class_IhDh.png')]:
        hits = sorted(glob.glob(f'{SHARED}/class_comparison/{cls}/*.tif'))
        if hits:
            save(out, norm_u8(tifffile.imread(hits[0])), 120)


if __name__ == '__main__':
    main()
