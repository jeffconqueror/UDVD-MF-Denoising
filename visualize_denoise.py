"""
Visualize denoising results: RAW | DENOISED | DIFFERENCE×5

Usage:
    python visualize_denoise.py \
        --raw      path/to/aligned.tif \
        --denoised path/to/denoised.npy \
        --output   path/to/comparison.png \
        [--n 5] [--crop 512]
"""

import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


def ssim_simple(a, b):
    """Simple SSIM estimate (no windowing)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = a.std()
    sig_b = b.std()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2*mu_a*mu_b + C1) * (2*sig_ab + C2)) / \
           ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))


def main(args):
    print(f"Loading raw  : {args.raw}")
    raw = tifffile.imread(args.raw).astype(np.float32)   # (T, H, W)
    print(f"Loading denoised: {args.denoised}")
    den = np.load(args.denoised).astype(np.float32)       # (T, H, W) or (T, 1, H, W)

    # Squeeze any channel dim
    if den.ndim == 4:
        den = den[:, 0]

    T = min(raw.shape[0], den.shape[0])
    raw = raw[:T]
    den = den[:T]

    H, W = raw.shape[1], raw.shape[2]
    crop = min(args.crop, H, W)
    ch, cw = H // 2 - crop // 2, W // 2 - crop // 2

    n = args.n
    indices = np.linspace(0, T - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    print(f"\n{'Frame':>6}  {'PSNR(raw→den)':>15}  {'SSIM':>8}")
    print("-" * 36)

    for row, idx in enumerate(indices):
        r = raw[idx, ch:ch+crop, cw:cw+crop]
        d = den[idx, ch:ch+crop, cw:cw+crop]

        # clip denoised to same range as raw for fair comparison
        d_clip = np.clip(d, 0, 255)
        diff = np.abs(r - d_clip) * 5   # amplify difference

        frame_psnr = psnr(r, d_clip)
        frame_ssim = ssim_simple(r, d_clip)
        print(f"{idx:>6}  {frame_psnr:>15.2f}  {frame_ssim:>8.4f}")

        axes[row, 0].imshow(norm01(r),     cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f'RAW   (frame {idx})', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(norm01(d_clip), cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f'DENOISED  PSNR={frame_psnr:.1f} dB', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(norm01(diff),   cmap='hot',  vmin=0, vmax=1)
        axes[row, 2].set_title('|RAW − DEN| × 5', fontsize=10)
        axes[row, 2].axis('off')

    plt.suptitle('UDVD-MF  ·  Liquid-cell TEM  ·  Centre 512×512 crop', fontsize=13, y=1.001)
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=120, bbox_inches='tight')
    print(f"\nSaved: {out}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw',      required=True, help='Aligned TIFF stack')
    p.add_argument('--denoised', required=True, help='Denoised NPY file')
    p.add_argument('--output',   required=True, help='Output PNG path')
    p.add_argument('--n',    default=5, type=int, help='Number of frames to show')
    p.add_argument('--crop', default=512, type=int, help='Centre crop size (px)')
    return p.parse_args()


if __name__ == '__main__':
    main(get_args())
