"""
Drift Correction for TEM/liquid-cell video (AVI → aligned TIFF stack)

Uses pystackreg (TurboReg algorithm) — fully automatic, no ROI selection needed.

Usage:
    python drift_correction.py --input path/to/video.avi \
                               --output path/to/aligned.tif \
                               [--reference first|previous] \
                               [--start 0] [--end -1]

Output:
    - aligned TIFF stack
    - <stem>_drift_plot.png
    - <stem>_drift_stats.txt
"""

import argparse
import numpy as np
import cv2
import tifffile
from pystackreg import StackReg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def read_video_frames(path, start=0, end=-1, resize=None):
    """Read AVI frames, return (T, H, W) uint8 grayscale + fps."""
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if end == -1 or end > total:
        end = total

    print(f"Video : {total} frames  {fps} fps  {W}x{H}")
    if resize:
        print(f"Resize: {W}x{H} → {resize}x{resize}")
    print(f"Range : frames {start}–{end}  ({end - start} frames)")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(gray)
        if (i + 1) % 200 == 0:
            print(f"  Read {i+1}/{end-start}")
    cap.release()

    return np.array(frames, dtype=np.uint8), fps


def drift_correct(frames, reference='first'):
    """
    Align all frames using pystackreg (TurboReg TRANSLATION mode).

    Args:
        frames    : (T, H, W) uint8
        reference : 'first'    — all frames aligned to frame 0
                    'previous' — each frame aligned to the previous one
                                 (better for large cumulative drift)
    Returns:
        aligned : (T, H, W) float32
        shifts  : (T, 2) float32  [row_shift, col_shift]
    """
    print(f"Running pystackreg (TRANSLATION, reference='{reference}')...")
    sr = StackReg(StackReg.TRANSLATION)

    # register_transform_stack does registration + transformation in one call
    aligned = sr.register_transform_stack(
        frames.astype(np.float32),
        reference=reference,
        verbose=True
    )

    # Extract shifts from the stored transformation matrices
    tmats = sr._tmats   # (3, 3) of last frame — need to re-register for all
    # Re-register to get all matrices
    tmats_all = sr.register_stack(
        frames.astype(np.float32),
        reference=reference,
        verbose=False
    )
    shifts = tmats_all[:, :2, 2].copy()

    return aligned.astype(np.float32), shifts


def save_drift_plot(shifts, fps, output_path):
    t = np.arange(len(shifts)) / fps

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(t, shifts[:, 1], label='X (col)', color='steelblue', linewidth=0.8)
    axes[0].plot(t, shifts[:, 0], label='Y (row)', color='coral',     linewidth=0.8)
    axes[0].set_ylabel('Shift (px)')
    axes[0].set_title('Drift trajectory')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    total = np.sqrt(shifts[:, 0]**2 + shifts[:, 1]**2)
    axes[1].plot(t, total, color='purple', linewidth=0.8)
    axes[1].set_ylabel('Total drift (px)')
    axes[1].set_title('Drift magnitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(shifts[:, 1], shifts[:, 0], linewidth=0.5, color='green')
    axes[2].scatter(shifts[0,  1], shifts[0,  0], color='blue', s=40, zorder=5, label='start')
    axes[2].scatter(shifts[-1, 1], shifts[-1, 0], color='red',  s=40, zorder=5, label='end')
    axes[2].set_xlabel('X shift (px)'); axes[2].set_ylabel('Y shift (px)')
    axes[2].set_title('Drift path'); axes[2].legend()
    axes[2].grid(True, alpha=0.3); axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"Drift plot : {output_path}")


def save_comparison(frames_raw, frames_aligned, output_path, n=5, crop=512):
    """Save side-by-side before/after for n evenly-spaced frames."""
    T = len(frames_raw)
    H, W = frames_raw.shape[1], frames_raw.shape[2]
    crop = min(crop, H, W)
    indices = np.linspace(0, T - 1, n, dtype=int)
    cx, cy = H // 2, W // 2
    h0, w0 = cx - crop//2, cy - crop//2

    rows = []
    for idx in indices:
        raw = np.clip(frames_raw[idx,  h0:h0+crop, w0:w0+crop], 0, 255).astype(np.uint8)
        aln = np.clip(frames_aligned[idx, h0:h0+crop, w0:w0+crop], 0, 255).astype(np.uint8)
        # normalize each independently for visibility
        def norm(x):
            mn, mx = x.min(), x.max()
            return ((x - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        row = np.concatenate([norm(raw), norm(aln)], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    fig, ax = plt.subplots(figsize=(12, 3 * n))
    ax.imshow(grid, cmap='gray', vmin=0, vmax=255)
    ax.set_title('Left: RAW   |   Right: ALIGNED  (centre crop)')
    ax.axis('off')
    for i, idx in enumerate(indices):
        ax.text(2, i * crop + 12, f'f{idx}', color='yellow', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"Comparison : {output_path}")


def main(args):
    out_path = Path(args.output)
    out_dir  = out_path.parent
    out_stem = out_path.stem

    # 1. Read frames
    frames, fps = read_video_frames(args.input, args.start, args.end)
    print(f"Loaded {len(frames)} frames  shape={frames.shape}\n")

    # 2. Drift correct (register + transform in one call)
    aligned, shifts = drift_correct(frames, reference=args.reference)

    # 3. Clip and optionally downscale
    aligned_clipped = np.clip(aligned, 0, 255).astype(np.float32)
    del aligned  # free memory

    sz = args.resize
    if sz:
        T = len(aligned_clipped)
        print(f"\nDownscaling: {aligned_clipped.shape[-1]}x{aligned_clipped.shape[-2]} → {sz}x{sz}")
        resized = np.zeros((T, sz, sz), dtype=np.float32)
        for i in range(T):
            resized[i] = cv2.resize(aligned_clipped[i], (sz, sz), interpolation=cv2.INTER_AREA)
        aligned_clipped = resized

    tifffile.imwrite(str(out_path), aligned_clipped)
    print(f"Aligned stack : {out_path}  {aligned_clipped.shape}")

    # 5. Drift stats
    stats_path = out_dir / f"{out_stem}_drift_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("frame,row_shift,col_shift,total_shift\n")
        for i, (dy, dx) in enumerate(shifts):
            f.write(f"{i},{dy:.4f},{dx:.4f},{np.sqrt(dy**2+dx**2):.4f}\n")
    print(f"Drift stats   : {stats_path}")

    # 5. Drift plot
    plot_path = out_dir / f"{out_stem}_drift_plot.png"
    save_drift_plot(shifts, fps, str(plot_path))

    # 6. Before/after comparison (use aligned output for both if resized)
    cmp_path = out_dir / f"{out_stem}_comparison.png"
    if sz:
        raw_resized = np.zeros((T, sz, sz), dtype=np.float32)
        for i in range(T):
            raw_resized[i] = cv2.resize(frames[i].astype(np.float32), (sz, sz), interpolation=cv2.INTER_AREA)
        save_comparison(raw_resized, aligned_clipped, str(cmp_path))
        del raw_resized
    else:
        save_comparison(frames, aligned_clipped, str(cmp_path))

    # 7. Summary
    total = np.sqrt(shifts[:, 0]**2 + shifts[:, 1]**2)
    print(f"\nSummary:")
    print(f"  Max X shift  : {np.abs(shifts[:,1]).max():.3f} px")
    print(f"  Max Y shift  : {np.abs(shifts[:,0]).max():.3f} px")
    print(f"  Max total    : {total.max():.3f} px")
    print(f"  Mean total   : {total.mean():.3f} px")


def get_args():
    parser = argparse.ArgumentParser(
        description='TEM/liquid-cell drift correction via pystackreg (TurboReg)')
    parser.add_argument('--input',     required=True,
                        help='Input AVI file')
    parser.add_argument('--output',    required=True,
                        help='Output aligned TIFF stack')
    parser.add_argument('--reference', default='first',
                        choices=['first', 'previous'],
                        help='Reference mode: first frame or previous frame (default: first)')
    parser.add_argument('--start',     default=0,  type=int,
                        help='Start frame index (default: 0)')
    parser.add_argument('--end',       default=-1, type=int,
                        help='End frame index, -1 = all (default: -1)')
    parser.add_argument('--resize',    default=None, type=int,
                        help='Resize frames to NxN (e.g. 512)')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
