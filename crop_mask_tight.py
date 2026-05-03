"""
Tightest crop: per-frame variable bounding box, resized to uniform output size.
Each particle fills the entire output frame.
"""
import argparse, os, numpy as np, cv2, tifffile


def main(args):
    denoised = np.load(args.denoised).astype(np.float32) if args.denoised.endswith('.npy') else tifffile.imread(args.denoised).astype(np.float32)
    if denoised.ndim == 4:
        denoised = denoised[:, 0]
    masks = tifffile.imread(args.masks).astype(np.uint8)
    T = min(len(denoised), len(masks))
    denoised = denoised[:T]; masks = masks[:T]

    def norm_u8(x):
        mn, mx = x.min(), x.max()
        return np.clip(((x-mn)/(mx-mn+1e-8)*255), 0, 255).astype(np.uint8)

    out_size = args.out_size
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, args.name)

    writer = cv2.VideoWriter(f'{base}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                              5.0, (out_size, out_size), isColor=False)
    resized_stack = []
    H, W = masks.shape[1], masks.shape[2]

    for i in range(T):
        ys, xs = np.where(masks[i] > 0)
        if len(xs) == 0:
            # No mask — blank frame
            resized = np.zeros((out_size, out_size), dtype=np.uint8)
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bw = x2 - x1
            bh = y2 - y1
            side = int(max(bw, bh) * args.shrink)
            # Use mask centroid (center of mass) for centering, not bbox center
            # This handles irregular masks where particle isn't at bbox center
            cx = int(xs.mean())
            cy = int(ys.mean())
            half = side // 2
            # Clamp inside frame
            x1c = max(0, cx - half)
            y1c = max(0, cy - half)
            x2c = min(W, x1c + side)
            y2c = min(H, y1c + side)
            x1c = max(0, x2c - side)
            y1c = max(0, y2c - side)
            crop = denoised[i, y1c:y2c, x1c:x2c]
            # Resize to uniform output
            resized = cv2.resize(norm_u8(crop), (out_size, out_size), interpolation=cv2.INTER_AREA)
        resized_stack.append(resized)
        writer.write(resized)

    writer.release()
    tifffile.imwrite(f'{base}.tif', np.stack(resized_stack))
    print(f'Saved: {base}.mp4 + .tif (uniform {out_size}x{out_size}, particle fills frame)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--denoised', required=True)
    p.add_argument('--masks', required=True)
    p.add_argument('--output-dir', default='/shared/jingchl6/material/lc-research/test')
    p.add_argument('--name', required=True)
    p.add_argument('--out-size', default=256, type=int)
    p.add_argument('--shrink', default=0.75, type=float, help='Shrink factor for bbox (0.75 = 75% of mask size)')
    args = p.parse_args()
    main(args)
