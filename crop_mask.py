"""
Crop each frame tight around the SAM 3 mask.
- Find each mask's bounding box
- Determine uniform crop size = 95th percentile max dim + padding
- Center crop on mask centroid at uniform size
- Output TIFF + mp4 at the tight crop size
"""
import argparse, os, numpy as np, cv2, tifffile


def main(args):
    print(f'Loading denoised: {args.denoised}')
    denoised = np.load(args.denoised).astype(np.float32) if args.denoised.endswith('.npy') else tifffile.imread(args.denoised).astype(np.float32)
    if denoised.ndim == 4:
        denoised = denoised[:, 0]

    print(f'Loading masks: {args.masks}')
    masks = tifffile.imread(args.masks).astype(np.uint8)

    T = min(len(denoised), len(masks))
    denoised = denoised[:T]
    masks = masks[:T]
    H, W = denoised.shape[1], denoised.shape[2]
    print(f'Frames: {T}, size {H}x{W}')

    # Per-frame bounding box
    bboxes = []
    for i in range(T):
        ys, xs = np.where(masks[i] > 0)
        if len(xs) == 0:
            bboxes.append(None)
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bboxes.append((x1, y1, x2, y2))

    # Pick uniform crop size: 95th percentile of max(width, height) + padding
    dims = [max(b[2]-b[0], b[3]-b[1]) for b in bboxes if b is not None]
    p95 = int(np.percentile(dims, 95))
    max_dim = max(dims)
    pad = args.padding
    crop_size = p95 + 2*pad
    # Ensure even
    crop_size = ((crop_size + 1) // 2) * 2
    # Clamp to frame size
    crop_size = min(crop_size, min(H, W))
    print(f'Mask dims: min={min(dims)}, max={max_dim}, p95={p95}')
    print(f'Crop size (with padding {pad}): {crop_size}x{crop_size}')

    half = crop_size // 2

    def norm_u8(x):
        mn, mx = x.min(), x.max()
        return np.clip(((x - mn) / (mx - mn + 1e-8) * 255), 0, 255).astype(np.uint8)

    crops = []
    centroids = []
    for i in range(T):
        bb = bboxes[i]
        if bb is None:
            # No mask: use image center
            cx, cy = W // 2, H // 2
        else:
            x1, y1, x2, y2 = bb
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        # Clamp centroid so crop stays within image
        cx = max(half, min(W - half, cx))
        cy = max(half, min(H - half, cy))
        centroids.append((cx, cy))

        crop = denoised[i, cy-half:cy+half, cx-half:cx+half]
        crops.append(crop)

    crops_arr = np.stack(crops)
    centroids_arr = np.array(centroids)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, args.name)
    tifffile.imwrite(f'{base}_cropped.tif', crops_arr.astype(np.float32))
    np.save(f'{base}_centroids.npy', centroids_arr)
    print(f'Saved: {base}_cropped.tif  shape={crops_arr.shape}')

    # Video
    writer = cv2.VideoWriter(f'{base}_cropped.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (crop_size, crop_size), isColor=False)
    for c in crops:
        writer.write(norm_u8(c))
    writer.release()
    print(f'Saved: {base}_cropped.mp4')

    # Also: cropped + mask contour overlay video
    writer2 = cv2.VideoWriter(f'{base}_cropped_overlay.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (crop_size, crop_size))
    for i in range(T):
        cx, cy = centroids[i]
        frame_crop = norm_u8(crops[i])
        mask_crop = masks[i, cy-half:cy+half, cx-half:cx+half]
        bgr = cv2.cvtColor(frame_crop, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 0), 1)
        writer2.write(bgr)
    writer2.release()
    print(f'Saved: {base}_cropped_overlay.mp4')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--denoised', required=True)
    p.add_argument('--masks', required=True)
    p.add_argument('--output-dir', default='/shared/jingchl6/material/lc-research/test')
    p.add_argument('--name', required=True)
    p.add_argument('--padding', default=20, type=int)
    args = p.parse_args()
    main(args)
