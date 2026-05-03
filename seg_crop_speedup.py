"""
SAM 3 segment + tight crop + speedup for a denoised video.
- Segments each frame with central box prompt
- Drops frames with no good mask (low score, area out of range, blur)
- Crops tight around mask centroid (shrink factor)
- Resizes to uniform out_size
- Outputs 4x sped up MP4
"""
import argparse, os, sys, numpy as np, cv2, tifffile, torch
from PIL import Image

sys.path.insert(0, '/home/jingchl6/.local/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def norm_u8(arr):
    mn, mx = arr.min(), arr.max()
    return np.clip(((arr - mn) / (mx - mn + 1e-8) * 255), 0, 255).astype(np.uint8)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, args.name)

    print(f'Loading denoised: {args.input}')
    stack = np.load(args.input).astype(np.float32) if args.input.endswith('.npy') else tifffile.imread(args.input).astype(np.float32)
    if stack.ndim == 4:
        stack = stack[:, 0]
    T, H, W = stack.shape
    print(f'Shape: {stack.shape}')

    print('Loading SAM 3 image model...')
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.conf_thresh)
    print('Model loaded')

    box = [0.5, 0.5, args.box_size, args.box_size]
    out_size = args.out_size
    half_out = out_size // 2

    kept_crops = []
    meta = []  # (frame_idx, kept, area, score, cx, cy)
    n_kept = 0

    with torch.no_grad():
        for i in range(T):
            frame_u8 = norm_u8(stack[i])
            pil = Image.fromarray(frame_u8).convert('RGB')
            state = processor.set_image(pil)
            output = processor.add_geometric_prompt(box=box, label=True, state=state)

            masks_out = output['masks']
            scores = output['scores']

            best_mask = None
            best_area = 0
            best_score = 0.0
            if len(scores) > 0:
                masks_np = masks_out.cpu().numpy().squeeze(1) if hasattr(masks_out, 'cpu') else np.array(masks_out).squeeze(1)
                scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
                for idx in range(len(masks_np)):
                    m = (masks_np[idx] > 0).astype(np.uint8)
                    area = int(m.sum())
                    if area < args.min_area or area > args.max_area:
                        continue
                    score = float(scores_np[idx])
                    if score > best_score:
                        best_mask = m
                        best_area = area
                        best_score = score

            kept = best_mask is not None and best_score >= args.conf_thresh
            cx, cy = -1, -1

            if kept:
                ys, xs = np.where(best_mask > 0)
                if len(xs) == 0:
                    kept = False
                else:
                    # Square bbox, shrunk by factor, centered on centroid
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    side = int(max(x2-x1, y2-y1) * args.shrink)
                    if side < 32:
                        kept = False
                    else:
                        cx = int(xs.mean())
                        cy = int(ys.mean())
                        half = side // 2
                        x1c = max(0, cx-half); y1c = max(0, cy-half)
                        x2c = min(W, x1c+side); y2c = min(H, y1c+side)
                        x1c = max(0, x2c-side); y1c = max(0, y2c-side)
                        crop = stack[i, y1c:y2c, x1c:x2c]
                        if crop.size == 0:
                            kept = False
                        else:
                            resized = cv2.resize(norm_u8(crop), (out_size, out_size), interpolation=cv2.INTER_AREA)
                            kept_crops.append(resized)
                            n_kept += 1

            meta.append((i, kept, best_area, round(best_score, 4), cx, cy))

            if (i+1) % 100 == 0:
                print(f'  {i+1}/{T}  kept={n_kept}  dropped={i+1-n_kept}')

    print(f'\nFinal: kept {n_kept}/{T}  ({n_kept/T*100:.1f}%)')

    # Save artifacts
    if not kept_crops:
        print('No frames kept — aborting save.')
        return

    arr = np.stack(kept_crops)
    tifffile.imwrite(f'{base}.tif', arr)
    print(f'Saved: {base}.tif  shape={arr.shape}')

    # 1x speed video at 5 fps
    w = cv2.VideoWriter(f'{base}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                        5.0, (out_size, out_size), isColor=False)
    for c in kept_crops: w.write(c)
    w.release()
    print(f'Saved: {base}.mp4 (5 fps native)')

    # 4x speed via ffmpeg
    FFMPEG = '/home/jingchl6/miniconda3/envs/sam3/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    fast_path = f'{base}_4x.mp4'
    cmd = (f'{FFMPEG} -i {base}.mp4 -vf "setpts=PTS/4" -r 20 '
           f'-c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -movflags +faststart '
           f'{fast_path} -y 2>&1 | tail -1')
    os.system(cmd)
    print(f'Saved: {fast_path} (4x speed)')

    # Save metadata
    import csv
    with open(f'{base}_meta.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'kept', 'area', 'score', 'cx', 'cy'])
        wr.writerows(meta)
    print(f'Saved: {base}_meta.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--box-size', default=0.4, type=float)
    p.add_argument('--conf-thresh', default=0.3, type=float)
    p.add_argument('--min-area', default=2000, type=int)
    p.add_argument('--max-area', default=80000, type=int)
    p.add_argument('--shrink', default=0.55, type=float)
    p.add_argument('--out-size', default=256, type=int)
    args = p.parse_args()
    main(args)
