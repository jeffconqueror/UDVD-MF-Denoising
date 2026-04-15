"""
SAM 3 segmentation on denoised 512x512 particle-centered NPY videos.
Uses a box prompt centered at image center (0.5, 0.5, 0.4, 0.4).
Drops frames with implausible mask area or low confidence.
"""
import argparse, os, sys, csv, numpy as np, cv2, tifffile, torch
from PIL import Image

sys.path.insert(0, '/home/jingchl6/.local/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def norm_u8(arr):
    mn, mx = arr.min(), arr.max()
    return np.clip(((arr - mn) / (mx - mn + 1e-8) * 255), 0, 255).astype(np.uint8)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading: {args.input}')
    stack = np.load(args.input).astype(np.float32)
    if stack.ndim == 4:
        stack = stack[:, 0]
    T, H, W = stack.shape
    print(f'Shape: {stack.shape}')

    print('Loading SAM 3 image model...')
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.conf_thresh)
    print('Model loaded')

    # Central box prompt in normalized [cx, cy, w, h]
    box = [0.5, 0.5, args.box_size, args.box_size]

    masks = []
    meta_rows = []
    kept_indices = []
    overlay_frames = []

    min_area = args.min_area
    max_area = args.max_area

    with torch.no_grad():
        for i in range(T):
            frame_u8 = norm_u8(stack[i])
            pil = Image.fromarray(frame_u8).convert('RGB')

            state = processor.set_image(pil)
            output = processor.add_geometric_prompt(box=box, label=True, state=state)

            masks_out = output['masks']
            scores = output['scores']

            kept = False
            best_mask = None
            best_area = 0
            best_score = 0.0
            best_circ = 0.0

            if len(scores) > 0:
                masks_np = masks_out.cpu().numpy().squeeze(1) if hasattr(masks_out, 'cpu') else np.array(masks_out).squeeze(1)
                scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)

                # Pick the mask with the most plausible area
                for idx in range(len(masks_np)):
                    m = (masks_np[idx] > 0).astype(np.uint8)
                    area = int(m.sum())
                    if area < min_area or area > max_area:
                        continue
                    # Circularity: 4*pi*area / perimeter^2
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    cnt = max(contours, key=cv2.contourArea)
                    perim = cv2.arcLength(cnt, True)
                    circ = 4 * np.pi * area / (perim**2) if perim > 0 else 0
                    score = float(scores_np[idx])
                    # Prefer higher score + area closer to "typical" ~30000
                    if score > best_score:
                        best_mask = m
                        best_area = area
                        best_score = score
                        best_circ = circ

                if best_mask is not None and best_score >= args.conf_thresh:
                    kept = True

            if kept:
                masks.append(best_mask.astype(np.float32))
                kept_indices.append(i)
                # Overlay: draw green contour on denoised frame
                bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)
                cv2.putText(bgr, f'f{i} a={best_area} s={best_score:.2f}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                overlay_frames.append(bgr)
            else:
                # Dropped: red X overlay
                bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
                cv2.line(bgr, (0, 0), (W, H), (0, 0, 255), 3)
                cv2.line(bgr, (W, 0), (0, H), (0, 0, 255), 3)
                cv2.putText(bgr, f'f{i} DROPPED',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                overlay_frames.append(bgr)

            meta_rows.append({
                'frame': i, 'kept': kept,
                'area': best_area, 'circularity': round(best_circ, 4),
                'score': round(best_score, 4),
            })

            if (i + 1) % 100 == 0:
                n_kept = sum(r['kept'] for r in meta_rows)
                print(f'  {i+1}/{T}  kept={n_kept} dropped={i+1-n_kept}')

    # Save outputs
    out_base = os.path.join(args.output_dir, args.name)

    if masks:
        mask_stack = np.stack(masks)
        tifffile.imwrite(f'{out_base}_masks.tif', mask_stack)
        print(f'Saved: {out_base}_masks.tif  {mask_stack.shape}')
    else:
        print('No masks kept!')

    with open(f'{out_base}_meta.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'kept', 'area', 'circularity', 'score'])
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f'Saved: {out_base}_meta.csv')

    # Overlay video
    fps = 5.0
    writer = cv2.VideoWriter(f'{out_base}_overlay.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in overlay_frames:
        writer.write(f)
    writer.release()
    print(f'Saved: {out_base}_overlay.mp4')

    n_kept = sum(r['kept'] for r in meta_rows)
    print(f'\nSummary: kept {n_kept}/{T} ({n_kept/T*100:.1f}%)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', default='/shared/jingchl6/material/lc-research/test')
    p.add_argument('--name', required=True)
    p.add_argument('--box-size', default=0.4, type=float, help='Normalized box side (0-1)')
    p.add_argument('--conf-thresh', default=0.3, type=float)
    p.add_argument('--min-area', default=2000, type=int)
    p.add_argument('--max-area', default=80000, type=int)
    args = p.parse_args()
    main(args)
