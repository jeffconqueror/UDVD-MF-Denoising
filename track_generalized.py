"""Generalized DM-style tracker — takes video path + template coords + particle offset."""
import cv2, numpy as np, tifffile, argparse, os
from pathlib import Path


def bandpass(img, low=1.5, high=15.0):
    img = img.astype(np.float32)
    k_low = max(3, int(6*low) | 1)
    k_high = max(3, int(6*high) | 1)
    return cv2.normalize(
        cv2.GaussianBlur(img, (k_low, k_low), low) - cv2.GaussianBlur(img, (k_high, k_high), high),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def detect_blur_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sharpness = []
    for _ in range(total):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.release()
    sharpness = np.array(sharpness)
    diff = np.diff(sharpness)
    diff_std = np.std(diff)
    blurry = np.where(diff < -3 * diff_std)[0] + 1
    return set(blurry.tolist()), sharpness


def main(args):
    video_path = args.video
    tcx, tcy = args.tcx, args.tcy
    off_x, off_y = args.off_x, args.off_y
    pcx, pcy = tcx + off_x, tcy + off_y
    template_size = args.template_size
    crop_size = args.crop_size
    half_t = template_size // 2
    half_c = crop_size // 2

    print(f'Detecting blur frames...')
    blur_frames, _ = detect_blur_frames(video_path)
    print(f'Blur frames: {sorted(blur_frames)}')

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ret, f0 = cap.read()
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g0_bp = bandpass(g0)
    ref = g0_bp[tcy-half_t:tcy+half_t, tcx-half_t:tcx+half_t].copy()
    print(f'Template {template_size}x{template_size} from ({tcx},{tcy}); particle at ({pcx},{pcy})')

    def extract_crop(g, cx, cy):
        cx, cy = int(cx), int(cy)
        x1, y1 = cx-half_c, cy-half_c
        pl, pt = max(0,-x1), max(0,-y1)
        pr, pb = max(0,x1+crop_size-W), max(0,y1+crop_size-H)
        x1, y1 = max(0,x1), max(0,y1)
        c = g[y1:min(H,y1+crop_size-pt), x1:min(W,x1+crop_size-pl)]
        if pl or pt or pr or pb:
            c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        return c[:crop_size,:crop_size]

    positions = {0: (pcx, pcy)}
    crops = [extract_crop(g0, pcx, pcy)]
    kept = [0]
    scores = [1.0]

    for i in range(1, total):
        ret, frame = cap.read()
        if not ret: break
        if i in blur_frames:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bp = bandpass(gray)
        result = cv2.matchTemplate(gray_bp, ref, cv2.TM_CCOEFF_NORMED)
        _, v, _, l = cv2.minMaxLoc(result)
        t_match_cx = l[0] + half_t
        t_match_cy = l[1] + half_t
        p_cx = t_match_cx + off_x
        p_cy = t_match_cy + off_y
        positions[i] = (p_cx, p_cy)
        crops.append(extract_crop(gray, p_cx, p_cy))
        kept.append(i)
        scores.append(v)
        if len(kept) % 200 == 0:
            print(f'  {len(kept)}/{total}, f{i}, particle=({p_cx},{p_cy}) score={v:.3f}')
    cap.release()

    cs = np.array(crops, dtype=np.uint8)
    out_base = args.out
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(f'{out_base}.tif', cs.astype(np.float32))
    w = cv2.VideoWriter(f'{out_base}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (crop_size, crop_size), isColor=False)
    for c in crops: w.write(c)
    w.release()
    np.save(f'{out_base}_positions.npy',
            np.array([(fi, *positions[fi], s) for fi, s in zip(kept, scores)]))
    print(f'Saved: {out_base}.tif + .mp4 ({len(kept)} frames)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tcx', type=int, required=True)
    p.add_argument('--tcy', type=int, required=True)
    p.add_argument('--off-x', type=int, default=200)
    p.add_argument('--off-y', type=int, default=250)
    p.add_argument('--template-size', type=int, default=900)
    p.add_argument('--crop-size', type=int, default=512)
    args = p.parse_args()
    main(args)
