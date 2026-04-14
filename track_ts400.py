"""
Run with 400x400 template centered at (1035, 830) — shifted left/up by 50
from particle center (1085, 880) to capture more useful features.
Report particle position (not template center) by adding back the offset.
"""
import cv2, numpy as np, tifffile

video_path = '/shared/jingchl6/material/lc-research/PtRu TiO2 chip1 FS30202_20211106_053243.avi'
blur_frames = {88, 246, 271, 291, 612, 613, 694}
# Template center (shifted)
tcx, tcy = 1035, 830
# Particle center (what we track and report)
pcx, pcy = 1085, 880
# Offset from template center to particle center
off_x, off_y = pcx - tcx, pcy - tcy  # (+50, +50)

template_size = 400
crop_size = 512
half_t = template_size // 2
half_c = crop_size // 2

BP_LOW = 1.5
BP_HIGH = 15.0

ground_truth_cells = {
    0: (1024, 512),
    89: (1024, 1536),
    247: (1024, 1024),
    272: (1024, 1536),
    292: (1024, 1536),
    624: (512, 1024),
    695: (1024, 1024),
    705: (1024, 1024),
}

def bandpass(img, low=BP_LOW, high=BP_HIGH):
    img = img.astype(np.float32)
    k_low = max(3, int(6*low) | 1)
    k_high = max(3, int(6*high) | 1)
    return cv2.normalize(
        cv2.GaussianBlur(img, (k_low, k_low), low) - cv2.GaussianBlur(img, (k_high, k_high), high),
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

ret, f0 = cap.read()
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
g0_bp = bandpass(g0)
ref = g0_bp[tcy-half_t:tcy+half_t, tcx-half_t:tcx+half_t].copy()

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

positions = {0: (pcx, pcy)}  # store PARTICLE positions
kept = [0]
scores = [1.0]
crops = [extract_crop(g0, pcx, pcy)]

for i in range(1, total):
    ret, frame = cap.read()
    if not ret: break
    if i in blur_frames:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bp = bandpass(gray)

    # Global NCC search
    result = cv2.matchTemplate(gray_bp, ref, cv2.TM_CCOEFF_NORMED)
    _, v, _, l = cv2.minMaxLoc(result)
    # Template match gives top-left; template center is +half_t
    t_match_cx = l[0] + half_t
    t_match_cy = l[1] + half_t
    # Convert to particle position by adding offset
    p_cx = t_match_cx + off_x
    p_cy = t_match_cy + off_y

    positions[i] = (p_cx, p_cy)
    scores.append(v)
    kept.append(i)
    crops.append(extract_crop(gray, p_cx, p_cy))

    if len(kept) % 200 == 0:
        print(f'  {len(kept)}, f{i}, particle=({p_cx},{p_cy}) score={v:.3f}')

cap.release()

print('\n=== Ground Truth (comparing sudden change frames) ===')
correct = 0
for fi, gt_cell in sorted(ground_truth_cells.items()):
    if fi in positions:
        px, py = positions[fi]
        cell = (px//512*512, py//512*512)
        ok = cell == gt_cell
        if ok: correct += 1
        marker = '<<< transition' if fi in [89, 247, 272, 292, 624, 695, 705] else ''
        print(f'  F{fi}: particle=({px},{py}) cell={cell} exp={gt_cell} → {"OK" if ok else "WRONG"} {marker}')
print(f'\n{correct}/{len(ground_truth_cells)} correct')

cs = np.array(crops, dtype=np.uint8)
tifffile.imwrite('/shared/jingchl6/material/lc-research/test/tracked_ts400.tif', cs.astype(np.float32))
w = cv2.VideoWriter('/shared/jingchl6/material/lc-research/test/tracked_ts400.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_size, crop_size), isColor=False)
for c in crops: w.write(c)
w.release()
np.save('/shared/jingchl6/material/lc-research/test/tracked_ts400_positions.npy',
        np.array([(fi, *positions[fi], s) for fi, s in zip(kept, scores)]))
print(f'Saved: tracked_ts400.tif + .mp4 ({len(kept)} frames)')

# Full-res overlay with 400x400 template box (shifted) AND particle center
print('\nCreating overlay...')
cap = cv2.VideoCapture(video_path)
writer = cv2.VideoWriter('/shared/jingchl6/material/lc-research/test/tracked_ts400_overlay.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
last_px, last_py, last_s = pcx, pcy, 1.0
idx_map = {fi: k for k, fi in enumerate(kept)}
for i in range(total):
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    is_blur = i in blur_frames
    if not is_blur and i in positions:
        last_px, last_py = positions[i]
        last_s = scores[idx_map[i]]
    # Template box = particle - offset, 400x400
    tbx, tby = last_px - off_x, last_py - off_y
    color = (0, 0, 255) if is_blur else (0, 255, 0)
    # Particle location (small circle)
    cv2.circle(bgr, (last_px, last_py), 15, (255, 0, 255), -1)  # magenta dot = particle center
    # Template box (400x400)
    cv2.rectangle(bgr, (tbx-half_t, tby-half_t), (tbx+half_t, tby+half_t), color, 8)
    cv2.putText(bgr, f'f{i} ({i/fps:.1f}s) {"BLUR" if is_blur else f"score={last_s:.2f}"}',
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    cv2.putText(bgr, f'particle=({last_px},{last_py}) template 400x400 [shifted -50,-50]',
                (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    writer.write(bgr)
    if (i+1) % 200 == 0:
        print(f'  overlay {i+1}/{total}')
writer.release()
cap.release()
print('Saved: tracked_ts400_overlay.mp4 (2048x2048)')
