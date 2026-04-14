"""Convert denoised NPY to MP4 video, optionally side-by-side with raw."""
import sys, numpy as np, cv2, tifffile

denoised_path = sys.argv[1] if len(sys.argv) > 1 else '/shared/jingchl6/material/lc-research/test/udvd_ts900.npy'
raw_path = sys.argv[2] if len(sys.argv) > 2 else '/shared/jingchl6/material/lc-research/test/tracked_ts900_particle.tif'
out_path = sys.argv[3] if len(sys.argv) > 3 else '/shared/jingchl6/material/lc-research/test/udvd_ts900_comparison.mp4'

print(f'Loading denoised: {denoised_path}')
denoised = np.load(denoised_path).astype(np.float32)
if denoised.ndim == 4:
    denoised = denoised[:, 0]

print(f'Loading raw: {raw_path}')
raw = tifffile.imread(raw_path).astype(np.float32)

T = min(len(raw), len(denoised))
raw = raw[:T]
denoised = denoised[:T]

def norm_u8(x):
    mn, mx = x.min(), x.max()
    return np.clip(((x - mn) / (mx - mn + 1e-8) * 255), 0, 255).astype(np.uint8)

H, W = raw.shape[1], raw.shape[2]
fps = 5.0

# Denoised-only video
print('Writing denoised-only video...')
dn_writer = cv2.VideoWriter(out_path.replace('_comparison.mp4', '.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), isColor=False)
for i in range(T):
    dn_writer.write(norm_u8(denoised[i]))
dn_writer.release()
print(f'  Saved: {out_path.replace("_comparison.mp4", ".mp4")}')

# Side-by-side raw | denoised
print('Writing side-by-side comparison video...')
cmp_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W*2, H))
for i in range(T):
    r = norm_u8(raw[i])
    d = norm_u8(denoised[i])
    combined = np.concatenate([r, d], axis=1)
    bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, 'RAW', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(bgr, 'DENOISED', (W+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(bgr, f'f{i}', (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cmp_writer.write(bgr)
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{T}')
cmp_writer.release()
print(f'  Saved: {out_path}')
