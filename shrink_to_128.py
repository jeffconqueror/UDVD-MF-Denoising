"""
Downscale a video's drift+denoise outputs to 128x128 to save disk space.

For each video: loads the 1024^2 aligned (raw), transfer-denoised, and fine-tune-denoised
stacks, downscales every frame to 128x128, then:
  - saves small 128^2 arrays: raw128.npy, transfer128.npy, finetune128.npy
  - builds a 3-panel comparison MP4 (raw | transfer | fine-tune) at 128/panel
Returns sizes so the caller can delete the giant originals.
"""
import argparse, os, numpy as np, cv2, tifffile

p = argparse.ArgumentParser()
p.add_argument('--dir', required=True)
p.add_argument('--name', required=True)
p.add_argument('--transfer-npy', required=True)   # actual transfer filename varies
p.add_argument('--size', type=int, default=128)
p.add_argument('--fps', type=float, default=20.0)
args = p.parse_args()

S = args.size
raw  = tifffile.imread(f'{args.dir}/aligned_{args.name}.tif')
xfer = np.load(args.transfer_npy)
ft   = np.load(f'{args.dir}/umvd_finetune_{args.name}/denoised.npy')
T = min(len(raw), len(xfer), len(ft))
print(f'{args.name}: T={T}, downscaling {raw.shape[1]}->{S}')

def norm(x):
    mn, mx = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)

# 3-panel comparison video only (no arrays saved)
out = f'{args.dir}/{args.name}_128_compare.mp4'
hdr = 22
w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (S*3, S+hdr), isColor=False)
for i in range(T):
    r = cv2.resize(norm(raw[i]),  (S, S), interpolation=cv2.INTER_AREA)
    x = cv2.resize(norm(xfer[i]), (S, S), interpolation=cv2.INTER_AREA)
    f = cv2.resize(norm(ft[i]),   (S, S), interpolation=cv2.INTER_AREA)
    strip = np.concatenate([r, x, f], axis=1)
    header = np.zeros((hdr, S*3), np.uint8)
    for k, lbl in enumerate(['RAW', 'TRANSFER', 'FINETUNE']):
        cv2.putText(header, lbl, (k*S+4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,), 1)
    w.write(np.concatenate([header, strip], axis=0))
w.release()
print(f'Saved: {out}')
