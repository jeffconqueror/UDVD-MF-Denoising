"""
Drift-correction quality experiment: does moving-average registration beat the
current reference='first' registration on the noisiest video (200C)?

Variants (all with bandpass + median/savgol trajectory smoothing, as production):
  v0  ref=first, moving_average=1   (current production baseline)
  v1  ref=first, moving_average=9
  v2  ref=first, moving_average=21

Metric: residual consecutive-frame motion of the aligned stack (phase correlation
on 150 sampled pairs), overall + early (first 20 s) vs late. Nothing is saved to
disk except the report — stacks are evaluated in memory.
"""
import numpy as np, cv2
from pystackreg import StackReg
from scipy.signal import savgol_filter, medfilt
from skimage.registration import phase_cross_correlation

VIDEO = '/shared/jingchl6/material/lc-research/temp_series/200C-20fps.avi'
PROC = 1024
MED, SAV = 31, 71


def bandpass(img, low=1.5, high=15.0):
    img = img.astype(np.float32)
    kl = max(3, int(6*low) | 1); kh = max(3, int(6*high) | 1)
    bp = cv2.GaussianBlur(img, (kl, kl), low) - cv2.GaussianBlur(img, (kh, kh), high)
    return cv2.normalize(bp, None, 0, 255, cv2.NORM_MINMAX)


def residual(stack):
    idxs = np.linspace(1, len(stack)-1, 150, dtype=int)
    ds = []
    for i in idxs:
        sh, _, _ = phase_cross_correlation(stack[i-1], stack[i], upsample_factor=10, normalization=None)
        ds.append((i, float(np.hypot(*sh))))
    ds = np.array(ds)
    early = ds[ds[:, 0] < 400, 1]; late = ds[ds[:, 0] >= 400, 1]
    return ds[:, 1].mean(), np.percentile(ds[:, 1], 90), ds[:, 1].max(), early.mean(), late.mean()


print('Loading 200C at', PROC)
cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (PROC, PROC), interpolation=cv2.INTER_AREA))
cap.release()
frames = np.array(frames, dtype=np.uint8)
print('frames:', frames.shape)
bp = np.stack([bandpass(f) for f in frames]).astype(np.float32)
ff = frames.astype(np.float32)

for name, ma in [('v0 ma=1 (current)', 1), ('v1 ma=9', 9), ('v2 ma=21', 21)]:
    sr = StackReg(StackReg.TRANSLATION)
    tm = sr.register_stack(bp, reference='first', moving_average=ma, verbose=True)
    tx, ty = tm[:, 0, 2].copy(), tm[:, 1, 2].copy()
    raw_jit = (np.std(np.diff(tx)), np.std(np.diff(ty)))
    tx = savgol_filter(medfilt(tx, MED), SAV, 2)
    ty = savgol_filter(medfilt(ty, MED), SAV, 2)
    tm[:, 0, 2] = tx; tm[:, 1, 2] = ty
    sr._tmats = tm
    aligned = sr.transform_stack(ff)
    # evaluate at 512 for speed
    small = np.stack([cv2.resize(a, (512, 512)) for a in aligned])
    m, p90, mx, e, l = residual(small)
    # scale x2 to full-res px
    print(f'RESULT {name}: raw-jitter=({raw_jit[0]:.1f},{raw_jit[1]:.1f})px  '
          f'residual mean={m*2:.2f} p90={p90*2:.2f} max={mx*2:.2f} early={e*2:.2f} late={l*2:.2f}  (full-res px)')
    del aligned, small
print('EXPERIMENT_DONE')
