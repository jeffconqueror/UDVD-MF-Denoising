#!/usr/bin/env python3
"""
Generate the two paper figures as standalone, editable SVG.

  fig1_pipeline.svg   — the full LC-TEM pipeline (4 stages + data + results)
  fig2_classifier.svg — the Stage-4 classifier (Swin-Tiny + dihedral TTA +
                        temporal post-processing)

No dependencies — writes SVG text directly. To rasterise:

    python make_figures.py
    google-chrome --headless --disable-gpu --screenshot=fig1_pipeline.png \
        --window-size=1680,1010 --default-background-color=ffffff fig1_pipeline.svg

Every number in these figures is taken from the code; see
docs/PIPELINE_METHODS.md for the matching text.
"""
import html
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette ---
INK       = '#16191d'   # primary text
INK2      = '#495159'   # secondary text
INK3      = '#6b747d'   # tertiary text
RULE      = '#d6dbe0'   # hairlines
PAPER     = '#ffffff'
PANEL_BG  = '#ffffff'
SOFT      = '#f5f7f9'   # soft fill for sub-boxes

STAGE = {
    1: dict(key='#2f6f9f', bg='#eef4f9', name='Drift correction'),
    2: dict(key='#1f7a68', bg='#e9f4f1', name='Denoising'),
    3: dict(key='#a06a14', bg='#fbf2e2', name='Segmentation'),
    4: dict(key='#6a4c9c', bg='#f1ecf9', name='Classification'),
}
ACCENT_RED = '#b3402f'

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', "
        "Arial, sans-serif")
MONO = "'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"


# ------------------------------------------------------------- primitives ---
def esc(s):
    return html.escape(str(s), quote=False)


def text(x, y, s, size=11.5, fill=INK2, weight=400, anchor='start',
         family=None, spacing=0, style=''):
    fam = family or FONT
    ls = f' letter-spacing="{spacing}"' if spacing else ''
    st = f' style="{style}"' if style else ''
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{ls}{st}>'
            f'{esc(s)}</text>')


def rect(x, y, w, h, fill=PANEL_BG, stroke=RULE, rx=8, sw=1, dash=None, op=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' fill-opacity="{op}"' if op is not None else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def line(x1, y1, x2, y2, stroke=RULE, sw=1, dash=None, cap='round'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="{cap}"{d}/>')


def arrow(x1, y1, x2, y2, stroke=INK3, sw=1.8, dash=None, marker='arrowhead'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d} '
            f'marker-end="url(#{marker})"/>')


def circle(cx, cy, r, fill, stroke='none', sw=1):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def step_marker(cx, cy, n, color):
    return (circle(cx, cy, 8.5, color) +
            text(cx, cy + 3.6, n, size=9.5, fill='#ffffff', weight=700,
                 anchor='middle'))


def para(x, y, lines, size=11.5, lh=15.5, fill=INK2, weight=400, family=None):
    """Render pre-wrapped lines; a line may be (str, fill) or (str, fill, weight)."""
    out = []
    for i, ln in enumerate(lines):
        f, w = fill, weight
        if isinstance(ln, tuple):
            if len(ln) == 3:
                ln, f, w = ln
            else:
                ln, f = ln
        out.append(text(x, y + i * lh, ln, size=size, fill=f, weight=w,
                        family=family))
    return ''.join(out)


def svg_doc(w, h, body, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" \
viewBox="0 0 {w} {h}" font-family="{FONT}">
<title>{esc(title)}</title>
<defs>
  <marker id="arrowhead" markerWidth="9" markerHeight="9" refX="7.4" refY="3.2"
          orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L7.6,3.2 L0,6.4 z" fill="{INK3}"/>
  </marker>
  <marker id="arrowhead_k" markerWidth="9" markerHeight="9" refX="7.4" refY="3.2"
          orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L7.6,3.2 L0,6.4 z" fill="{INK}"/>
  </marker>
  <marker id="arrowhead_p" markerWidth="9" markerHeight="9" refX="7.4" refY="3.2"
          orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L7.6,3.2 L0,6.4 z" fill="{STAGE[4]['key']}"/>
  </marker>
</defs>
<rect width="{w}" height="{h}" fill="{PAPER}"/>
{body}
</svg>
'''


# ============================================================== FIGURE 1 ====
def figure1():
    W, H = 1680, 960
    s = []

    # --- title -------------------------------------------------------------
    s.append(text(40, 46, 'Figure 1  |  End-to-end pipeline for phase analysis of '
                          'liquid-cell TEM nanoparticle videos',
                  size=21, fill=INK, weight=700))
    s.append(text(40, 72, 'Raw low-dose AVI  →  drift correction  →  self-supervised '
                          'denoising  →  SAM 3 segmentation  →  per-frame '
                          'crystallographic phase timeline',
                  size=13, fill=INK3))
    s.append(line(40, 88, W - 40, 88, RULE, 1))

    PY, PH = 150, 380          # panel band
    MID = PY + PH / 2

    # --- input chip --------------------------------------------------------
    ix, iw = 40, 200
    iy = MID - 56
    s.append(rect(ix, iy, iw, 112, SOFT, RULE, rx=10))
    s.append(text(ix + 16, iy + 26, 'INPUT', size=9.5, fill=INK3, weight=700, spacing=1.2))
    s.append(text(ix + 16, iy + 50, 'Raw AVI movie', size=13, fill=INK, weight=600))
    s.append(para(ix + 16, iy + 70, [
        '2048 × 2048 px, 8-bit',
        '866–1980 frames',
        '5 fps (RT) / 20 fps (heating)',
    ], size=10.5, lh=14, fill=INK2))

    panels = [
        dict(n=1, x=280, w=320,
             script='drift_correction.py · track_generalized.py',
             sub='TurboReg translation registration',
             steps=[
                 ['Band-pass DoG (σ = 1.5 / 15) before matching',
                  ('— raw cross-correlation fails on HRTEM', ACCENT_RED)],
                 ['pystackreg TurboReg, TRANSLATION only,',
                  'reference = frame 0, registered at 1024²'],
                 ['Moving-average registration, N = 9 frames'],
                 ['Trajectory: median filter (11–31) kills spikes,',
                  'then Savitzky–Golay (51–71, order 2) kills',
                  'jitter but keeps the slow physical drift'],
                 ['Smoothed shifts applied to the RAW frames'],
             ],
             result='max single-frame step  250–870 px → 3–5 px',
             out='aligned.tif (float32) + shift table + drift plot'),
        dict(n=2, x=630, w=320,
             script='UMVD/train_lc.py · inference_lc.py',
             sub='UMVD — self-supervised, 1.79 M params',
             steps=[
                 ['7-frame window {I t−3 … I t+3}, index-clamped',
                  'at the sequence boundaries'],
                 ['Depthwise convs → 21 channels per frame',
                  '(no cross-frame mixing yet — mask stays exact)'],
                 [('BLIND-FRAME MASK   w = [1,1,1,0,1,1,1]', ACCENT_RED, 700),
                  ('the centre frame is zeroed: the network cannot', ACCENT_RED),
                  ('see the frame it is asked to predict', ACCENT_RED)],
                 ['Concat 7 × 21 = 147 ch → U-Net (48/48/96)',
                  '→ 1×1 head  96 → 384 → 96 → 1'],
                 ['Loss = ‖ f(masked window) − I t ‖² / 2 —',
                  'the target is the NOISY centre frame'],
             ],
             result='no clean reference used anywhere',
             out='denoised.npy (float32, original intensity units)'),
        dict(n=3, x=980, w=310,
             script='seg_crop_speedup.py',
             sub='SAM 3 image model, box prompt',
             steps=[
                 ['Central box prompt  [0.5, 0.5, 0.4, 0.4]'],
                 ['Keep the best mask with score ≥ 0.3 and',
                  'area ∈ [5 000, 600 000] px; else drop frame'],
                 ['Square side = max(Δx, Δy) × 0.6, centred on',
                  'the mask CENTROID — stable frame-to-frame,',
                  'unlike the bounding-box centre'],
                 ['Area-resize every crop to 256 × 256'],
             ],
             result='particle fills a constant fraction of the frame',
             out='segcrop.tif (T × 256²) + meta.csv + 4× MP4'),
        dict(n=4, x=1320, w=320,
             script='classifier/train.py · classify_video.py',
             sub='Swin-Tiny, 27.5 M params  (see Fig. 2)',
             steps=[
                 ['Swin-Tiny @ 224², ImageNet-1k init, warm-started',
                  '3 classes:  Dh / FCC / Ih-family'],
                 ['8-fold dihedral TTA (4 rotations × flip)',
                  ('95.88 % → 96.62 % val accuracy', STAGE[4]['key'], 600)],
                 ['Temporal median filter over 11 frames'],
                 ['Segment, then absorb runs < 10 frames'],
                 ['Ih→Dh recovered inside Ih-family segments',
                  'using a second (drop-Ih→Dh) model'],
             ],
             result='96.62 % particle-level validation accuracy',
             out='timeline.png · segments.csv · overlay MP4'),
    ]

    for p in panels:
        col = STAGE[p['n']]
        x, w = p['x'], p['w']
        s.append(rect(x, PY, w, PH, PANEL_BG, RULE, rx=12))
        # header
        s.append(f'<path d="M{x},{PY+12} a12,12 0 0 1 12,-12 h{w-24} '
                 f'a12,12 0 0 1 12,12 v34 h{-w} z" fill="{col["bg"]}"/>')
        s.append(line(x, PY + 46, x + w, PY + 46, RULE, 1))
        s.append(circle(x + 26, PY + 23, 11, col['key']))
        s.append(text(x + 26, PY + 27, p['n'], size=12.5, fill='#ffffff',
                      weight=700, anchor='middle'))
        s.append(text(x + 46, PY + 22, col['name'], size=14, fill=INK, weight=700))
        s.append(text(x + 46, PY + 37, p['script'], size=9, fill=INK3,
                      family=MONO))
        # sub-header
        s.append(text(x + 16, PY + 68, p['sub'], size=10.5, fill=col['key'],
                      weight=600))

        # steps
        y = PY + 92
        for i, lines in enumerate(p['steps'], start=1):
            s.append(step_marker(x + 24, y - 4, i, col['key']))
            s.append(para(x + 40, y, lines, size=11, lh=14.5))
            y += 14.5 * len(lines) + 12

        # result + output strip
        ry = PY + PH - 66
        s.append(line(x + 16, ry, x + w - 16, ry, RULE, 1))
        s.append(text(x + 16, ry + 18, p['result'], size=10.5, fill=INK,
                      weight=600))
        s.append(rect(x + 16, ry + 28, w - 32, 26, col['bg'], 'none', rx=6))
        s.append(text(x + 26, ry + 45, p['out'], size=9.5, fill=col['key'],
                      family=MONO))

    # --- arrows between panels --------------------------------------------
    for a, b in [(240, 280), (600, 630), (950, 980), (1290, 1320)]:
        s.append(arrow(a + 4, MID, b - 4, MID, INK3, 2))

    # --- bottom band -------------------------------------------------------
    BY, BH = 600, 330

    # (left) headline results
    s.append(rect(40, BY, 960, BH, PANEL_BG, RULE, rx=12))
    s.append(text(60, BY + 28, 'Headline results', size=14, fill=INK, weight=700))
    s.append(text(60, BY + 46, 'validation on the real, particle-level split '
                               '(1 214 images) unless noted',
                  size=10, fill=INK3))
    s.append(line(60, BY + 58, 980, BY + 58, RULE, 1))

    cols = [60, 470, 700, 860]
    hdr = ['Configuration', 'Backbone', 'Data', 'Val acc.']
    for cx, hlabel in zip(cols, hdr):
        s.append(text(cx, BY + 78, hlabel, size=9.5, fill=INK3, weight=700,
                      spacing=0.8))

    rows = [
        ('3-class merged  +  8-fold dihedral TTA   ★ best', 'Swin-Tiny', 'real',
         '96.62 %', True),
        ('3-class merged, no TTA', 'Swin-Tiny', 'real', '95.88 %', False),
        ('3-class, Ih→Dh frames dropped', 'Swin-Tiny', 'real', '96.58 %', False),
        ('3-class merged', 'Swin-Base', 'real', '95.55 %', False),
        ('4-class (Ih→Dh kept separate)', 'Swin-Base', 'real', '89.29 %', False),
        ('real + synthetic, validated on real only', 'Swin-Base', 'combined',
         '94.81 %', False),
        ('synthetic only, synthetic validation', 'Swin-Base', 'synthetic',
         '100 %', False),
    ]
    y = BY + 98
    for label, arch, data, acc, best in rows:
        c = INK if best else INK2
        wgt = 700 if best else 400
        if best:
            s.append(rect(52, y - 12, 936, 22, STAGE[4]['bg'], 'none', rx=5))
        s.append(text(cols[0], y + 3, label, size=11, fill=c, weight=wgt))
        s.append(text(cols[1], y + 3, arch, size=11, fill=c, weight=wgt))
        s.append(text(cols[2], y + 3, data, size=11, fill=c, weight=wgt))
        s.append(text(cols[3], y + 3, acc, size=11, fill=c, weight=wgt))
        y += 23

    s.append(line(60, y + 2, 980, y + 2, RULE, 1))
    s.append(para(60, y + 22, [
        ('Ensembling was tested and rejected — averaging the best model with the two '
         'combined-data models reaches only 95.88 %,', INK2),
        ('below the single best model with TTA. Single frames of Ih and Ih→Dh are not '
         'separable, which is why the 4-class model', INK2),
        ('loses 7 points; the transition class is recovered from temporal context '
         'instead (Stage 4, step 5).', INK2),
    ], size=10.5, lh=15))

    # (right) training data for stage 4
    tx, tw = 1030, 610
    s.append(rect(tx, BY, tw, BH, PANEL_BG, STAGE[4]['key'], rx=12, dash='5 4'))
    s.append(text(tx + 20, BY + 28, 'Stage-4 training data', size=14, fill=INK,
                  weight=700))
    s.append(text(tx + 20, BY + 46, 'offline — not part of per-video inference',
                  size=10, fill=INK3))
    s.append(line(tx + 20, BY + 58, tx + tw - 20, BY + 58, RULE, 1))

    # real corpus
    s.append(rect(tx + 20, BY + 72, 280, 150, SOFT, RULE, rx=8))
    s.append(text(tx + 34, BY + 94, 'Real labelled corpus', size=11.5, fill=INK,
                  weight=700))
    s.append(text(tx + 34, BY + 110, '512² Ag particle TIFFs, hand-labelled',
                  size=9.5, fill=INK3))
    counts = [('Dh', '2 396'), ('FCC', '680'), ('Ih', '7 766'),
              ('Ih→Dh', '1 274'), ('total', '12 116')]
    yy = BY + 130
    for i, (k, v) in enumerate(counts):
        bold = 700 if k == 'total' else 400
        s.append(text(tx + 34, yy, k, size=10.5, fill=INK2, weight=bold))
        s.append(text(tx + 180, yy, v, size=10.5, fill=INK2, weight=bold,
                      anchor='end'))
        yy += 16

    # synthetic
    s.append(rect(tx + 315, BY + 72, 275, 150, SOFT, RULE, rx=8))
    s.append(text(tx + 329, BY + 94, 'Synthetic (abTEM)', size=11.5, fill=INK,
                  weight=700))
    s.append(para(tx + 329, BY + 110, [
        'multislice HRTEM of Pt clusters:',
        'Icosahedron / Decahedron / trunc. Oct.',
        '300 kV, Cs = −13 µm, defocus 50–90 Å',
        'dose 1500–3500 e⁻/Å², Poisson + Gauss',
        'random tilt x/y/z ∈ [0°, 90°]',
        'vacuum pad → particle fills 40–95 %',
        '5 000 train + 1 000 val per class',
    ], size=9.5, lh=13.5))

    s.append(para(tx + 20, BY + 246, [
        ('Split by PARTICLE, 90/10 — both frames of a particle stay on one side; a',
         ACCENT_RED),
        ('per-image split would leak particle identity into validation.', ACCENT_RED),
    ], size=10, lh=14))
    s.append(para(tx + 20, BY + 286, [
        ('Class imbalance (Ih : FCC ≈ 11 : 1) is handled with an inverse-frequency',
         INK2),
        ('WeightedRandomSampler + class-weighted cross-entropy (label smoothing 0.05).',
         INK2),
    ], size=10, lh=14))

    # dashed connector: training data → panel 4
    s.append(arrow(tx + tw / 2, BY - 6, tx + tw / 2, PY + PH + 8,
                   STAGE[4]['key'], 1.6, dash='6 5', marker='arrowhead_p'))
    s.append(text(tx + tw / 2 + 10, (BY + PY + PH) / 2 + 4, 'trains',
                  size=10, fill=STAGE[4]['key'], weight=600))

    return svg_doc(W, H, '\n'.join(s),
                   'Figure 1 — LC-TEM nanoparticle analysis pipeline')


# ============================================================== FIGURE 2 ====
def figure2():
    W, H = 1680, 1000
    K = STAGE[4]['key']
    BG = STAGE[4]['bg']
    s = []

    s.append(text(40, 46, 'Figure 2  |  Stage-4 classifier: Swin-Tiny with dihedral '
                          'test-time augmentation and temporal post-processing',
                  size=21, fill=INK, weight=700))
    s.append(text(40, 72, 'One 256 × 256 segmented crop per frame  →  8 equivalent '
                          'views  →  Swin-Tiny  →  averaged posterior  →  '
                          'median-smoothed phase timeline',
                  size=13, fill=INK3))
    s.append(line(40, 88, W - 40, 88, RULE, 1))

    RY, RH = 120, 320          # row-1 band
    MID = RY + RH / 2

    # ---- input ------------------------------------------------------------
    s.append(rect(40, RY + 70, 150, 180, SOFT, RULE, rx=10))
    s.append(text(115, RY + 96, 'INPUT', size=9.5, fill=INK3, weight=700,
                  anchor='middle', spacing=1.2))
    # little particle glyph
    s.append(rect(70, RY + 108, 90, 90, '#2b2f35', '#0f1216', rx=4))
    s.append(circle(115, RY + 153, 30, '#8d949c'))
    s.append(circle(115, RY + 153, 30, 'none', '#e4e8ec', 1.5))
    for k in range(-3, 4):
        s.append(line(89 + 0, RY + 153 + k * 8, 141, RY + 153 + k * 8,
                      '#cfd4da', 0.8))
    s.append(text(115, RY + 218, 'segcrop frame', size=11, fill=INK, weight=600,
                  anchor='middle'))
    s.append(text(115, RY + 234, '256 × 256, uint8', size=10, fill=INK3,
                  anchor='middle'))

    # ---- preprocessing ----------------------------------------------------
    px, pw = 215, 180
    s.append(rect(px, RY + 70, pw, 180, PANEL_BG, RULE, rx=10))
    s.append(text(px + 16, RY + 96, 'PREPROCESS', size=9.5, fill=INK3,
                  weight=700, spacing=1.2))
    s.append(para(px + 16, RY + 122, [
        ('Resize → 256', INK, 600),
        'CenterCrop → 224 × 224',
        ('Grayscale → RGB', INK, 600),
        'replicate 1 ch → 3 ch so the',
        'ImageNet weights apply as-is',
        ('ImageNet normalisation', INK, 600),
        'μ = .485/.456/.406',
        'σ = .229/.224/.225',
    ], size=10, lh=15))

    # ---- TTA fan-out ------------------------------------------------------
    tx, tw = 420, 150
    s.append(rect(tx, RY + 70, tw, 180, BG, K, rx=10))
    s.append(text(tx + tw / 2, RY + 96, '8-FOLD TTA', size=9.5, fill=K,
                  weight=700, anchor='middle', spacing=1.2))
    # 4x2 grid of tiny oriented tiles
    marks = ['0°', '90°', '180°', '270°', '⇄0°', '⇄90°', '⇄180°', '⇄270°']
    for i, m in enumerate(marks):
        cx = tx + 22 + (i % 4) * 28
        cy = RY + 118 + (i // 4) * 34
        s.append(rect(cx, cy, 22, 22, '#ffffff', K, rx=3, sw=1))
        s.append(line(cx + 4, cy + 18, cx + 18, cy + 18, K, 1.6))
        s.append(text(cx + 11, cy + 34, m, size=7.5, fill=K, anchor='middle'))
    s.append(text(tx + tw / 2, RY + 214, 'dihedral group D₄', size=10.5, fill=INK,
                  weight=600, anchor='middle'))
    s.append(text(tx + tw / 2, RY + 230, '4 rotations × flip', size=10, fill=INK3,
                  anchor='middle'))
    s.append(text(tx + tw / 2, RY + 246, 'shared weights', size=10, fill=INK3,
                  anchor='middle'))

    # ---- backbone container ----------------------------------------------
    bx, bw = 595, 780
    s.append(rect(bx, RY, bw, RH, PANEL_BG, RULE, rx=12))
    s.append(f'<path d="M{bx},{RY+12} a12,12 0 0 1 12,-12 h{bw-24} '
             f'a12,12 0 0 1 12,12 v30 h{-bw} z" fill="{BG}"/>')
    s.append(line(bx, RY + 42, bx + bw, RY + 42, RULE, 1))
    s.append(text(bx + 16, RY + 27, 'Swin-Tiny backbone', size=13.5, fill=INK,
                  weight=700))
    s.append(text(bx + 176, RY + 27,
                  'swin_tiny_patch4_window7_224 · ImageNet-1k pretrained · '
                  '27.5 M params', size=10, fill=INK3, family=MONO))

    # patch embed
    pe_x, pe_w = 609, 96
    s.append(rect(pe_x, RY + 62, pe_w, 200, SOFT, RULE, rx=8))
    s.append(text(pe_x + pe_w / 2, RY + 88, 'Patch', size=11, fill=INK,
                  weight=700, anchor='middle'))
    s.append(text(pe_x + pe_w / 2, RY + 103, 'Embed', size=11, fill=INK,
                  weight=700, anchor='middle'))
    s.append(text(pe_x + pe_w / 2, RY + 126, '4 × 4 conv', size=10, fill=INK2,
                  anchor='middle'))
    s.append(text(pe_x + pe_w / 2, RY + 141, 'stride 4', size=10, fill=INK2,
                  anchor='middle'))
    s.append(line(pe_x + 14, RY + 156, pe_x + pe_w - 14, RY + 156, RULE, 1))
    s.append(text(pe_x + pe_w / 2, RY + 176, '224 × 224', size=10, fill=INK3,
                  anchor='middle'))
    s.append(text(pe_x + pe_w / 2, RY + 191, '↓', size=11, fill=INK3,
                  anchor='middle'))
    s.append(text(pe_x + pe_w / 2, RY + 208, '56 × 56', size=10.5, fill=K,
                  weight=700, anchor='middle'))
    s.append(text(pe_x + pe_w / 2, RY + 224, '× 96 ch', size=10.5, fill=K,
                  weight=700, anchor='middle'))

    # 4 stages
    stages = [
        ('Stage 1', '2 blocks', '56 × 56', '96', '3 heads', 56),
        ('Stage 2', '2 blocks', '28 × 28', '192', '6 heads', 28),
        ('Stage 3', '6 blocks', '14 × 14', '384', '12 heads', 14),
        ('Stage 4', '2 blocks', '7 × 7', '768', '24 heads', 7),
    ]
    sx, sw_, gap = 719, 154, 10
    for i, (nm, blocks, res, ch, heads, _) in enumerate(stages):
        x = sx + i * (sw_ + gap)
        s.append(rect(x, RY + 62, sw_, 200, PANEL_BG, K, rx=8, sw=1.3))
        s.append(rect(x, RY + 62, sw_, 26, BG, 'none', rx=8))
        s.append(rect(x, RY + 80, sw_, 8, BG, 'none', rx=0))
        s.append(text(x + 12, RY + 80, nm, size=11.5, fill=INK, weight=700))
        s.append(text(x + sw_ - 12, RY + 80, blocks, size=10, fill=K,
                      weight=600, anchor='end'))
        # block stack glyph
        nb = 2 if blocks.startswith('2') else 6
        for b in range(nb):
            yb = RY + 100 + b * (13 if nb == 6 else 20)
            lbl = 'W-MSA' if b % 2 == 0 else 'SW-MSA'
            s.append(rect(x + 12, yb, sw_ - 24, 11 if nb == 6 else 16,
                          BG if b % 2 == 0 else '#ffffff', K, rx=3, sw=0.9))
            s.append(text(x + sw_ / 2, yb + (8 if nb == 6 else 11.5), lbl,
                          size=7.8 if nb == 6 else 9, fill=K, weight=600,
                          anchor='middle'))
        s.append(line(x + 12, RY + 190, x + sw_ - 12, RY + 190, RULE, 1))
        s.append(text(x + 12, RY + 210, 'tokens', size=9.5, fill=INK3))
        s.append(text(x + sw_ - 12, RY + 210, res, size=10.5, fill=INK,
                      weight=600, anchor='end'))
        s.append(text(x + 12, RY + 227, 'dim', size=9.5, fill=INK3))
        s.append(text(x + sw_ - 12, RY + 227, ch, size=10.5, fill=INK,
                      weight=600, anchor='end'))
        s.append(text(x + 12, RY + 244, 'attn', size=9.5, fill=INK3))
        s.append(text(x + sw_ - 12, RY + 244, heads, size=10.5, fill=INK,
                      weight=600, anchor='end'))
        # patch-merging arrow between stages
        if i < 3:
            axm = x + sw_
            s.append(arrow(axm + 1, RY + 162, axm + gap - 1, RY + 162, K, 1.6))
            s.append(text(axm + gap / 2, RY + 152, '↓2', size=8, fill=K,
                          weight=700, anchor='middle'))
    s.append(arrow(pe_x + pe_w + 1, RY + 162, sx - 1, RY + 162, K, 1.6))
    s.append(text(bx + bw / 2, RY + 292, 'patch merging halves the token grid and '
                                         'doubles the channel width between stages',
                  size=10, fill=INK3, anchor='middle'))

    # ---- head -------------------------------------------------------------
    hx, hw = 1400, 240
    s.append(rect(hx, RY + 70, hw, 180, PANEL_BG, RULE, rx=10))
    s.append(text(hx + 16, RY + 96, 'HEAD', size=9.5, fill=INK3, weight=700,
                  spacing=1.2))
    s.append(para(hx + 16, RY + 122, [
        ('LayerNorm → global average pool', INK2),
        ('768-d particle embedding', INK, 600),
        ('Linear 768 → 3   (head re-initialised', INK2),
        ('on warm-start)', INK2),
    ], size=10, lh=15))
    s.append(rect(hx + 16, RY + 190, hw - 32, 46, BG, 'none', rx=6))
    s.append(text(hx + 28, RY + 209, 'softmax over 3 classes', size=10, fill=K,
                  weight=600))
    s.append(text(hx + 28, RY + 226, 'Dh   ·   FCC   ·   Ih-family',
                  size=10.5, fill=INK, weight=700))

    # row-1 arrows
    for a, b in [(190, 215), (395, 420), (570, 595), (1375, 1400)]:
        s.append(arrow(a + 3, MID, b - 3, MID, INK3, 1.8))

    # ============================ row 2 ====================================
    R2, R2H = 480, 300

    # ---- Swin block detail ------------------------------------------------
    dx, dw = 40, 520
    s.append(rect(dx, R2, dw, R2H, PANEL_BG, RULE, rx=12))
    s.append(text(dx + 20, R2 + 28, 'Inside a Swin block pair', size=14, fill=INK,
                  weight=700))
    s.append(text(dx + 20, R2 + 46, 'attention runs inside 7 × 7 windows; the window '
                                    'grid shifts by half a', size=10, fill=INK3))
    s.append(text(dx + 20, R2 + 60, 'window in the next block, which is what connects '
                                    'them', size=10, fill=INK3))
    s.append(line(dx + 20, R2 + 74, dx + dw - 20, R2 + 74, RULE, 1))

    # block A / block B chains
    for j, (label, attn, note) in enumerate([
            ('block 2k', 'W-MSA', 'windows are disjoint'),
            ('block 2k+1', 'SW-MSA', 'grid shifted by ⌊7/2⌋ = 3')]):
        by = R2 + 118 + j * 100
        s.append(text(dx + 20, by + 4, label, size=10.5, fill=INK, weight=700))
        chain = [('LN', 44), (attn, 70), ('+', 26), ('LN', 44), ('MLP ×4', 62),
                 ('+', 26)]
        cx = dx + 92
        for nm, ww in chain:
            fill = BG if nm in ('W-MSA', 'SW-MSA') else SOFT
            s.append(rect(cx, by - 12, ww, 30, fill, K if nm in
                          ('W-MSA', 'SW-MSA') else RULE, rx=6))
            s.append(text(cx + ww / 2, by + 7, nm, size=10,
                          fill=K if nm in ('W-MSA', 'SW-MSA') else INK2,
                          weight=700 if nm in ('W-MSA', 'SW-MSA') else 400,
                          anchor='middle'))
            if nm != '+':
                s.append(arrow(cx + ww + 2, by + 3, cx + ww + 10, by + 3, INK3, 1.4))
            cx += ww + 12
        # residual arcs
        s.append(f'<path d="M{dx+92},{by-15} C{dx+92},{by-36} {dx+236},{by-36} '
                 f'{dx+236},{by-15}" fill="none" stroke="{INK3}" '
                 f'stroke-width="1.2" stroke-dasharray="4 3"/>')
        s.append(text(dx + 20, by + 34, note, size=9.5, fill=INK3))

    s.append(text(dx + 20, R2 + 272,
                  'Why a windowed transformer here: five-fold twin symmetry is a '
                  'whole-particle relation,',
                  size=10, fill=INK2))

    # ---- TTA detail -------------------------------------------------------
    ax_, aw = 590, 380
    s.append(rect(ax_, R2, aw, R2H, PANEL_BG, K, rx=12, sw=1.3))
    s.append(text(ax_ + 20, R2 + 28, 'Why dihedral TTA is principled', size=14,
                  fill=INK, weight=700))
    s.append(line(ax_ + 20, R2 + 44, ax_ + aw - 20, R2 + 44, RULE, 1))
    s.append(para(ax_ + 20, R2 + 68, [
        ('The crystallographic phase of a particle is', INK2),
        ('invariant under rotation and reflection — a', INK2),
        ('decahedron rotated by 90° is still a decahedron.', INK2),
        ('', INK2),
        ('So the 8 dihedral views are 8 genuinely', INK2),
        ('equivalent observations of the same label.', INK2),
        ('Averaging their softmax outputs reduces', INK2),
        ('variance without introducing bias.', INK2),
    ], size=10.5, lh=15))
    s.append(rect(ax_ + 20, R2 + 200, aw - 40, 46, BG, 'none', rx=6))
    s.append(text(ax_ + 34, R2 + 220, 'p̄(c | frame)  =  ⅛ Σ  softmax( f( g·x ) )',
                  size=11, fill=K, weight=600, family=MONO))
    s.append(text(ax_ + 34, R2 + 238, 'g ∈ D₄,  no retraining, 8× inference cost',
                  size=9.5, fill=INK3))
    s.append(text(ax_ + 20, R2 + 268, '95.88 %  →  96.62 %', size=15, fill=K,
                  weight=700))
    s.append(text(ax_ + 190, R2 + 268, '(+0.74 pt, measured)', size=10, fill=INK3))

    # ---- temporal post-processing ----------------------------------------
    ox, ow = 1000, 640
    s.append(rect(ox, R2, ow, R2H, PANEL_BG, RULE, rx=12))
    s.append(text(ox + 20, R2 + 28, 'Temporal post-processing → phase timeline',
                  size=14, fill=INK, weight=700))
    s.append(line(ox + 20, R2 + 44, ox + ow - 20, R2 + 44, RULE, 1))

    tsteps = [
        ('Stack per-frame posteriors into a T × 3 array', None),
        ('Median-filter each class stream over 11 frames — the median '
         'kills isolated', None),
        ('single-frame flips but preserves a step, which is what a real '
         'phase change is', 'cont'),
        ('argmax → group equal-label runs → absorb runs shorter than '
         '10 frames', None),
        ('Inside each Ih-family segment, query a second model trained '
         'without', None),
        ('Ih→Dh: p(Ih) > 0.6 ∧ p(Ih) > p(Dh)+0.2 → Ih;  mirror → Dh;  '
         'otherwise Ih→Dh', 'cont'),
    ]
    yy = R2 + 70
    n = 0
    for txt_, cont in tsteps:
        if cont != 'cont':
            n += 1
            s.append(step_marker(ox + 30, yy - 4, n, K))
        s.append(text(ox + 48, yy, txt_, size=10.5, fill=INK2))
        yy += 16 if cont == 'cont' else 20

    # timeline glyph
    ty = R2 + 196
    s.append(text(ox + 20, ty - 8, 'output timeline', size=9.5, fill=INK3,
                  weight=700, spacing=1))
    segs = [(0.00, 0.30, '#2f6f9f', 'Ih'), (0.30, 0.42, ACCENT_RED, 'Ih→Dh'),
            (0.42, 0.74, '#1f7a68', 'Dh'), (0.74, 0.84, '#a06a14', 'FCC'),
            (0.84, 1.00, '#1f7a68', 'Dh')]
    tlx, tlw = ox + 20, ow - 40
    for a, b, c, lbl in segs:
        s.append(rect(tlx + a * tlw, ty, (b - a) * tlw, 34, c, 'none', rx=3,
                      op=0.85))
        s.append(text(tlx + (a + b) / 2 * tlw, ty + 22, lbl, size=10,
                      fill='#ffffff', weight=700, anchor='middle'))
    s.append(line(tlx, ty + 40, tlx + tlw, ty + 40, RULE, 1))
    for f in [0, 0.25, 0.5, 0.75, 1.0]:
        s.append(line(tlx + f * tlw, ty + 40, tlx + f * tlw, ty + 45, RULE, 1))
    s.append(text(tlx, ty + 60, 'frame 0', size=9.5, fill=INK3))
    s.append(text(tlx + tlw, ty + 60, 'frame T', size=9.5, fill=INK3,
                  anchor='end'))
    s.append(text(tlx + tlw / 2, ty + 60, 'time', size=9.5, fill=INK3,
                  anchor='middle'))
    s.append(text(ox + 20, R2 + 284,
                  'written as timeline.png, segments.csv (start / end / duration / '
                  'label / confidence) and an overlay MP4',
                  size=10, fill=INK3))

    # continuation line for the swin panel note
    s.append(text(dx + 20, R2 + 288,
                  'while the fringe evidence that separates the phases is local — '
                  'a shifted-window model has both.',
                  size=10, fill=INK2))

    # ---- footer -----------------------------------------------------------
    FY = 810
    s.append(rect(40, FY, W - 80, 150, SOFT, RULE, rx=12))
    s.append(text(64, FY + 30, 'Training configuration', size=13, fill=INK,
                  weight=700))
    s.append(line(64, FY + 42, W - 64, FY + 42, RULE, 1))
    cols = [
        ('Objective', ['cross-entropy, label smoothing 0.05',
                       'inverse-frequency class weights']),
        ('Sampler', ['WeightedRandomSampler, power 1.0',
                     '(classes fully equalised per epoch)']),
        ('Optimiser', ['AdamW, lr 1e-4 (warm-start) / 3e-4',
                       'weight decay 1e-4, cosine annealing']),
        ('Schedule', ['batch 32, 20–25 epochs, seed 44',
                      'best-val checkpoint kept']),
        ('Augmentation', ['RandomResizedCrop(224, 0.6–1.0), H/V flip',
                          'exact 0/90/180/270° rotation, jitter 0.2']),
        ('Warm start', ['backbone loaded from a prior 3-class ckpt;',
                        'needed to escape the imbalance minimum']),
    ]
    cw = (W - 128) / 3
    for i, (h, body) in enumerate(cols):
        cx = 64 + (i % 3) * cw
        cy = FY + 70 + (i // 3) * 46
        s.append(text(cx, cy, h, size=10.5, fill=K, weight=700))
        s.append(para(cx, cy + 16, [(b, INK2) for b in body], size=9.8, lh=13))

    return svg_doc(W, H, '\n'.join(s),
                   'Figure 2 — Swin-Tiny phase classifier with dihedral TTA')


if __name__ == '__main__':
    for name, fn in [('fig1_pipeline', figure1), ('fig2_classifier', figure2)]:
        path = os.path.join(OUT_DIR, name + '.svg')
        with open(path, 'w') as f:
            f.write(fn())
        print('wrote', path)
