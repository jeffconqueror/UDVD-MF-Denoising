#!/usr/bin/env python3
"""
Generate the two paper figures as standalone, editable SVG.

  fig1_pipeline.svg   — the full LC-TEM pipeline drawn as a network graph:
                        real data tensors at every stage, the UMVD denoiser
                        expanded inline, and expanded sub-network panels below.
  fig2_classifier.svg — the Stage-4 classifier: Swin-Tiny drawn as a feature-
                        volume diagram, with dihedral TTA and the temporal head.

Image thumbnails are real frames from the 053243 video, produced by
make_thumbnails.py and embedded as base64 so each SVG is self-contained.

    python make_thumbnails.py      # once, needs the umvd env + /shared
    python make_figures.py
    google-chrome --headless --disable-gpu --screenshot=fig1_pipeline.png \
        --window-size=1720,1010 --default-background-color=ffffff fig1_pipeline.svg

Every number in these figures is taken from the code; see
docs/PIPELINE_METHODS.md for the matching text.
"""
import base64
import html
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(OUT_DIR, 'assets')

# ---------------------------------------------------------------- palette ---
INK    = '#16191d'
INK2   = '#495159'
INK3   = '#727b84'
RULE   = '#d6dbe0'
PAPER  = '#ffffff'
SOFT   = '#f5f7f9'

FLOW     = '#2f6f9f'      # dataflow arrows
TEAL     = '#2e9e8f'      # named module blocks
TEAL_D   = '#1d6f65'
AMBER    = '#c2762a'      # expanded-network container
AMBER_L  = '#fdf3e6'
GREEN    = '#e2f0df'      # expanded sub-network panels
GREEN_S  = '#8fbf87'
PURPLE   = '#6a4c9c'
PURPLE_L = '#f1ecf9'
RED      = '#b3402f'

# layer-bar colours inside the expanded panels
BAR = {
    'conv': ('#f2c9a0', '#c2762a'),
    'norm': ('#a8cbe8', '#2f6f9f'),
    'act':  ('#c9b8e8', '#6a4c9c'),
    'pool': ('#f4b8ae', '#b3402f'),
    'attn': ('#a9dcd3', '#1d6f65'),
    'misc': ('#d8dde2', '#727b84'),
}

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', "
        "Arial, sans-serif")
MONO = "'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"


# ------------------------------------------------------------- primitives ---
def esc(s):
    return html.escape(str(s), quote=False)


def text(x, y, s, size=11, fill=INK2, weight=400, anchor='start', family=None,
         spacing=0, rotate=None):
    fam = family or FONT
    ls = f' letter-spacing="{spacing}"' if spacing else ''
    tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{ls}{tr}>'
            f'{esc(s)}</text>')


def rect(x, y, w, h, fill=PAPER, stroke=RULE, rx=6, sw=1, dash=None, op=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' fill-opacity="{op}"' if op is not None else ''
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')


def line(x1, y1, x2, y2, stroke=RULE, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>')


def arrow(x1, y1, x2, y2, stroke=FLOW, sw=2, dash=None, marker='ah_flow'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d} '
            f'marker-end="url(#{marker})"/>')


def poly_arrow(pts, stroke=FLOW, sw=2, dash=None, marker='ah_flow'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    return (f'<polyline points="{p}" fill="none" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"'
            f'{d} marker-end="url(#{marker})"/>')


def circle(cx, cy, r, fill, stroke='none', sw=1):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def op_node(cx, cy, sym, color=FLOW, r=13):
    """Element-wise operator node."""
    return (circle(cx, cy, r, '#ffffff', color, 2) +
            text(cx, cy + 5.5, sym, size=15, fill=color, weight=700,
                 anchor='middle'))


_IMG_CACHE = {}


def img(x, y, w, h, name, stroke=INK, sw=1.5):
    """Embed a real thumbnail as a base64 data URI (keeps the SVG standalone)."""
    if name not in _IMG_CACHE:
        with open(os.path.join(ASSETS, name), 'rb') as f:
            _IMG_CACHE[name] = base64.b64encode(f.read()).decode()
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" '
            f'preserveAspectRatio="none" '
            f'xlink:href="data:image/png;base64,{_IMG_CACHE[name]}"/>'
            + rect(x, y, w, h, 'none', stroke, rx=2, sw=sw))


def img_d(x, y, w, h, name, k, stroke=INK3, sw=1):
    """Thumbnail under the k-th dihedral transform (4 rotations x optional flip)."""
    cx_, cy_ = x + w / 2, y + h / 2
    t = f'rotate({(k % 4) * 90} {cx_:.1f} {cy_:.1f})'
    if k >= 4:
        t = f'translate({2*cx_:.1f},0) scale(-1,1) ' + t
    return (f'<g transform="{t}">' + img(x, y, w, h, name, stroke, sw) + '</g>')


def tensor(x, y, w, h, name, caption, shape, cap_above=True):
    """A data tensor: thumbnail with name and shape annotation."""
    o = [img(x, y, w, h, name)]
    if cap_above:
        o.append(text(x + w / 2, y - 20, caption, size=11.5, fill=INK,
                      weight=700, anchor='middle'))
        o.append(text(x + w / 2, y - 7, shape, size=10, fill=INK3,
                      anchor='middle', family=MONO))
    else:
        o.append(text(x + w / 2, y + h + 17, caption, size=11.5, fill=INK,
                      weight=700, anchor='middle'))
        o.append(text(x + w / 2, y + h + 30, shape, size=10, fill=INK3,
                      anchor='middle', family=MONO))
    return ''.join(o)


def module(x, y, w, h, label, sub=None, fill=TEAL, stroke=TEAL_D, tc='#ffffff'):
    """A named network module block."""
    o = [rect(x, y, w, h, fill, stroke, rx=8, sw=1.5)]
    ls = label.split('\n')
    y0 = y + h / 2 - (len(ls) - 1) * 7 + (0 if sub is None else -6)
    for i, l in enumerate(ls):
        o.append(text(x + w / 2, y0 + i * 14 + 4, l, size=11.5, fill=tc,
                      weight=700, anchor='middle'))
    if sub:
        o.append(text(x + w / 2, y + h - 9, sub, size=9, fill=tc,
                      anchor='middle', family=MONO))
    return ''.join(o)


def volume(x, y, fw, fh, d, face, top, side, label=None, sub=None):
    """A 3-D feature-map volume (front face + top and right parallelograms)."""
    o = [f'<polygon points="{x:.1f},{y:.1f} {x+d:.1f},{y-d:.1f} '
         f'{x+fw+d:.1f},{y-d:.1f} {x+fw:.1f},{y:.1f}" fill="{top}" '
         f'stroke="{INK3}" stroke-width="0.8"/>',
         f'<polygon points="{x+fw:.1f},{y:.1f} {x+fw+d:.1f},{y-d:.1f} '
         f'{x+fw+d:.1f},{y+fh-d:.1f} {x+fw:.1f},{y+fh:.1f}" fill="{side}" '
         f'stroke="{INK3}" stroke-width="0.8"/>',
         rect(x, y, fw, fh, face, INK3, rx=0, sw=0.8)]
    if label:
        o.append(text(x + fw / 2 + d / 2, y + fh + 16, label, size=9.5,
                      fill=INK, weight=700, anchor='middle', family=MONO))
    if sub:
        o.append(text(x + fw / 2 + d / 2, y + fh + 28, sub, size=8.5,
                      fill=INK3, anchor='middle'))
    return ''.join(o)


def panel(x, y, w, h, title, note=None):
    """A green expanded sub-network panel."""
    o = [rect(x, y, w, h, GREEN, GREEN_S, rx=10, sw=1.2),
         text(x + 14, y + 21, title, size=11.5, fill=INK, weight=700)]
    if note:
        o.append(text(x + w - 14, y + 21, note, size=9.5, fill=INK2,
                      anchor='end'))
    return ''.join(o)


def chain_row(x, y, items, bw=50, bh=118, gap=12):
    """A horizontal chain of labelled layer blocks with arrows between them."""
    o = []
    for i, (lbl, kind) in enumerate(items):
        bx = x + i * (bw + gap)
        o.append(rect(bx, y, bw, bh, BAR[kind][0], BAR[kind][1], rx=5))
        parts = lbl.split('\n')
        for j, l in enumerate(parts):
            o.append(text(bx + bw / 2, y + bh / 2 - (len(parts) - 1) * 6 + j * 12 + 4,
                          l, size=8.5, fill=INK, weight=600, anchor='middle'))
        if i < len(items) - 1:
            o.append(arrow(bx + bw + 1, y + bh / 2, bx + bw + gap - 2,
                           y + bh / 2, INK3, 1.4, marker='ah_ink'))
    return ''.join(o)


def note_lines(x, y, lines, size=9.5, lh=14, fill=INK2, family=None):
    return ''.join(text(x, y + i * lh, l, size=size, fill=fill, family=family)
                   for i, l in enumerate(lines))


def svg_doc(w, h, body, title):
    heads = ''.join(
        f'<marker id="{i}" markerWidth="9" markerHeight="9" refX="7.2" refY="3.2" '
        f'orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L7.4,3.2 L0,6.4 z" fill="{c}"/></marker>'
        for i, c in [('ah_flow', FLOW), ('ah_ink', INK3), ('ah_amber', AMBER),
                     ('ah_green', GREEN_S), ('ah_purple', PURPLE),
                     ('ah_red', RED)])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" '
            f'height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<title>{esc(title)}</title>\n<defs>{heads}</defs>\n'
            f'<rect width="{w}" height="{h}" fill="{PAPER}"/>\n{body}\n</svg>\n')


# ============================================================== FIGURE 1 ====
def figure1():
    W, H = 1760, 1060
    s = []

    s.append(text(40, 44, 'Figure 1  |  Pipeline architecture for crystallographic '
                          'phase analysis of liquid-cell TEM nanoparticle videos',
                  size=20, fill=INK, weight=700))
    s.append(text(40, 68, 'Every thumbnail is real data — frame 500 of the 053243 '
                          'PtRu/TiO₂ series, at the corresponding stage of the '
                          'pipeline', size=12, fill=INK3))
    s.append(line(40, 84, W - 40, 84, RULE, 1))

    # ---------------------------------------------------------------- row A --
    TY, TS = 152, 148
    MY = TY + TS / 2

    s.append(tensor(50, TY, TS, TS, '01_raw_full.png', 'Raw HRTEM movie',
                    'T × 2048 × 2048 × 1'))
    s.append(arrow(204, MY, 240, MY))
    s.append(module(244, MY - 44, 118, 88, 'Drift-\nCorrection Net', sub='Stage 1'))
    s.append(arrow(366, MY, 402, MY))
    s.append(tensor(406, TY, TS, TS, '02_aligned.png', 'Aligned stack',
                    'T × 512 × 512 × 1'))
    s.append(arrow(560, MY, 596, MY))

    # --- UMVD expanded inline ---------------------------------------------
    cx, cw = 600, 860
    s.append(rect(cx, TY - 52, cw, TS + 116, AMBER_L, AMBER, rx=14, sw=2))
    s.append(text(cx + 16, TY - 32, 'UMVD Denoiser  —  self-supervised, '
                                    'blind-frame temporal interpolation',
                  size=12, fill=AMBER, weight=700))
    s.append(text(cx + cw - 16, TY - 32, 'Stage 2  ·  1.79 M params', size=10,
                  fill=AMBER, weight=700, anchor='end'))

    # 7-frame input window, centre frame marked as masked
    fx, fy = cx + 20, TY + 14
    for k in range(7):
        blind = (k == 3)
        s.append(rect(fx + k * 9, fy, 28, 62,
                      '#fdeceb' if blind else '#ffffff',
                      RED if blind else INK3, rx=2, sw=1.8 if blind else 0.8))
    s.append(text(fx + 45, fy + 80, 'I t−3 … I t+3', size=9.5, fill=INK,
                  weight=700, anchor='middle', family=MONO))
    s.append(text(fx + 45, fy + 92, '7-frame window', size=9, fill=INK3,
                  anchor='middle'))
    s.append(text(fx + 41, fy - 6, 'I t', size=9.5, fill=RED, weight=700,
                  anchor='middle', family=MONO))
    s.append(arrow(fx + 100, MY, fx + 124, MY, AMBER, 1.8, marker='ah_amber'))

    # depthwise conv stack
    dx = fx + 128
    for k in range(3):
        s.append(rect(dx + k * 21, MY - 42, 17, 84, BAR['conv'][0],
                      BAR['conv'][1], rx=4))
    s.append(text(dx + 27, MY - 64, 'DepthConv 3×3  ×3', size=9, fill=INK,
                  weight=600, anchor='middle'))
    s.append(text(dx + 27, MY - 52, '→ 21 ch/frame, no mixing', size=9,
                  fill=INK3, anchor='middle'))

    # blind-frame mask
    mx = dx + 92
    s.append(arrow(dx + 65, MY, mx - 16, MY, AMBER, 1.8, marker='ah_amber'))
    s.append(op_node(mx, MY, '×', RED, 14))
    s.append(text(mx, MY - 26, 'blind-frame mask', size=9.5, fill=RED,
                  weight=700, anchor='middle'))
    s.append(text(mx + 22, MY + 58, 'w = [1,1,1,0,1,1,1]', size=9.5, fill=RED,
                  weight=700, anchor='middle', family=MONO))
    s.append(text(mx + 22, MY + 70, 'centre frame zeroed', size=9, fill=RED,
                  anchor='middle'))

    # concat
    ux = mx + 56
    s.append(arrow(ux, MY, ux + 20, MY, AMBER, 1.8, marker='ah_amber'))
    s.append(rect(ux + 24, MY - 42, 19, 84, BAR['misc'][0], BAR['misc'][1], rx=4))
    s.append(text(ux + 33, MY, 'concat 147 ch', size=8.5, fill=INK, weight=600,
                  anchor='middle', rotate=-90))
    s.append(arrow(ux + 45, MY, ux + 62, MY, AMBER, 1.8, marker='ah_amber'))

    # U-Net glyph
    nx, ny = ux + 68, MY
    for ex, eh, ch in [(0, 84, '48'), (25, 60, '48'), (50, 40, '48')]:
        s.append(rect(nx + ex, ny - eh / 2, 19, eh, BAR['conv'][0],
                      BAR['conv'][1], rx=3))
        s.append(text(nx + ex + 9, ny + eh / 2 + 10, ch, size=8, fill=INK3,
                      anchor='middle', family=MONO))
    for ex, eh, ch in [(75, 60, '96'), (100, 84, '96')]:
        s.append(rect(nx + ex, ny - eh / 2, 19, eh, BAR['act'][0],
                      BAR['act'][1], rx=3))
        s.append(text(nx + ex + 9, ny + eh / 2 + 10, ch, size=8, fill=INK3,
                      anchor='middle', family=MONO))
    s.append(f'<path d="M{nx+9},{ny-45} C{nx+28},{ny-68} {nx+91},{ny-68} '
             f'{nx+110},{ny-45}" fill="none" stroke="{INK3}" stroke-width="1.1" '
             f'stroke-dasharray="4 3"/>')
    s.append(f'<path d="M{nx+34},{ny-33} C{nx+48},{ny-52} {nx+72},{ny-52} '
             f'{nx+85},{ny-33}" fill="none" stroke="{INK3}" stroke-width="1.1" '
             f'stroke-dasharray="4 3"/>')
    s.append(text(nx + 59, ny + 62, 'U-Net (skip-connected)', size=9, fill=INK2,
                  anchor='middle'))
    s.append(text(nx + 59, ny - 74, 'max-pool ↓2   ·   nearest ↑2', size=8.5,
                  fill=INK3, anchor='middle'))

    # 1x1 head
    hx = nx + 128
    s.append(arrow(hx - 9, MY, hx + 1, MY, AMBER, 1.8, marker='ah_amber'))
    for k, (ch, hh) in enumerate([('96', 40), ('384', 76), ('96', 40), ('1', 18)]):
        s.append(rect(hx + 5 + k * 19, MY - hh / 2, 15, hh, BAR['norm'][0],
                      BAR['norm'][1], rx=3))
        s.append(text(hx + 12 + k * 19, MY + hh / 2 + 10, ch, size=8, fill=INK3,
                      anchor='middle', family=MONO))
    s.append(text(hx + 43, MY - 50, '1×1 conv head', size=9, fill=INK2,
                  anchor='middle'))

    s.append(arrow(cx + cw - 4, MY, cx + cw + 30, MY))
    s.append(tensor(cx + cw + 34, TY, TS, TS, '03_denoised_umvd.png', 'Denoised',
                    'T × 512 × 512 × 1'))

    # self-supervised loss feedback
    lx = cx + cw + 34 + TS / 2
    s.append(poly_arrow([(lx, TY + TS + 8), (lx, TY + TS + 34),
                         (fx + 45, TY + TS + 34), (fx + 45, TY + TS + 18)],
                        RED, 1.5, dash='5 4', marker='ah_red'))
    s.append(text((lx + fx) / 2, TY + TS + 48,
                  'self-supervised loss      L = ‖ f( masked window ) − I t ‖² / 2',
                  size=10.5, fill=RED, weight=700, anchor='middle', family=MONO))
    s.append(text((lx + fx) / 2, TY + TS + 62,
                  'the target is the NOISY centre frame — the network never sees a '
                  'clean reference, because none exists for a fluxional surface',
                  size=9.5, fill=RED, anchor='middle'))

    # elbow into row B
    s.append(poly_arrow([(lx, TY + TS + 76), (lx, 412), (124, 412), (124, 448)],
                        FLOW, 2))

    # ---------------------------------------------------------------- row B --
    BY = 452
    BM = BY + TS / 2

    s.append(tensor(50, BY, TS, TS, '03_denoised_umvd.png', 'Denoised',
                    'T × 512 × 512 × 1', cap_above=False))
    s.append(arrow(204, BM, 240, BM))
    s.append(module(244, BM - 44, 118, 88, 'SAM 3\nSegmenter', sub='Stage 3'))
    s.append(arrow(366, BM, 396, BM))
    s.append(op_node(414, BM, '⊗', TEAL_D, 15))
    s.append(text(414, BM - 28, 'mask ∩ centroid crop', size=9.5, fill=TEAL_D,
                  weight=600, anchor='middle'))
    s.append(arrow(432, BM, 462, BM))
    s.append(tensor(466, BY, TS, TS, '04_segcrop.png', 'Segmented crop',
                    'T × 256 × 256 × 1', cap_above=False))
    s.append(arrow(620, BM, 656, BM))
    s.append(module(660, BM - 44, 152, 88, 'Phase Classifier',
                    sub='Stage 4 — Fig. 2', fill=PURPLE, stroke='#4b3570'))
    s.append(arrow(816, BM, 852, BM))

    # per-frame posterior heat strip
    px, pw2 = 856, 100
    s.append(rect(px, BM - 46, pw2, 92, '#ffffff', INK3, rx=4))
    for r in range(3):
        for c in range(6):
            v = ([0.10, 0.16, 0.26, 0.55, 0.72, 0.80][c] if r == 0 else
                 (0.08 if r == 1 else [0.82, 0.74, 0.62, 0.30, 0.15, 0.10][c]))
            col = ['#2f6f9f', '#c2762a', '#1d6f65'][r]
            s.append(rect(px + 7 + c * 15, BM - 38 + r * 28, 13, 22, col,
                          'none', rx=2, op=0.12 + 0.88 * v))
    for r, lb in enumerate(['Dh', 'FCC', 'Ih']):
        s.append(text(px - 5, BM - 22 + r * 28, lb, size=8.5, fill=INK3,
                      anchor='end'))
    s.append(text(px + pw2 / 2, BM + 63, 'per-frame posterior', size=11.5,
                  fill=INK, weight=700, anchor='middle'))
    s.append(text(px + pw2 / 2, BM + 76, 'T × 3', size=10, fill=INK3,
                  anchor='middle', family=MONO))
    s.append(arrow(px + pw2 + 4, BM, px + pw2 + 38, BM))

    s.append(module(px + pw2 + 42, BM - 44, 138, 88, 'Temporal\nSmoothing Head',
                    sub='median 11 + rules', fill=PURPLE, stroke='#4b3570'))
    s.append(arrow(px + pw2 + 184, BM, px + pw2 + 218, BM))

    tlx, tlw = px + pw2 + 222, 372
    segs = [(0.00, 0.30, '#2f6f9f', 'Ih'), (0.30, 0.42, RED, 'Ih→Dh'),
            (0.42, 0.76, '#1d6f65', 'Dh'), (0.76, 0.86, '#c2762a', 'FCC'),
            (0.86, 1.00, '#1d6f65', 'Dh')]
    for a, b, c, lbl in segs:
        s.append(rect(tlx + a * tlw, BM - 22, (b - a) * tlw, 44, c, 'none',
                      rx=3, op=0.88))
        if (b - a) > 0.09:
            s.append(text(tlx + (a + b) / 2 * tlw, BM + 4, lbl, size=10.5,
                          fill='#ffffff', weight=700, anchor='middle'))
    s.append(rect(tlx, BM - 22, tlw, 44, 'none', INK3, rx=3, sw=1))
    s.append(text(tlx + tlw / 2, BM - 34, 'Phase timeline', size=11.5, fill=INK,
                  weight=700, anchor='middle'))
    s.append(text(tlx, BM + 38, 'frame 0', size=9, fill=INK3))
    s.append(text(tlx + tlw / 2, BM + 38, 'time →', size=9, fill=INK3,
                  anchor='middle'))
    s.append(text(tlx + tlw, BM + 38, 'frame T', size=9, fill=INK3, anchor='end'))

    # ---------------------------------------------- expanded sub-networks ----
    GY, GH = 700, 258
    pw3, gap = 404, 26
    xs = [50, 50 + pw3 + gap, 50 + 2 * (pw3 + gap), 50 + 3 * (pw3 + gap)]

    # (a) drift-correction net
    x = xs[0]
    s.append(panel(x, GY, pw3, GH, 'Drift-Correction Net', 'drift_correction.py'))
    s.append(chain_row(x + 16, GY + 40, [
        ('Band-pass\nDoG σ 1.5/15', 'conv'), ('TurboReg\nTRANSLATION', 'attn'),
        ('Moving avg\nN = 9', 'misc'), ('Median\n11–31', 'pool'),
        ('Savitzky–\nGolay 51–71', 'norm'), ('Warp raw\nframes', 'act')]))
    s.append(note_lines(x + 16, GY + 182, [
        'registration runs on the band-passed frames; the smoothed shifts',
        'are applied to the raw ones. Trajectory filtering is two-stage —',
        'the median rejects registration spikes, Savitzky–Golay removes',
        'jitter while preserving the slow physical drift.']))
    s.append(text(x + 16, GY + 246, 'max single-frame step   250–870 px → 3–5 px',
                  size=10, fill=RED, weight=700))

    # (b) UMVD U-Net detail
    x = xs[1]
    s.append(panel(x, GY, pw3, GH, 'UMVD U-Net (expanded)', 'UMVD/model.py'))
    rows = [('encoder 1', '147 → 48', '5 × conv3×3 + ReLU, max-pool ↓2'),
            ('encoder 2', '48 → 48', '5 × conv3×3 + ReLU, max-pool ↓2'),
            ('encoder 3', '48 → 48', '5 × conv3×3 + ReLU, no ↓'),
            ('decoder 2', '96 → 96', 'up ↑2, concat enc-1, 6 × conv3×3'),
            ('decoder 1', '96+147 → 96', 'up ↑2, concat input, 6 × conv3×3'),
            ('head', '96→384→96→1', '3 × conv1×1 + ReLU')]
    for i, (nm, ch, note) in enumerate(rows):
        yy = GY + 46 + i * 25
        s.append(rect(x + 14, yy - 13, pw3 - 28, 23,
                      '#ffffff' if i % 2 == 0 else 'none', 'none', rx=4))
        s.append(text(x + 24, yy + 3, nm, size=9.5, fill=INK, weight=700))
        s.append(text(x + 100, yy + 3, ch, size=9, fill=BAR['conv'][1],
                      weight=700, family=MONO))
        s.append(text(x + 196, yy + 3, note, size=9, fill=INK2))
    s.append(note_lines(x + 16, GY + 210, [
        'replication padding throughout, no bias anywhere, ReLU activations.',
        'Adam, lr 1e-3 halved every 10 epochs, 128² patches.']))
    s.append(text(x + 16, GY + 246, 'transfer ≈ fine-tune ≈ from-scratch on this '
                                    'data', size=10, fill=RED, weight=700))

    # (c) SAM 3 crop
    x = xs[2]
    s.append(panel(x, GY, pw3, GH, 'SAM 3 Segment + Tight Crop',
                   'seg_crop_speedup.py'))
    s.append(chain_row(x + 16, GY + 40, [
        ('Box prompt\n[.5,.5,.4,.4]', 'attn'), ('SAM 3\nimage model', 'conv'),
        ('Select mask\nscore ≥ 0.3', 'norm'), ('Area gate\n5k–600k px', 'pool'),
        ('Centroid\nsquare ×0.6', 'act'), ('Resize\n256 × 256', 'misc')]))
    s.append(note_lines(x + 16, GY + 182, [
        'the crop is centred on the mask CENTROID, not the bounding-box',
        'centre — the centroid stays stable when the mask boundary flickers.',
        'Frames with no admissible mask are dropped, which is why T shrinks',
        'between Stage 3 and Stage 4.']))
    s.append(text(x + 16, GY + 246, 'particle fills a constant fraction of every '
                                    'frame', size=10, fill=RED, weight=700))

    # (d) temporal head
    x = xs[3]
    s.append(panel(x, GY, pw3, GH, 'Temporal Smoothing Head',
                   'classifier/classify_video.py'))
    s.append(chain_row(x + 16, GY + 40, [
        ('Median\n11 frames', 'pool'), ('argmax\nper frame', 'norm'),
        ('Group\nruns', 'misc'), ('Absorb\n< 10 fr', 'act'),
        ('Ih→Dh\nrule', 'attn'), ('Segment\ntable', 'conv')]))
    s.append(note_lines(x + 16, GY + 182, [
        'a median (not a mean) suppresses isolated single-frame flips while',
        'preserving a step — which is what a real phase change is. Inside an',
        'Ih-family segment a second (drop-Ih→Dh) model decides:']))
    s.append(text(x + 16, GY + 226, 'p(Ih)>.6 ∧ p(Ih)>p(Dh)+.2 → Ih;  mirror → Dh;'
                                    '  else Ih→Dh', size=9, fill=INK2,
                  family=MONO))
    s.append(text(x + 16, GY + 246, 'recovers the class a single frame cannot '
                                    'decide', size=10, fill=RED, weight=700))

    # dashed connectors module → expanded panel
    # routed through the vertical corridors that stay clear of row B
    s.append(poly_arrow([(303, MY + 46), (303, 340), (220, 340), (220, 664),
                         (xs[0] + pw3 / 2, 664), (xs[0] + pw3 / 2, GY - 4)],
                        GREEN_S, 1.6, dash='5 4', marker='ah_green'))
    s.append(poly_arrow([(nx + 59, TY + TS + 82), (nx + 59, 392), (1157, 392),
                         (1157, 670), (xs[1] + pw3 / 2, 670),
                         (xs[1] + pw3 / 2, GY - 4)], GREEN_S, 1.6, dash='5 4',
                        marker='ah_green'))
    s.append(poly_arrow([(303, BM + 46), (303, 628), (380, 628), (380, 676),
                         (xs[2] + pw3 / 2, 676), (xs[2] + pw3 / 2, GY - 4)],
                        GREEN_S, 1.6, dash='5 4', marker='ah_green'))
    s.append(poly_arrow([(px + pw2 + 111, BM + 46), (px + pw2 + 111, 628),
                         (977, 628), (977, 682), (xs[3] + pw3 / 2, 682),
                         (xs[3] + pw3 / 2, GY - 4)], GREEN_S, 1.6, dash='5 4',
                        marker='ah_green'))

    s.append(text(40, H - 18, 'Stages 1–3 are chained by pipeline.py; Stage 4 runs '
                              'on the resulting crop stack. All four stages are '
                              'self-supervised or pre-trained — the only '
                              'hand-labelled data in the pipeline trains the '
                              'Stage-4 classifier.', size=10, fill=INK3))
    return svg_doc(W, H, '\n'.join(s),
                   'Figure 1 — LC-TEM nanoparticle analysis pipeline')


# ============================================================== FIGURE 2 ====
def figure2():
    W, H = 1900, 1010
    s = []

    s.append(text(40, 44, 'Figure 2  |  Stage-4 phase classifier: Swin-Tiny with '
                          '8-fold dihedral test-time augmentation',
                  size=20, fill=INK, weight=700))
    s.append(text(40, 68, 'One segmented crop per frame → 8 symmetry-equivalent '
                          'views → shared Swin-Tiny → averaged posterior → '
                          'per-frame phase label', size=12, fill=INK3))
    s.append(line(40, 84, W - 40, 84, RULE, 1))

    RY = 214

    # ---- input + preprocessing + TTA fan ----------------------------------
    s.append(tensor(46, RY - 64, 128, 128, '04_segcrop.png', 'Segmented crop',
                    '256 × 256 × 1'))
    s.append(arrow(178, RY, 210, RY))
    s.append(module(214, RY - 48, 92, 96, 'Pre-\nprocess', sub='224², RGB',
                    fill='#7fa8c9', stroke='#2f6f9f'))
    s.append(text(260, RY + 64, 'Resize 256 → CenterCrop 224', size=9, fill=INK3,
                  anchor='middle'))
    s.append(text(260, RY + 76, '1 ch → 3 ch  ·  ImageNet norm', size=9,
                  fill=INK3, anchor='middle'))
    s.append(arrow(310, RY, 342, RY))

    for k in range(8):
        s.append(img_d(346 + k * 5, RY - 44 - k * 5, 72, 72, '04_segcrop.png',
                       7 - k))
    s.append(text(400, RY + 46, '8 dihedral views', size=11, fill=INK,
                  weight=700, anchor='middle'))
    s.append(text(400, RY + 60, '4 rotations × flip   (group D₄)', size=9.5,
                  fill=INK3, anchor='middle'))
    s.append(text(400, RY + 74, 'shared weights', size=9.5,
                  fill=INK3, anchor='middle'))
    s.append(arrow(428, RY, 460, RY))

    # ---- backbone container ----------------------------------------------
    cx, cw = 464, 1010
    s.append(rect(cx, RY - 132, cw, 274, AMBER_L, AMBER, rx=14, sw=2))
    s.append(text(cx + 16, RY - 112, 'Swin-Tiny backbone', size=12.5, fill=AMBER,
                  weight=700))
    s.append(text(cx + 172, RY - 112, 'swin_tiny_patch4_window7_224  ·  '
                                      'ImageNet-1k pretrained  ·  27.5 M params',
                  size=9.5, fill=AMBER, family=MONO))

    FACE, TOP, SIDE = '#cfe0ee', '#e8f1f8', '#a8c6dd'

    def stage_block(x, y, w, h, title, sub, n_blk):
        o = [rect(x, y, w, h, '#ffffff', AMBER, rx=7, sw=1.4),
             f'<path d="M{x},{y+7} a7,7 0 0 1 7,-7 h{w-14} a7,7 0 0 1 7,7 '
             f'v14 h{-w} z" fill="#f7e4cd"/>',
             text(x + w / 2, y + 15, title, size=9.5, fill=INK, weight=700,
                  anchor='middle')]
        avail = h - 44
        for b in range(n_blk):
            hb = avail / n_blk - 3
            yb = y + 27 + b * (avail / n_blk)
            lbl = 'W-MSA' if b % 2 == 0 else 'SW-MSA'
            o.append(rect(x + 6, yb, w - 12, hb,
                          '#e8f2f9' if b % 2 == 0 else '#ffffff', AMBER,
                          rx=3, sw=0.8))
            if hb >= 9:
                o.append(text(x + w / 2, yb + hb / 2 + 3, lbl,
                              size=7 if n_blk > 2 else 8.5, fill=INK2,
                              weight=600, anchor='middle'))
        o.append(text(x + w / 2, y + h - 7, sub, size=8, fill=AMBER,
                      weight=700, anchor='middle'))
        return ''.join(o)

    x = cx + 20
    s.append(volume(x, RY - 30, 54, 54, 14, FACE, TOP, SIDE, '224×224×3',
                    'input'))
    x += 82
    s.append(arrow(x - 16, RY, x - 3, RY, AMBER, 1.6, marker='ah_amber'))
    s.append(rect(x, RY - 46, 54, 92, '#ffffff', AMBER, rx=7, sw=1.4))
    s.append(text(x + 27, RY - 12, 'Patch', size=9.5, fill=INK, weight=700,
                  anchor='middle'))
    s.append(text(x + 27, RY + 1, 'Embed', size=9.5, fill=INK, weight=700,
                  anchor='middle'))
    s.append(text(x + 27, RY + 17, '4×4 conv', size=8, fill=INK3,
                  anchor='middle', family=MONO))
    s.append(text(x + 27, RY + 28, 'stride 4', size=8, fill=INK3,
                  anchor='middle', family=MONO))
    x += 70

    specs = [(46, 20, '56×56', '96', 'Stage 1', 2, '2 blocks · 3 heads'),
             (37, 27, '28×28', '192', 'Stage 2', 2, '2 blocks · 6 heads'),
             (28, 36, '14×14', '384', 'Stage 3', 6, '6 blocks · 12 heads'),
             (20, 45, '7×7', '768', 'Stage 4', 2, '2 blocks · 24 heads')]
    for i, (fs, dp, res, ch, title, nb, sub) in enumerate(specs):
        s.append(arrow(x - 16, RY, x - 3, RY, AMBER, 1.6, marker='ah_amber'))
        s.append(stage_block(x, RY - 52, 80, 104, title, sub, nb))
        x += 94
        s.append(arrow(x - 16, RY, x - 3, RY, AMBER, 1.6, marker='ah_amber'))
        s.append(volume(x, RY - fs / 2, fs, fs, dp, FACE, TOP, SIDE,
                        f'{res}×{ch}', 'tokens × dim'))
        if i < 3:
            s.append(text(x + fs / 2, RY - 62, '↓2', size=9, fill=AMBER,
                          weight=700, anchor='middle'))
        x += fs + dp + 22

    s.append(arrow(x - 16, RY, x - 3, RY, AMBER, 1.6, marker='ah_amber'))
    s.append(rect(x, RY - 32, 48, 64, '#ffffff', AMBER, rx=7, sw=1.4))
    s.append(text(x + 24, RY - 6, 'LN', size=9.5, fill=INK, weight=700,
                  anchor='middle'))
    s.append(text(x + 24, RY + 9, 'GAP', size=9.5, fill=INK, weight=700,
                  anchor='middle'))
    x += 62
    s.append(arrow(x - 16, RY, x - 3, RY, AMBER, 1.6, marker='ah_amber'))
    s.append(rect(x, RY - 40, 14, 80, BAR['norm'][0], BAR['norm'][1], rx=3))
    s.append(text(x + 7, RY + 54, '768-d', size=8.5, fill=INK3, anchor='middle',
                  family=MONO))
    x += 30
    s.append(arrow(x - 16, RY, x - 3, RY, AMBER, 1.6, marker='ah_amber'))
    s.append(rect(x, RY - 18, 14, 36, BAR['act'][0], BAR['act'][1], rx=3))
    s.append(text(x + 7, RY + 52, 'FC → 3', size=8.5, fill=INK3, anchor='middle',
                  family=MONO))

    # ---- TTA average ------------------------------------------------------
    ax_ = cx + cw + 42
    s.append(arrow(cx + cw - 2, RY, ax_ - 20, RY))
    s.append(op_node(ax_, RY, '⊕', PURPLE, 16))
    s.append(text(ax_, RY - 36, 'mean over the', size=9.5, fill=PURPLE,
                  weight=600, anchor='middle'))
    s.append(text(ax_, RY - 24, '8 dihedral views', size=9.5, fill=PURPLE,
                  weight=600, anchor='middle'))
    s.append(text(ax_ - 10, RY + 78, 'p̄ = ⅛ Σ softmax( f(g·x) ),  g ∈ D₄',
                  size=9, fill=PURPLE, anchor='middle', family=MONO))

    bx = ax_ + 42
    s.append(arrow(ax_ + 20, RY, bx - 6, RY))
    s.append(text(bx, RY - 70, 'per-frame posterior', size=11, fill=INK,
                  weight=700))
    for i, (cls, p, col) in enumerate([('Dh', 0.03, '#2f6f9f'),
                                       ('FCC', 0.02, '#c2762a'),
                                       ('Ih-family', 0.95, '#1d6f65')]):
        yy = RY - 52 + i * 38
        s.append(text(bx, yy + 12, cls, size=10, fill=INK, weight=700))
        s.append(rect(bx + 66, yy + 2, 104, 13, SOFT, 'none', rx=3))
        s.append(rect(bx + 66, yy + 2, 104 * p, 13, col, 'none', rx=3))
        s.append(text(bx + 178, yy + 12, f'{p:.2f}', size=9.5, fill=INK2,
                      family=MONO))

    # ---------------------------------------------- expanded sub-networks ----
    GY, GH = 430, 250
    pw3, gap = 404, 26
    xs = [50, 50 + pw3 + gap, 50 + 2 * (pw3 + gap), 50 + 3 * (pw3 + gap)]

    # (a) Swin block pair
    x = xs[0]
    s.append(panel(x, GY, pw3, GH, 'Swin Transformer block (×2)',
                   'shifted-window attention'))
    for j, (lbl, attn) in enumerate([('block 2k', 'W-MSA'),
                                     ('block 2k+1', 'SW-MSA')]):
        by = GY + 68 + j * 82
        s.append(text(x + 16, by + 4, lbl, size=9.5, fill=INK, weight=700))
        chain = [('LN', 32, 'norm'), (attn, 60, 'attn'), ('⊕', 22, 'misc'),
                 ('LN', 32, 'norm'), ('MLP ×4', 54, 'act'), ('⊕', 22, 'misc')]
        cxx = x + 84
        for nm, ww, kind in chain:
            s.append(rect(cxx, by - 12, ww, 30, BAR[kind][0], BAR[kind][1], rx=5))
            s.append(text(cxx + ww / 2, by + 7, nm, size=9, fill=INK, weight=600,
                          anchor='middle'))
            if nm != '⊕':
                s.append(arrow(cxx + ww + 1, by + 3, cxx + ww + 8, by + 3, INK3,
                               1.3, marker='ah_ink'))
            cxx += ww + 9
        s.append(f'<path d="M{x+84},{by-14} C{x+84},{by-32} {x+208},{by-32} '
                 f'{x+208},{by-14}" fill="none" stroke="{INK3}" '
                 f'stroke-width="1.1" stroke-dasharray="4 3"/>')
        s.append(f'<path d="M{x+215},{by-14} C{x+215},{by-32} {x+340},{by-32} '
                 f'{x+340},{by-14}" fill="none" stroke="{INK3}" '
                 f'stroke-width="1.1" stroke-dasharray="4 3"/>')
    s.append(note_lines(x + 16, GY + 216, [
        'two residual sub-blocks: windowed self-attention, then a 4× MLP.',
        'The pair is the unit — attention only crosses windows via the shift.']))

    # (b) window vs shifted window
    x = xs[1]
    s.append(panel(x, GY, pw3, GH, 'W-MSA vs SW-MSA', 'why the shift matters'))
    for j, (title, shifted) in enumerate([('W-MSA — disjoint 7×7 windows', False),
                                          ('SW-MSA — grid shifted by 3', True)]):
        gx0 = x + 30 + j * 196
        s.append(text(gx0 + 61, GY + 46, title, size=9, fill=INK, weight=700,
                      anchor='middle'))
        G, cell = 122, 122 / 14
        gy0 = GY + 56
        s.append(rect(gx0, gy0, G, G, '#ffffff', INK3, rx=2))
        for r in range(15):
            s.append(line(gx0, gy0 + r * cell, gx0 + G, gy0 + r * cell, '#c3cad1', 0.6))
            s.append(line(gx0 + r * cell, gy0, gx0 + r * cell, gy0 + G, '#c3cad1', 0.6))
        sh = (cell * 3.5) if shifted else 0
        for k in range(-1, 4):
            v = sh + k * cell * 7
            if 0 < v < G:
                s.append(line(gx0 + v, gy0, gx0 + v, gy0 + G, TEAL_D, 2.2))
                s.append(line(gx0, gy0 + v, gx0 + G, gy0 + v, TEAL_D, 2.2))
        s.append(rect(gx0, gy0, G, G, 'none', TEAL_D, rx=2, sw=2.2))
    s.append(note_lines(x + 16, GY + 200, [
        'consecutive blocks alternate the two, so information crosses window',
        'boundaries. Long-range structure — a five-fold twin spanning the whole',
        'particle — becomes visible while attention stays local and cheap.']))

    # (c) dihedral TTA
    x = xs[2]
    s.append(panel(x, GY, pw3, GH, 'Why dihedral TTA is principled',
                   '+0.74 pt, no retraining'))
    for k in range(8):
        gx0 = x + 24 + (k % 4) * 94
        gy0 = GY + 44 + (k // 4) * 64
        s.append(img_d(gx0, gy0, 44, 44, '05_class_IhDh.png', k))
        s.append(text(gx0 + 22, gy0 + 57, ['0°', '90°', '180°', '270°', '⇄ 0°',
                                           '⇄ 90°', '⇄ 180°', '⇄ 270°'][k],
                      size=8.5, fill=INK2, anchor='middle'))
        if k % 4 < 3:
            s.append(arrow(gx0 + 48, gy0 + 22, gx0 + 62, gy0 + 22, INK3, 1.2,
                           marker='ah_ink'))
    s.append(note_lines(x + 16, GY + 190, [
        'the crystallographic phase is invariant under rotation and reflection',
        '— a decahedron turned 90° is still a decahedron. These are therefore 8',
        'genuinely equivalent observations of one label, and averaging them',
        'reduces variance without introducing bias.']))

    # (d) the classes
    x = xs[3]
    s.append(panel(x, GY, pw3, GH, 'The three classes', '4-class → 3-class'))
    for k, (thumb, nm, note) in enumerate([('05_class_Dh.png', 'Dh', 'decahedral'),
                                           ('05_class_Ih.png', 'Ih', 'icosahedral'),
                                           ('05_class_IhDh.png', 'Ih→Dh',
                                            'transition')]):
        gx0 = x + 26 + k * 124
        s.append(img(gx0, GY + 40, 96, 96, thumb, INK3, 1.2))
        s.append(text(gx0 + 48, GY + 152, nm, size=11, fill=INK, weight=700,
                      anchor='middle'))
        s.append(text(gx0 + 48, GY + 165, note, size=9, fill=INK3,
                      anchor='middle'))
    s.append(f'<path d="M{x+174},{GY+172} C{x+174},{GY+186} {x+398},{GY+186} '
             f'{x+398},{GY+172}" fill="none" stroke="{RED}" stroke-width="1.6"/>')
    s.append(text(x + 286, GY + 196, 'merged', size=9, fill=RED, weight=700,
                  anchor='middle'))
    s.append(note_lines(x + 16, GY + 212, [
        'Ih and Ih→Dh are not separable from a single frame — a 4-class model',
        'reaches only 89.3 %. They are merged here, and the transition is',
        'recovered from temporal context (Fig. 1, Temporal Smoothing Head).'],
        fill=RED))

    # connectors
    s.append(poly_arrow([(cx + 320, RY + 56), (cx + 320, 404),
                         (xs[0] + pw3 / 2, 404), (xs[0] + pw3 / 2, GY - 4)],
                        GREEN_S, 1.6, dash='5 4', marker='ah_green'))
    s.append(poly_arrow([(cx + 500, RY + 56), (cx + 500, 398),
                         (xs[1] + pw3 / 2, 398), (xs[1] + pw3 / 2, GY - 4)],
                        GREEN_S, 1.6, dash='5 4', marker='ah_green'))
    s.append(poly_arrow([(400, RY + 84), (400, 392), (xs[2] + pw3 / 2, 392),
                         (xs[2] + pw3 / 2, GY - 4)], GREEN_S, 1.6, dash='5 4',
                        marker='ah_green'))
    s.append(poly_arrow([(bx + 100, RY + 46), (bx + 100, 386),
                         (xs[3] + pw3 / 2, 386), (xs[3] + pw3 / 2, GY - 4)],
                        GREEN_S, 1.6, dash='5 4', marker='ah_green'))

    # ---- results + training strip ----------------------------------------
    FY, FH = 712, 250
    s.append(rect(50, FY, 834, FH, PAPER, RULE, rx=12))
    s.append(text(70, FY + 28, 'Validation accuracy', size=13.5, fill=INK,
                  weight=700))
    s.append(text(70, FY + 45, 'real particle-level split — 1 214 of 12 116 images, '
                               'split 90/10 by particle', size=9.5, fill=INK3))
    s.append(line(70, FY + 56, 864, FY + 56, RULE, 1))
    rows = [('3-class merged + 8-fold TTA', 'Swin-Tiny', 'real', 96.62, True),
            ('3-class merged, no TTA', 'Swin-Tiny', 'real', 95.88, False),
            ('3-class, Ih→Dh dropped', 'Swin-Tiny', 'real', 96.58, False),
            ('3-class merged', 'Swin-Base', 'real', 95.55, False),
            ('4-class (Ih→Dh separate)', 'Swin-Base', 'real', 89.29, False),
            ('real + synthetic (val on real)', 'Swin-Base', 'combined', 94.81,
             False)]
    yy = FY + 78
    for nm, arch, data, acc, best in rows:
        if best:
            s.append(rect(62, yy - 13, 810, 24, PURPLE_L, 'none', rx=5))
        c = INK if best else INK2
        wgt = 700 if best else 400
        s.append(text(70, yy + 4, nm, size=10.5, fill=c, weight=wgt))
        s.append(text(316, yy + 4, arch, size=10.5, fill=c, weight=wgt))
        s.append(text(414, yy + 4, data, size=10.5, fill=c, weight=wgt))
        s.append(rect(500, yy - 6, 250, 11, SOFT, 'none', rx=3))
        s.append(rect(500, yy - 6, 250 * (acc - 85) / 15, 11,
                      PURPLE if best else '#a89bc4', 'none', rx=3))
        s.append(text(768, yy + 4, f'{acc:.2f} %', size=10.5, fill=c, weight=wgt,
                      family=MONO))
        yy += 26
    s.append(text(70, FY + 238, 'Ensembling was tested and rejected — the weaker '
                                'members drag the mean below the single best model '
                                'with TTA.', size=9.5, fill=INK2))

    s.append(rect(910, FY, 940, FH, SOFT, RULE, rx=12))
    s.append(text(930, FY + 28, 'Training configuration', size=13.5, fill=INK,
                  weight=700))
    s.append(line(930, FY + 42, 1830, FY + 42, RULE, 1))
    cfg = [('Loss', 'cross-entropy, label smoothing 0.05, inverse-frequency class '
                    'weights'),
           ('Sampler', 'WeightedRandomSampler power 1.0 — classes fully equalised '
                       'per epoch'),
           ('Optimiser', 'AdamW, lr 1e-4 (warm-start) / 3e-4, weight decay 1e-4, '
                         'cosine annealing'),
           ('Schedule', 'batch 32, 20–25 epochs, seed 44, best-val checkpoint kept'),
           ('Augmentation', 'RandomResizedCrop(224, 0.6–1.0), H/V flip, exact '
                            '0/90/180/270° rotation,'),
           ('', 'jitter 0.2 — exact rotations only, because interpolation destroys '
                'the'),
           ('', 'lattice fringes the model classifies on'),
           ('Warm start', 'backbone from a prior 3-class checkpoint; without it '
                          'training collapses'),
           ('', 'into the majority-class minimum'),
           ('Splits', 'by PARTICLE, not by image — both frames of a particle stay '
                      'on one side')]
    yy = FY + 66
    for k, v in cfg:
        if k:
            s.append(text(930, yy, k, size=10, fill=PURPLE, weight=700))
        s.append(text(1032, yy, v, size=9.8, fill=INK2))
        yy += 19

    return svg_doc(W, H, '\n'.join(s),
                   'Figure 2 — Swin-Tiny phase classifier with dihedral TTA')


if __name__ == '__main__':
    for name, fn in [('fig1_pipeline', figure1), ('fig2_classifier', figure2)]:
        path = os.path.join(OUT_DIR, name + '.svg')
        with open(path, 'w') as f:
            f.write(fn())
        print('wrote', path, f'({os.path.getsize(path)/1024:.0f} KB)')
