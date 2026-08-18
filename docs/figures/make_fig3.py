#!/usr/bin/env python3
"""
Figure 3 — compact classifier architecture panel.

A single, self-contained network-structure panel in the classic CNN-diagram
style (tilted layer slabs + legend + flatten/FC/classification bars), for use
where Figure 2 is too broad. Shows only the Stage-4 model: Swin-Tiny from the
224x224x3 input through the four stages to the 3-class output.

    python make_fig3.py
    google-chrome --headless --disable-gpu --screenshot=fig3_classifier_arch.png \
        --window-size=1460,660 --default-background-color=ffffff \
        fig3_classifier_arch.svg

Numbers are from classifier/train.py and the timm swin_tiny_patch4_window7_224
configuration; see docs/PIPELINE_METHODS.md section 4.
"""
import os

from make_figures import (INK, INK2, INK3, RULE, MONO, FONT, text, rect, line,
                          img, svg_doc, esc)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

K = 0.80                      # shear: how far a slab's top edge rises per px of thickness
CY = 340                      # vertical centre line of the network
LABY = 510                    # common baseline for the stage captions

# slab palette — one hue per stage, light/dark alternating for W-MSA / SW-MSA
ST = [
    dict(dark='#b0524d', light='#d99992', name='Stage 1'),
    dict(dark='#3d87a0', light='#8cc0d0', name='Stage 2'),
    dict(dark='#7f9145', light='#bcc887', name='Stage 3'),
    dict(dark='#71629a', light='#b0a6cb', name='Stage 4'),
]
MERGE = '#f0d34a'             # patch-merging slab
NORM  = '#dde3e8'             # LayerNorm slab
EMBED = '#4a5560'             # patch-embedding slab
BARC  = '#b9aede'             # flatten / FC / classification cells
ORANGE = '#e8792b'


def poly(pts, fill, stroke=INK, sw=1.1, op=None):
    p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    o = f' fill-opacity="{op}"' if op is not None else ''
    return (f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round"{o}/>')


def quad(x, w, h, cy=CY, dy=0.0, hh=None):
    """The sheared parallelogram for a slab of thickness w and face height h."""
    hh = h if hh is None else hh
    y0 = cy - h / 2 + dy
    return [(x, y0), (x + w, y0 - K * w), (x + w, y0 - K * w + hh), (x, y0 + hh)]


def slab(x, w, h, fill, stroke=INK, sw=1.1, cy=CY):
    return poly(quad(x, w, h, cy), fill, stroke, sw)


def face_pt(x, h, dx, frac, cy=CY):
    """A point on a slab's face: dx along the thickness, frac down the height."""
    return (x + dx, cy - h / 2 - K * dx + frac * h)


def cube(x, y, s, fill, top, side):
    """Small 3-D cube for the legend."""
    d = s * 0.42
    return (poly([(x, y), (x + d, y - d), (x + s + d, y - d), (x + s, y)], top, INK, 0.9) +
            poly([(x + s, y), (x + s + d, y - d), (x + s + d, y + s - d), (x + s, y + s)],
                 side, INK, 0.9) +
            rect(x, y, s, s, fill, INK, rx=0, sw=0.9))


def block_arrow(x, cy, w=44, h=26):
    """Orange block arrow, as in the reference panel."""
    t = h * 0.42
    return poly([(x, cy - t), (x + w * 0.58, cy - t), (x + w * 0.58, cy - h / 2),
                 (x + w, cy), (x + w * 0.58, cy + h / 2), (x + w * 0.58, cy + t),
                 (x, cy + t)], ORANGE, '#b45a1a', 1)


def cell_bar(x, cy, n, cw=26, ch=19, fill=BARC):
    """A vertical stack of n cells (flatten / fully-connected / classification)."""
    o = []
    y0 = cy - n * ch / 2
    for i in range(n):
        o.append(rect(x, y0 + i * ch, cw, ch, fill, INK, rx=0, sw=0.9))
    return ''.join(o), y0, y0 + n * ch


def figure3():
    W, H = 1460, 620
    s = []

    s.append(text(40, 40, 'Figure 3  |  Swin-Tiny phase classifier — network '
                          'architecture', size=17, fill=INK, weight=700))
    s.append(text(40, 61, 'one 256 × 256 segmented particle crop in, three '
                          'crystallographic phase scores out', size=11.5,
                  fill=INK3))

    # ------------------------------------------------------------- legend ---
    lx, ly = 44, 132
    for i, (label, sub) in enumerate([
            ('Swin Transformer block', 'W-MSA / SW-MSA + MLP, residual'),
            ('Patch merging layer', 'tokens ↓2 per side, channels ×2'),
            ('LayerNorm', 'pre-norm on every sub-block')]):
        y = ly + i * 62
        if i == 0:
            for j, c in enumerate(ST[:3]):
                s.append(cube(lx + j * 26, y, 22, c['dark'], c['light'], c['dark']))
        elif i == 1:
            s.append(cube(lx, y, 22, MERGE, '#f8e78f', '#d0b52f'))
        else:
            s.append(cube(lx, y, 22, NORM, '#eef2f5', '#c2cad1'))
        s.append(text(lx + 96, y + 12, label, size=11.5, fill=INK, weight=700))
        s.append(text(lx + 96, y + 27, sub, size=9.5, fill=INK3))

    # -------------------------------------------------------------- input ---
    ix, iw = 316, 152
    s.append(img(ix, CY - iw / 2, iw, iw, '04_segcrop.png', INK, 1.5))
    s.append(text(ix + iw / 2, CY - iw / 2 - 22, 'Input crop', size=11.5,
                  fill=INK, weight=700, anchor='middle'))
    s.append(text(ix + iw / 2, CY - iw / 2 - 8, '224 × 224 × 3', size=10,
                  fill=INK3, anchor='middle', family=MONO))
    # the 7x7 attention window, marked as in the reference
    wx, wy, ws = ix + 46, CY - 22, 40
    s.append(rect(wx, wy, ws, ws, 'none', '#e03a2f', rx=0, sw=2))
    s.append(text(ix + iw / 2, CY + iw / 2 + 18, '7 × 7 attention window',
                  size=9.5, fill='#e03a2f', anchor='middle'))

    # ------------------------------------------------------------- stages ---
    # (thickness encodes channel width, face height encodes token-grid size)
    x = 520
    s.append(slab(x, 18, 262, EMBED))
    s.append(text(x + 9, CY - 262 / 2 - 26, 'Patch Embed', size=10, fill=INK,
                  weight=700, anchor='middle'))
    s.append(text(x + 9, CY - 262 / 2 - 13, '4×4 conv, stride 4', size=8.5,
                  fill=INK3, anchor='middle', family=MONO))
    x += 18

    specs = [  # (blocks, thickness, face height, tokens, channels)
        (2, 34, 250, '56 × 56', '96'),
        (2, 48, 192, '28 × 28', '192'),
        (6, 74, 138, '14 × 14', '384'),
        (2, 92, 96,  '7 × 7',   '768'),
    ]
    marks = []                      # face anchors for the projection lines
    for i, (nb, tw, h, tok, ch) in enumerate(specs):
        x += 16
        s.append(slab(x, 7, h, NORM))          # LayerNorm
        x += 9
        bw = tw / nb
        for b in range(nb):
            s.append(slab(x + b * bw, bw, h,
                          ST[i]['dark'] if b % 2 == 0 else ST[i]['light'],
                          sw=0.9, cy=CY - K * b * bw))
        s.append(poly(quad(x, tw, h), 'none', INK, 1.6))   # group outline
        marks.append((x, tw, h))
        # labels on one common baseline, with a leader up to the slab
        cxm = x + tw / 2
        ybot = CY + h / 2 - K * tw / 2
        s.append(line(cxm, ybot + 4, cxm, LABY - 14, RULE, 1))
        s.append(text(cxm, LABY, ST[i]['name'], size=11, fill=INK,
                      weight=700, anchor='middle'))
        s.append(text(cxm, LABY + 15, f'{tok} × {ch}', size=9.5,
                      fill=INK2, anchor='middle', family=MONO))
        s.append(text(cxm, LABY + 28,
                      f'{nb} blocks · {[3, 6, 12, 24][i]} heads', size=9,
                      fill=INK3, anchor='middle'))
        x += tw
        if i < 3:
            x += 6
            s.append(slab(x, 11, specs[i + 1][2], MERGE))   # patch merging
            x += 11

    # dashed projection lines: input window → stage 1 → … → stage 4
    src = [(wx, wy), (wx + ws, wy), (wx, wy + ws), (wx + ws, wy + ws)]
    for i, (sx, tw, h) in enumerate(marks):
        f = 0.17 - i * 0.012
        dst = [face_pt(sx, h, 0, 0.5 - f), face_pt(sx, h, 0, 0.5 - f),
               face_pt(sx, h, 0, 0.5 + f), face_pt(sx, h, 0, 0.5 + f)]
        w2 = h * f * 0.9
        box = [(dst[0][0], dst[0][1]), (dst[0][0] + w2, dst[0][1] - K * w2),
               (dst[2][0] + w2, dst[2][1] - K * w2), (dst[2][0], dst[2][1])]
        s.append(poly(box, '#ffffff', '#e03a2f', 1.4, op=0.0))
        for a, b in zip(src, [box[0], box[1], box[3], box[2]]):
            s.append(line(a[0], a[1], b[0], b[1], INK3, 0.7, dash='3 3'))
        src = [box[0], box[1], box[3], box[2]]

    # --------------------------------------------- head: GAP → FC → classes --
    hx = x + 30
    s.append(block_arrow(hx, CY))
    b1, t1, bot1 = cell_bar(hx + 62, CY, 11)
    s.append(b1)
    s.append(text(hx + 75, bot1 + 22, 'Global pool', size=11, fill='#5c4f8a',
                  weight=700, anchor='middle'))
    s.append(text(hx + 75, bot1 + 36, '768-d', size=9.5, fill=INK3,
                  anchor='middle', family=MONO))

    s.append(block_arrow(hx + 100, CY))
    b2, t2, bot2 = cell_bar(hx + 162, CY, 3, cw=30, ch=30)
    s.append(b2)
    s.append(text(hx + 177, bot2 + 22, 'Fully connected', size=11,
                  fill='#5c4f8a', weight=700, anchor='middle'))
    s.append(text(hx + 177, bot2 + 36, '768 → 3 logits', size=9.5, fill=INK3,
                  anchor='middle', family=MONO))

    s.append(block_arrow(hx + 200, CY))
    cls = [('Dh', '#2f6f9f'), ('FCC', '#c2762a'), ('Ih', '#1d6f65')]
    y0 = CY - 3 * 30 / 2
    for i, (nm, col) in enumerate(cls):
        s.append(rect(hx + 262, y0 + i * 30, 30, 30, col, INK, rx=0, sw=0.9,
                      op=0.85 if i == 2 else 0.30))
        s.append(text(hx + 305, y0 + i * 30 + 20, nm, size=10.5, fill=INK,
                      weight=700 if i == 2 else 400))
    s.append(text(hx + 277, y0 + 3 * 30 + 22, 'Classification', size=11,
                  fill='#5c4f8a', weight=700, anchor='middle'))
    s.append(text(hx + 277, y0 + 3 * 30 + 36, 'softmax, 3 classes', size=9.5,
                  fill=INK3, anchor='middle'))

    # ------------------------------------------------------------- caption --
    s.append(line(40, H - 54, W - 40, H - 54, RULE, 1))
    s.append(text(40, H - 34,
                  'swin_tiny_patch4_window7_224 · ImageNet-1k pretrained · '
                  '27.5 M parameters · slab thickness ∝ channel width, face '
                  'height ∝ token-grid size',
                  size=10, fill=INK2))
    s.append(text(40, H - 18,
                  'Attention is confined to 7 × 7 windows; the window grid '
                  'shifts by 3 between consecutive blocks, which is what lets '
                  'information cross window boundaries.',
                  size=10, fill=INK3))
    return svg_doc(W, H, '\n'.join(s),
                   'Figure 3 — Swin-Tiny classifier architecture')


if __name__ == '__main__':
    p = os.path.join(OUT_DIR, 'fig3_classifier_arch.svg')
    with open(p, 'w') as f:
        f.write(figure3())
    print('wrote', p, f'({os.path.getsize(p)/1024:.0f} KB)')
