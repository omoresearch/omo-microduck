"""Render the README call-to-action buttons (light + dark variants).

Same shape and type as the omoresearch/omo-space buttons (IBM Plex Mono on an
ink pill), plus a dark-theme set so they read on GitHub's dark background.
    python3 docs/media/make_buttons.py path/to/fonts
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fonts"

CREAM = (245, 240, 232, 255)
INK = (26, 16, 8, 255)
SAND = (196, 184, 168, 255)
ORANGE = (226, 104, 36, 255)
H, R, PADX = 86, 16, 52

# name, glyph, label, primary?
BUTTONS = [
    ("star", "star", "Star on GitHub", True),
    ("download", "dl", "Download v1.0.0", True),
    ("reels", "play", "Watch the reels", True),
]


def glyph(d, kind, x, cy, col):
    if kind == "star":
        import math
        r1, r2, cx = 14, 6, x + 11
        pts = []
        for i in range(10):
            r = r1 if i % 2 == 0 else r2
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + 1 + r * math.sin(a)))
        d.polygon(pts, fill=col)
        return 22
    if kind == "play":
        d.polygon([(x, cy - 12), (x, cy + 12), (x + 21, cy)], fill=col)
        return 21
    if kind == "dl":
        d.rectangle((x + 8, cy - 14, x + 13, cy + 3), fill=col)
        d.polygon([(x, cy), (x + 21, cy), (x + 10.5, cy + 11)], fill=col)
        d.rectangle((x, cy + 14, x + 21, cy + 17), fill=col)
        return 21
    return 0


def render(name, kind, label, primary, theme):
    f = ImageFont.truetype(str(FONTS / "IBMPlexMono-Medium.ttf"), 31)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw = probe.textlength(label, font=f)
    gw = 21 + 26 if kind else 0
    w = int(PADX * 2 + gw + tw)
    img = Image.new("RGBA", (w, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if theme == "light":
        fill, fg, edge = (INK, CREAM, INK) if primary else (CREAM, INK, INK)
    else:
        fill, fg, edge = (CREAM, INK, CREAM) if primary else ((0, 0, 0, 0), CREAM, SAND)
    d.rounded_rectangle((1, 1, w - 2, H - 2), radius=R, fill=fill, outline=edge, width=3)
    x = PADX
    if kind:
        glyph(d, kind, x, H // 2, ORANGE if kind in ("play", "star") else fg)
        x += gw
    d.text((x, H // 2), label, font=f, fill=fg, anchor="lm")
    out = HERE / f"btn-{name}{'' if theme == 'light' else '-dark'}.png"
    img.save(out, optimize=True)
    print("wrote", out, img.size)


if __name__ == "__main__":
    for b in BUTTONS:
        for t in ("light", "dark"):
            render(*b, t)
