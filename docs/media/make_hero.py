"""Compose docs/media/hero.png from the real MuJoCo render (still-hero.png).

The duck is the offscreen render of the actual Pollen Microduck model; this
script only crops it, warms it into the omo.sh palette and sets the type.
Fonts: Crimson Text + IBM Plex Mono (SIL OFL). Run from the repo root:
    python3 docs/media/make_hero.py [path/to/fonts] [path/to/omo-mark.png]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageChops

HERE = Path(__file__).resolve().parent
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fonts"
MARK = Path(sys.argv[2]) if len(sys.argv) > 2 else None

CREAM = (245, 240, 232)
INK = (26, 16, 8)
BROWN = (74, 59, 47)
LINE = (212, 205, 192)
SAND = (196, 184, 168)
ORANGE = (226, 104, 36)

W, H = 2400, 960


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def main():
    img = Image.new("RGB", (W, H), CREAM)

    # Real render: crop around the duck and fit it to the right 56 % of the banner.
    src = Image.open(HERE / "still-hero.png").convert("RGB")
    crop = src.crop((250, 20, 1110, 720))  # 860 x 700
    rh = H
    rw = round(crop.width * rh / crop.height)
    duck = crop.resize((rw, rh), Image.LANCZOS)
    # Warm the cool studio sky/floor towards the cream paper (multiply-ish tint).
    warm = Image.new("RGB", duck.size, (255, 246, 232))
    duck = ImageChops.multiply(duck, warm)
    duck = Image.blend(duck, Image.new("RGB", duck.size, CREAM), 0.10)

    x0 = W - rw
    # Horizontal fade into the paper on the left edge of the render.
    mask = Image.new("L", duck.size, 255)
    md = ImageDraw.Draw(mask)
    fade = 420
    for x in range(fade):
        t = x / fade
        md.line([(x, 0), (x, rh)], fill=round(255 * (t * t * (3 - 2 * t))))
    img.paste(duck, (x0, 0), mask)

    d = ImageDraw.Draw(img)
    lx = 150

    # Eyebrow pill.
    eb = font("IBMPlexMono-Medium.ttf", 34)
    label = "OMO RESEARCH  ·  MICRODUCK"
    tw = d.textlength(label, font=eb)
    d.rounded_rectangle((lx - 4, 168, lx + tw + 44, 228), radius=30, fill=INK)
    d.text((lx + 20, 178), label, font=eb, fill=CREAM)

    serif = font("CrimsonText-SemiBold.ttf", 132)
    d.text((lx, 262), "A real robot duck,", font=serif, fill=INK)
    d.text((lx, 400), "alive in Minecraft.", font=serif, fill=INK)

    d.text((lx, 566), "Real CAD. Real servos. A real RL brain, simulated live.",
           font=font("CrimsonText-Italic.ttf", 54), fill=BROWN)

    d.rectangle((lx, 668, lx + 120, 673), fill=ORANGE)
    mono = font("IBMPlexMono-Regular.ttf", 34)
    d.text((lx, 702), "MuJoCo  ·  Pollen Robotics policies  ·  Fabric 26.2",
           font=mono, fill=INK)

    # Footer mark + url.
    fy = 840
    if MARK and MARK.exists():
        m = Image.open(MARK).convert("RGBA").resize((64, 64), Image.LANCZOS)
        img.paste(m, (lx, fy - 14), m)
        tx = lx + 84
    else:
        tx = lx
    d.text((tx, fy), "omo.sh", font=font("IBMPlexMono-Medium.ttf", 32), fill=BROWN)

    # Thin paper frame.
    d.rectangle((0, 0, W - 1, H - 1), outline=LINE, width=4)

    out = img.resize((W // 3 * 2, H // 3 * 2), Image.LANCZOS)  # 1600 x 640
    out.save(HERE / "hero.png", optimize=True)
    print("wrote", HERE / "hero.png", out.size)


if __name__ == "__main__":
    main()
