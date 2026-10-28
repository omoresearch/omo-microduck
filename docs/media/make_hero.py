"""Compose the README hero (light + dark) from a real in-game frame.

The picture is reel-lab-still.jpg: an unedited frame of the Microduck flock in
the Omo Space duck lab, taken from the stream recording behind the "secret lab"
reel, cropped clear of the face cam, chat, hotbar and held item, with the
crosshair removed. This script only frames it and sets the type.
Fonts: Crimson Text + IBM Plex Mono (SIL OFL). Run from the repo root:
    python3 docs/media/make_hero.py path/to/fonts [path/to/omo-mark.png]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
PAD = 48
THEMES = {
    "light": dict(bg=CREAM, fg=INK, sub=BROWN, pill_bg=INK, pill_fg=CREAM, frame=LINE),
    "dark": dict(bg=(18, 12, 7), fg=CREAM, sub=SAND, pill_bg=CREAM, pill_fg=INK, frame=BROWN),
}


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def smooth(t):
    return t * t * (3 - 2 * t)


def build(theme):
    c = THEMES[theme]
    img = Image.new("RGB", (W, H), c["bg"])

    # Real in-game frame as a tall photo plate on the right.
    src = Image.open(HERE / "reel-lab-still.jpg").convert("RGB")
    ph = H - 2 * PAD
    pw = round(src.width * ph / src.height)
    shot = src.resize((pw, ph), Image.LANCZOS)
    px, py = W - PAD - pw, PAD
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=28, fill=255)
    img.paste(shot, (px, py), mask)

    d = ImageDraw.Draw(img)
    d.rounded_rectangle((px - 1, py - 1, px + pw, py + ph), radius=29, outline=c["frame"], width=3)

    # Small honest caption on the photo itself.
    cap = "IN-GAME  ·  FROM THE STREAM"
    cf = font("IBMPlexMono-Medium.ttf", 22)
    cw = d.textlength(cap, font=cf)
    cx, cy = px + 28, py + 28
    d.rounded_rectangle((cx, cy, cx + cw + 40, cy + 44), radius=22, fill=INK)
    d.text((cx + 20, cy + 8), cap, font=cf, fill=CREAM)
    lx = 140

    eb = font("IBMPlexMono-Medium.ttf", 32)
    label = "OMO RESEARCH  ·  MICRODUCK"
    tw = d.textlength(label, font=eb)
    d.rounded_rectangle((lx - 4, 150, lx + tw + 44, 208), radius=29, fill=c["pill_bg"])
    d.text((lx + 20, 159), label, font=eb, fill=c["pill_fg"])

    serif = font("CrimsonText-SemiBold.ttf", 146)
    d.text((lx, 248), "A real robot duck,", font=serif, fill=c["fg"])
    d.text((lx, 410), "alive in Minecraft.", font=serif, fill=c["fg"])
    d.text((lx, 590), "Real CAD. Real servos. A real RL brain, simulated live.",
           font=font("CrimsonText-Italic.ttf", 56), fill=c["sub"])

    d.rectangle((lx, 694, lx + 120, 699), fill=ORANGE)
    d.text((lx, 726), "MuJoCo  ·  Pollen Robotics policies  ·  Fabric 26.2",
           font=font("IBMPlexMono-Regular.ttf", 33), fill=c["fg"])

    fy = 852
    tx = lx
    if MARK and MARK.exists():
        m = Image.open(MARK).convert("RGBA").resize((60, 60), Image.LANCZOS)
        if theme == "dark":  # recolour the mark to cream, keep its alpha
            a = m.split()[3]
            m = Image.merge("RGBA", (*Image.new("RGB", m.size, CREAM).split(), a))
        img.paste(m, (lx, fy - 12), m)
        tx = lx + 80
    d.text((tx, fy), "omo.sh", font=font("IBMPlexMono-Medium.ttf", 30), fill=c["sub"])

    d.rectangle((0, 0, W - 1, H - 1), outline=c["frame"], width=4)
    out = img.resize((1600, 640), Image.LANCZOS)
    name = "hero.png" if theme == "light" else "hero-dark.png"
    out.save(HERE / name, optimize=True)
    print("wrote", HERE / name, out.size)


if __name__ == "__main__":
    for t in THEMES:
        build(t)
