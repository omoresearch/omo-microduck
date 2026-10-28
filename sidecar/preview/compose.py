"""Compose compare.png: BEFORE row, AFTER row, and the real robot (reference photo) next to the AFTER head view.
Runs with the system python (needs Pillow, not numpy)."""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F, f2 = font(30), font(20)
before = Image.open(os.path.join(HERE, "before.png")).convert("RGB")
after = Image.open(os.path.join(HERE, "after.png")).convert("RGB")
iris = Image.open(os.path.join(HERE, "after_iris_head.png")).convert("RGB")
ref = Image.open(os.path.join(HERE, "ref", "scale-in-hand.png")).convert("RGB").crop((800, 160, 1300, 900))
W, H = before.size
ref = ref.resize((int(ref.width * H / ref.height), H))
head = Image.open(os.path.join(HERE, "after_head.png")).convert("RGB")

pad = 44
rows = [("BEFORE  (flat normals, MJCF colours, 13 871 tris, double-sided)", before),
        ("AFTER  (smooth normals, real colours + AO, eye details, 25 536 tris, shadow)", after)]
third_w = ref.width + head.width + iris.width + 12
out = Image.new("RGB", (max(W, third_w), 3 * (H + pad)), (24, 24, 28))
d = ImageDraw.Draw(out)
y = 0
for label, im in rows:
    d.text((10, y + 6), label, font=F, fill=(240, 240, 240))
    out.paste(im, (0, y + pad))
    y += H + pad
d.text((10, y + 6), "REAL MICRODUCK (Pollen Robotics photo)   |   AFTER, head   |   AFTER with an Iris shader pack (flat quad normals)",
       font=f2, fill=(240, 240, 240))
out.paste(ref, (0, y + pad))
out.paste(head, (ref.width + 6, y + pad))
out.paste(iris, (ref.width + head.width + 12, y + pad))
out.save(os.path.join(HERE, "compare.png"))
ref_full = Image.open(os.path.join(HERE, "ref", "scale-in-hand.png")).convert("RGB")
ref_full.save(os.path.join(HERE, "reference_microduck.png"))
print("compare.png", out.size)
