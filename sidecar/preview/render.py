"""Offline preview of the Microduck as the Fabric mod draws it (numpy software rasteriser).

Mirrors the mod: body-local verts -> sim world (pose) -> MC axes C(x,y,z)=(x,z,-y); vanilla entity lighting
(light.glsl: Light0=(0.2,1,-0.7), Light1=(-0.2,1,0.7) normalised, POWER 0.6, AMBIENT 0.4, Gouraud per vertex);
colour = vertex colour (or mesh rgba) * lighting * lightmap(=1 here).
  mode "before": flat per-triangle normals, double-sided entity_cutout (PER_FACE_LIGHTING: back faces get -light)
  mode "after":  per-vertex normals/colours from model.json when present, entity_cutout_cull (back faces culled)
  mode "iris":   like after, but flat face normals (Iris shader packs recompute per-quad face normals)

usage: render.py model.json pose.json out_prefix before|after|iris [blob_shadow_strength]
writes <out_prefix>_front.png, _side.png, _head.png, and <out_prefix>.png (the three views side by side)
"""
import json
import struct
import sys
import zlib

import numpy as np

L0 = np.array([0.2, 1.0, -0.7]); L0 /= np.linalg.norm(L0)
L1 = np.array([-0.2, 1.0, 0.7]); L1 /= np.linalg.norm(L1)
BG_TOP = np.array([0.62, 0.66, 0.72])
BG_BOT = np.array([0.36, 0.34, 0.33])


def write_png(path, img):
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def light(n):
    a = np.maximum(0, n @ L0) + np.maximum(0, n @ L1)
    return np.minimum(1.0, a * 0.6 + 0.4)


def build(model, pose, mode):
    """-> world (MC axes) triangle arrays: P (T,3,3), front-lit colour (T,3,3), back-lit colour (T,3,3) or None."""
    P, CF, CB = [], [], []
    for body in model["bodies"]:
        bp = pose["bodies"].get(body["name"])
        if bp is None:
            continue
        R = C @ quat_mat(bp[3:7])
        t = C @ np.array(bp[:3])
        for m in body["meshes"]:
            rgba = np.array(m.get("rgba", [0.8, 0.8, 0.8, 1.0]), dtype=float)
            if rgba[3] < 0.1:
                continue
            v = np.array(m["verts"], dtype=float).reshape(-1, 3) @ (C @ np.eye(3)).T  # body-local, MC axes (mod pre-bakes C)
            f = np.array(m["tris"], dtype=np.int64).reshape(-1, 3)
            vw = v @ (C @ quat_mat(bp[3:7]) @ C.T).T + t                           # R_mc = C R C^T
            tri = vw[f]
            if mode in ("after", "iris") and "colors" in m:
                col = np.array(m["colors"], dtype=float).reshape(-1, 3)[f] / 255.0
            else:
                col = np.broadcast_to(rgba[:3], tri.shape).copy()
            if mode == "after" and "normals" in m:
                nl = np.array(m["normals"], dtype=float).reshape(-1, 3) @ C.T
                nw = nl @ (C @ quat_mat(bp[3:7]) @ C.T).T
                nw = nw[f]
            else:
                e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
                fn = np.cross(e1, e2)
                fn /= np.maximum(1e-12, np.linalg.norm(fn, axis=1))[:, None]
                nw = np.repeat(fn[:, None, :], 3, axis=1)
            nw = nw / np.maximum(1e-12, np.linalg.norm(nw, axis=2))[..., None]
            P.append(tri)
            CF.append(col * light(nw)[..., None])
            if mode == "before" or not (m.get("cull", False) and "normals" in m):  # double-sided entity_cutout
                CB.append(col * light(-nw)[..., None])
            else:
                CB.append(np.full_like(col, np.nan))       # culled pipeline: back faces never drawn
    P = np.concatenate(P)
    CF = np.concatenate(CF)
    CB = np.concatenate(CB) if CB else None
    return P, CF, CB


def look_at(eye, target, up=(0, 1, 0)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f])


def rasterize(P, CF, CB, eye, target, fov_deg, W, H, ss=3, shadow=None):
    w, h = W * ss, H * ss
    Rv = look_at(np.asarray(eye, float), target)
    cam = (P - eye) @ Rv.T                                  # camera space, looking down -z
    fpx = 0.5 * h / np.tan(np.radians(fov_deg) / 2)
    z = -cam[..., 2]
    sx = w / 2 + fpx * cam[..., 0] / z
    sy = h / 2 - fpx * cam[..., 1] / z
    yy, xx = np.mgrid[0:h, 0:w]
    t = (yy / h)[..., None]
    img = (BG_TOP * (1 - t) + BG_BOT * t).astype(np.float64)
    if shadow is not None:
        img = shadow(img, xx, yy, Rv, eye, fpx, w, h)
    zbuf = np.full((h, w), np.inf)
    # screen-space signed area (> 0 = CCW on screen = front-facing, y flipped)
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    front = area < 0
    order = np.arange(len(P))
    for i in order:
        if (z[i] <= 0.01).any() or abs(area[i]) < 1e-9:
            continue
        cols = CF[i] if front[i] else CB[i]
        if np.isnan(cols[0, 0]):
            continue                                        # culled pipeline
        x0 = max(int(np.floor(sx[i].min())), 0); x1 = min(int(np.ceil(sx[i].max())), w - 1)
        y0 = max(int(np.floor(sy[i].min())), 0); y1 = min(int(np.ceil(sy[i].max())), h - 1)
        if x0 > x1 or y0 > y1:
            continue
        px = xx[y0:y1 + 1, x0:x1 + 1] + 0.5
        py = yy[y0:y1 + 1, x0:x1 + 1] + 0.5
        ax, bx, cx = sx[i]; ay, by, cy = sy[i]
        a = area[i]
        w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / a
        w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / a
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        iz = w0 / z[i, 0] + w1 / z[i, 1] + w2 / z[i, 2]       # perspective-correct depth / attributes
        depth = 1 / np.where(inside, iz, 1e-9)
        zb = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (depth < zb)
        if not win.any():
            continue
        zb[win] = depth[win]
        q0 = (w0 / z[i, 0] * depth)[win]; q1 = (w1 / z[i, 1] * depth)[win]; q2 = (w2 / z[i, 2] * depth)[win]
        sub = img[y0:y1 + 1, x0:x1 + 1]
        sub[win] = q0[:, None] * cols[0] + q1[:, None] * cols[1] + q2[:, None] * cols[2]
    img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
    return img


def floor_shadow_factory(floor_y, centre, radius, strength):
    """Floor plane + optional soft blob shadow (same falloff the mod uses)."""
    def shade(img, xx, yy, Rv, eye, fpx, w, h):
        d = np.stack([(xx + 0.5 - w / 2) / fpx, -(yy + 0.5 - h / 2) / fpx, -np.ones_like(xx, dtype=float)], -1) @ Rv
        tt = (floor_y - eye[1]) / np.where(np.abs(d[..., 1]) < 1e-9, -1e-9, d[..., 1])
        hit = tt > 0
        X = eye[0] + tt * d[..., 0]; Z = eye[2] + tt * d[..., 2]
        chk = ((np.floor(X / 0.25) + np.floor(Z / 0.25)) % 2 == 0)   # one MC block at scale 4
        base = np.where(chk[..., None], np.array([0.47, 0.46, 0.45]), np.array([0.43, 0.42, 0.41]))
        if strength > 0:
            r = np.hypot(X - centre[0], Z - centre[1]) / radius
            sh = 1 - strength * np.clip(1 - r * r, 0, 1) ** 2
            base = base * sh[..., None]
        return np.where(hit[..., None], base, img)
    return shade


def main():
    model = json.load(open(sys.argv[1]))
    pose = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    mode = sys.argv[4]
    shadow = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    P, CF, CB = build(model, pose, mode)
    print(f"{mode}: {len(P)} triangles")
    # duck centre (MC axes, metres; 1 sim metre = 1 unit here)
    tb = pose["bodies"]["trunk_base"]
    c = C @ np.array(tb[:3])
    head = C @ np.array(pose["bodies"]["jaw_soft"][:3]) if "jaw_soft" in pose["bodies"] else c + [0, 0.08, 0]
    target = np.array([c[0], 0.12, c[2]])
    # duck faces sim +x == MC +x
    views = {
        "front": (target + np.array([0.42, 0.13, 0.30]), target, 36),
        "side": (target + np.array([0.0, 0.03, -0.55]), target, 33),
        "head": (head + np.array([0.26, 0.06, 0.15]), head + np.array([0.03, -0.012, 0]), 30),
    }
    fl = floor_shadow_factory(0.0, (c[0], c[2]), 0.11, shadow)
    imgs = []
    for k, (eye, tgt, fov) in views.items():
        img = rasterize(P, CF, CB, eye, tgt, fov, 520, 560, ss=3, shadow=fl)
        write_png(f"{out}_{k}.png", img)
        imgs.append(img)
    sep = np.ones((560, 6, 3))
    write_png(f"{out}.png", np.concatenate([imgs[0], sep, imgs[1], sep, imgs[2]], axis=1))


if __name__ == "__main__":
    main()
