"""Build /model.json: body-local, decimated visual meshes with real-robot colours.

version 1 fields (always present, old clients keep working): per mesh "rgba", "verts", "tris".
Optional extras (v1-compatible, ignored by old clients):
  "name"    : MJCF mesh name
  "normals" : per-vertex unit normals [nx,ny,nz,...] in the body-local frame, smoothed within CREASE_DEG
              (vertices are split along creases, so hard CAD edges stay sharp)
  "colors"  : per-vertex sRGB bytes [r,g,b,...] 0-255 = palette colour x baked ambient occlusion
  "cull"    : true when the decimated mesh is closed and consistently outward-wound (back-face culling safe)
"""

import json
import os
import zlib

import mujoco
import numpy as np

EXPORT_VERSION = 2
TRI_BUDGET = 25000        # informational: the per-part targets below sum to ~this
CREASE_DEG = 35.0
AO_RAYS = 24
AO_RANGE = 0.012          # metres: occluders further away than this don't darken
AO_STRENGTH = 0.50        # 0 = no AO, 1 = fully occluded vertices go black
MIN_TRIS = 12

# Real Microduck look (Pollen Robotics photos): white shells, light-grey face plate, orange-yellow eye ring
# around a black camera lens, orange beak / lower head shell and feet with yellow-orange soles, black XL330
# servos, grey brackets. Per MJCF mesh name: (sRGB colour or None = MJCF material, triangle target per
# instance; 0 = drop). Parts fully enclosed on the real robot (PCBs, speaker, inner bearings, soft mouth
# insert) are dropped: in the standing pose < 3 % of their surface can see out (preview/exposure.py).
WHITE = (0.925, 0.918, 0.898)
ORANGE = (0.965, 0.380, 0.110)
AMBER = (0.975, 0.650, 0.120)
SERVO = (0.105, 0.105, 0.112)
DARK = (0.150, 0.158, 0.160)
BRACKET = (0.640, 0.650, 0.665)
METAL = (0.720, 0.740, 0.770)
PARTS = {
    "top_head_shell": (WHITE, 3200),
    "bottom_head_shell": (ORANGE, 2400),
    "face_part": ((0.860, 0.860, 0.850), 1400),
    "noenoeil": (AMBER, 984),
    "lens": ((0.045, 0.048, 0.055), 400),
    "jaw": (ORANGE, 1200),
    "left_shell": (WHITE, 2200),
    "right_shell": (WHITE, 2200),
    "upper_leg_left": (WHITE, 800),
    "upper_leg_right": (WHITE, 800),
    "ankle_left": (ORANGE, 700),
    "ankle_right": (ORANGE, 700),
    "sole_left": (AMBER, 600),
    "sole_right": (AMBER, 600),
    "foot_left": (ORANGE, 120),
    "foot_right": (ORANGE, 120),
    "leg": (BRACKET, 500),
    "hip_l": (BRACKET, 500),
    "neck": (BRACKET, 250),
    "upper_leg_rigidity_plate": (BRACKET, 200),
    "power_support": (BRACKET, 250),
    "neck_pitch": ((0.30, 0.32, 0.34), 500),
    "yaw2roll": (DARK, 350),
    "bearing_roll": (DARK, 150),
    "trunk_base": (DARK, 350),
    "motor_support": (DARK, 200),
    "yaw_roll_motion": (DARK, 100),
    "xl330": (SERVO, 90),
    "np_f970": ((0.20, 0.20, 0.21), 120),
    "banana_pcb_locker": (BRACKET, 50),
    "seeed_bearing__configuration__22x16x4": (METAL, 80),
    "seeed_bearing__configuration_default": (METAL, 80),
    "elec_rpi_robot_hat_pcb": (None, 0),
    "pcb__raspberry_pi_zero_2_w": (None, 0),
    "m12_lens_holder": (None, 0),
    "speaker": (None, 0),
    "soft_mouth_top": (None, 0),
    "jaw_soft": (None, 0),
}
DEFAULT_TARGET = 150
# clustering cell sizes tried before quadric simplification (see _decimate); default DEFAULT_CELLS
CELLS = {"face_part": (4e-4, 3e-4)}
# bearings that sit fully inside other parts (exposure < ~10 % in the standing pose) are dropped on these bodies
DROP_BEARING_BODIES = {"trunk_base", "hip_l", "hip_l_2", "upper_leg_left", "upper_leg_right",
                       "neck_pitch", "yaw_roll_motion", "jaw_soft"}


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def _area(v, f):
    a = v[f[:, 1]] - v[f[:, 0]]
    b = v[f[:, 2]] - v[f[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def _weld(v, f, tol=1e-7):
    key = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    f2 = inv.reshape(-1)[f]
    good = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    return v[idx], f2[good]


def _compact(v, f):
    used = np.unique(f)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def _cluster(v, f, cell):
    """Vertex-clustering decimation: snap to a grid of `cell` metres, average each cell."""
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1
    nv = np.zeros((n, 3))
    np.add.at(nv, inv, v)
    cnt = np.bincount(inv, minlength=n)[:, None]
    nv /= cnt
    f2 = inv[f]
    good = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = f2[good]
    if len(f2):
        s = np.sort(f2, axis=1)
        _, keep = np.unique(s, axis=0, return_index=True)
        f2 = f2[np.sort(keep)]
    return _compact(nv, f2)


def _cluster_search(v, f, target):
    ext = float(np.max(v.max(0) - v.min(0))) or 1e-3
    lo, hi = ext / 2000, ext / 2
    best = None
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        cv, cf = _cluster(v, f, mid)
        if len(cf) > target:
            lo = mid
        else:
            hi = mid
            best = (cv, cf)
    if best is None or len(best[1]) == 0:
        best = _cluster(v, f, lo)
    return best


def _qem(v, f, target):
    import fast_simplification
    for _ in range(3):
        if len(f) <= target * 1.05:
            break
        red = 1.0 - target / len(f)
        v2, f2 = fast_simplification.simplify((v * 1000).astype(np.float32), f.astype(np.int64),
                                              target_reduction=max(0.0, red), agg=7)
        if len(f2) == 0 or len(f2) >= len(f) * 0.98:
            break
        v, f = v2.astype(np.float64) / 1000, np.asarray(f2, dtype=np.int64)
    return v, f


DEFAULT_CELLS = (2e-4, 1e-4, 4e-4, 0.0)


def _decimate(v, f, target, cells=DEFAULT_CELLS):
    """Quadric simplification (in mm: float32-safe). On these CAD meshes it stalls (sliver fans) at ~4k tris and,
    run alone on the big curved shells, leaves fold-over gashes; regularising first with fine vertex clustering
    (0.2 mm by default; per-part override in PARTS, chosen by eye with preview/variants.py) fixes both. Takes
    the first cell size whose result is within 15 % of the target; coarse clustering only as a last resort."""
    v, f = _weld(v, f)
    if len(f) <= target:
        return _compact(v, f)
    for cell in cells:
        cv, cf = (v, f) if cell == 0 else _cluster(v, f, cell)
        if len(cf) == 0:
            continue
        cv, cf = _qem(cv, cf, target)
        if len(cf) <= target * 1.15:
            return _compact(cv, cf)
    return _compact(*_cluster_search(v, f, target))


def _closed(f):
    """Closed 2-manifold with consistent winding (every directed edge once, every edge twice)."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, dcnt = np.unique(e, axis=0, return_counts=True)
    _, ucnt = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    return bool((dcnt == 1).all() and (ucnt == 2).all())


def _smooth(v, f, crease_deg=CREASE_DEG):
    """Split vertices along creases; return (verts, normals, tris) with angle-weighted smooth normals."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1), 1e-20)[:, None]

    def ang(a, b, c):
        u, w = b - a, c - a
        cu = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-20)
        return np.arccos(np.clip(cu, -1, 1))
    cang = np.stack([ang(p0, p1, p2), ang(p1, p2, p0), ang(p2, p0, p1)], 1).reshape(-1)  # corner angles
    cosc = np.cos(np.radians(crease_deg))
    nt = len(f)
    cv = f.reshape(-1)                         # corner -> vertex
    ct = np.repeat(np.arange(nt), 3)           # corner -> face
    order = np.argsort(cv, kind="stable")
    sv = cv[order]
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    ends = np.r_[starts[1:], len(sv)]
    cn = np.zeros((nt * 3, 3))
    for s, e in zip(starts, ends):
        cs = order[s:e]                        # corners sharing this vertex
        nrm = fn[ct[cs]]
        blend = nrm @ nrm.T >= cosc            # which incident faces are smoothed together
        cn[cs] = (blend * cang[cs][None, :]) @ nrm
    cn /= np.maximum(np.linalg.norm(cn, axis=1), 1e-20)[:, None]
    # weld corners with the same vertex and (almost) the same normal
    key = np.concatenate([cv[:, None], np.round(cn * 2000).astype(np.int64)], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    nv = v[cv[first]]
    nn = np.zeros((len(first), 3))
    np.add.at(nn, inv, cn)
    nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-20)[:, None]
    return nv, nn, inv.reshape(-1, 3)


def _fib_dirs(n):
    """Cosine-weighted hemisphere directions around +z (Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    phi = i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - r * r)], 1)


def _ao(model, data, body, pw, nw):
    """Per-vertex ambient occlusion from same-body geometry only (pose-invariant, so it stays valid while the
    duck moves). pw/nw: world-space points/normals in the qpos0 pose. Returns visibility 0..1."""
    dirs = _fib_dirs(AO_RAYS)
    gg = np.array([0, 0, 1, 0, 0, 0], np.uint8)
    gid = np.zeros(AO_RAYS, np.int32)
    dist = np.zeros(AO_RAYS)
    out = np.ones(len(pw))
    for i in range(len(pw)):
        n = nw[i]
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        t = np.cross(n, a)
        t /= np.linalg.norm(t)
        b = np.cross(n, t)
        d = dirs @ np.stack([t, b, n])
        mujoco.mj_multiRay(model, data, pw[i] + n * 3e-4, d.reshape(-1).copy(), gg, 1, -1, gid, dist, None,
                           AO_RAYS, AO_RANGE)
        hit = (gid >= 0) & (dist >= 0) & (dist < AO_RANGE)
        hit &= model.geom_bodyid[np.maximum(gid, 0)] == body
        out[i] = 1 - np.where(hit, 1 - dist / AO_RANGE, 0.0).mean()
    return out


def body_name(model, b):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    return n if n else f"body_{b}"


def _disc(centre, nrm, ax_u, ax_v, ru, rv, rgba, name, seg=20, square=False):
    """Flat elliptical (or rounded-rectangle) decal facing `nrm` (CCW seen from the front)."""
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    cu, cv = np.cos(ang), np.sin(ang)
    if square:                                  # superellipse |x|^4 + |y|^4 = 1
        cu, cv = np.sign(cu) * np.abs(cu) ** 0.5, np.sign(cv) * np.abs(cv) ** 0.5
    ring = centre + cu[:, None] * ax_u * ru + cv[:, None] * ax_v * rv
    v = np.vstack([centre, ring])
    f = np.array([[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)])
    if np.dot(np.cross(v[f[0, 1]] - v[0], v[f[0, 2]] - v[0]), nrm) < 0:
        f = f[:, [0, 2, 1]]
    col = [int(round(c * 255)) for c in rgba[:3]] * len(v)
    return {"name": name, "rgba": [round(float(c), 4) for c in rgba],
            "verts": [round(float(x), 5) for x in v.reshape(-1)], "tris": [int(i) for i in f.reshape(-1)],
            "normals": [round(float(x), 3) for x in np.tile(nrm, len(v))], "colors": col, "cull": True}


def _add_face_details(model, data, bodies):
    """The real Microduck's face: camera eye in an orange ring (meshes), plus a small black depth-sensor window
    beside it (robot's left) and a catch-light on the lens. The CAD has neither, so add two tiny decals."""
    for b, meshes in bodies.items():
        by = {m["name"]: m for m in meshes}
        if not {"noenoeil", "face_part", "lens"} <= set(by):
            continue
        Rb = data.xmat[b].reshape(3, 3)
        fwd_g, left_g = Rb.T @ np.array([1.0, 0, 0]), Rb.T @ np.array([0, 1.0, 0])   # world +x / +y, body frame
        rv = np.array(by["noenoeil"]["verts"]).reshape(-1, 3)
        rn = np.array(by["noenoeil"]["normals"]).reshape(-1, 3)
        front = rn @ fwd_g > 0.7
        if front.sum() < 3:
            continue
        nf = rn[front].mean(0)
        nf /= np.linalg.norm(nf)
        left = left_g - nf * (left_g @ nf)
        left /= np.linalg.norm(left)
        up = np.cross(nf, left)
        c = rv.mean(0)
        dia = float((rv @ left).max() - (rv @ left).min())
        # depth sensor window: the face plate has a cut-out for it beside the eye (robot's left); find the
        # cut-out's side walls just behind the plate front and cover it with a black rounded rectangle
        pv = np.array(by["face_part"]["verts"]).reshape(-1, 3)
        pn = np.array(by["face_part"]["normals"]).reshape(-1, 3)
        fr = pn @ nf > 0.95
        if not fr.any():
            continue
        front_d = float((pv[fr] @ nf).max())
        d = pv - c
        du, dl = d @ up, d @ left
        wall = ((np.abs(pn @ nf) < 0.3) & (pv @ nf > front_d - 6e-4) & (np.hypot(du, dl) > 0.58 * dia)
                & (np.hypot(du, dl) < 1.0 * dia) & (dl > 0) & (np.abs(du) < 0.35 * dia))
        if wall.sum() >= 4:
            cl, cu = (dl[wall].min() + dl[wall].max()) / 2, (du[wall].min() + du[wall].max()) / 2
            hl = (dl[wall].max() - dl[wall].min()) / 2 + 6e-4
            hu = (du[wall].max() - du[wall].min()) / 2 + 6e-4
        else:                                       # no cut-out in this CAD revision: use the photo layout
            cl, cu, hl, hu = 0.82 * dia, 0.0, 0.13 * dia, 0.08 * dia
        s = c + left * cl + up * cu
        meshes.append(_disc(s + nf * (front_d - s @ nf + 3e-4), nf, left, up, hl, hu,
                            (0.035, 0.035, 0.045, 1.0), "tof_window", seg=24, square=True))
        # catch-light on the lens (upper, toward the robot's right)
        g = c + up * 0.17 * dia - left * 0.10 * dia
        lv = np.array(by["lens"]["verts"]).reshape(-1, 3)
        lat = lv - g
        lat -= np.outer(lat @ nf, nf)
        near = np.linalg.norm(lat, axis=1) < 0.15 * dia
        if near.any():
            depth = max(float((lv[near] @ nf).max()), float((lv @ nf).max()) - 2e-3)
            meshes.append(_disc(g + nf * (depth - g @ nf + 3e-4), nf, left, up, 0.055 * dia, 0.055 * dia,
                                (0.93, 0.94, 0.96, 1.0), "eye_glint", seg=12))


def build_model_json(model, budget=TRI_BUDGET, ao=True):
    """Return (dict, body_ids, ntris) for every body with at least one visual mesh geom.
    (`budget` is kept for API compatibility; triangle targets come from PARTS.)"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = {}
    cache = {}
    ntris = 0
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_bodyid[g] == 0:
            continue
        # visual = group 2 and non-colliding (class "visual")
        if model.geom_group[g] != 2 or (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        mid = int(model.geom_dataid[g])
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh_{mid}"
        b = int(model.geom_bodyid[g])
        colour, target = PARTS.get(mname, (None, DEFAULT_TARGET))
        if mname.startswith("seeed_bearing") and body_name(model, b) in DROP_BEARING_BODIES:
            target = 0
        if target <= 0:
            continue
        if model.geom_matid[g] >= 0:
            rgba = np.array(model.mat_rgba[model.geom_matid[g]], dtype=float)
        else:
            rgba = np.array(model.geom_rgba[g], dtype=float)
        if colour is not None:
            rgba[:3] = colour
        if mid not in cache:
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            v = model.mesh_vert[va:va + vn].astype(np.float64)
            f = model.mesh_face[fa:fa + fn].astype(np.int64)
            dv, df = _decimate(v, f, max(MIN_TRIS, target), CELLS.get(mname, DEFAULT_CELLS))
            closed = _closed(df)
            sv, sn, sf = _smooth(dv, df)
            cache[mid] = (sv, sn, sf, closed)
        sv, sn, sf, closed = cache[mid]
        R = _quat_to_mat(model.geom_quat[g])
        vb = sv @ R.T + model.geom_pos[g]           # body-local
        nb = sn @ R.T
        if ao:
            Rb = data.xmat[b].reshape(3, 3)
            vis = _ao(model, data, b, vb @ Rb.T + data.xpos[b], nb @ Rb.T)
        else:
            vis = np.ones(len(vb))
        shade = 1 - AO_STRENGTH * (1 - vis)
        cols = np.clip(np.round(rgba[:3][None, :] * shade[:, None] * 255), 0, 255).astype(np.int64)
        bodies.setdefault(b, []).append({
            "name": mname,
            "rgba": [round(float(c), 4) for c in rgba],
            "verts": [round(float(x), 5) for x in vb.reshape(-1)],
            "tris": [int(i) for i in sf.reshape(-1)],
            "normals": [round(float(x), 3) for x in nb.reshape(-1)],
            "colors": [int(c) for c in cols.reshape(-1)],
            "cull": closed,
        })
        ntris += len(sf)
    _add_face_details(model, data, bodies)
    ntris = sum(len(m["tris"]) // 3 for ms in bodies.values() for m in ms)
    body_ids = sorted(bodies)
    out = {"version": 1, "bodies": [{"name": body_name(model, b), "meshes": bodies[b]} for b in body_ids]}
    return out, body_ids, ntris


def _sig(model):
    parts = json.dumps([sorted((k, v[1], v[0]) for k, v in PARTS.items()), sorted(CELLS.items()),
                        DEFAULT_CELLS], separators=(",", ":"))
    return (f"v{EXPORT_VERSION}-{model.ngeom}-{model.nmesh}-{model.nbody}-{CREASE_DEG}-{AO_RAYS}-{AO_RANGE}"
            f"-{AO_STRENGTH}-{zlib.crc32(parts.encode()):08x}")


def load_or_build(model, cache_path):
    """Cache the JSON on disk (decimation + baked AO take ~half a minute)."""
    src_sig = _sig(model)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                blob = json.load(fh)
            if blob.get("_sig") == src_sig:
                ids = blob["_body_ids"]
                return blob["model"], ids, blob["_ntris"]
        except Exception:
            pass
    out, ids, ntris = build_model_json(model)
    with open(cache_path + ".tmp", "w") as fh:
        json.dump({"_sig": src_sig, "_body_ids": ids, "_ntris": ntris, "model": out}, fh, separators=(",", ":"))
    os.replace(cache_path + ".tmp", cache_path)
    return out, ids, ntris
