"""Minecraft terrain for the Microduck sim: static box geoms built from the blocks around the anchor.

The mod samples block collision shapes (VoxelShape AABBs) around the anchor, merges them greedily and POSTs
them here in SIM metres (see CONTRACT.md "Terrain"). This module

  * validates/hashes a terrain payload                         -> Terrain
  * adds the boxes to an MjSpec before compile                  -> apply(spec, terrain)
  * answers standable-height / safe-spawn queries in numpy      -> Terrain.find_spawn(...)
  * steers the duck away from walls with mj_ray (group 4 only)  -> avoid(sim, v)
  * rebuilds the MjModel off the sim lock and swaps it in, keeping the duck's state -> Manager

Terrain geoms live in geom group 4 (robot collision = 3, visual = 2, floor = 0) so rays can see only terrain.
"""

import hashlib
import math
import threading
import time

import numpy as np

import mujoco

MAX_BOXES = 4000
MAX_COORD = 64.0          # sim metres (|anchor-relative|): 256 blocks at scale 4
TERRAIN_GROUP = 4
RAMP_MAX_H = 0.035        # boxes this low (carpets, snow layers, repeaters) get ramps on their sides
RAMP_SLOPE = 0.1          # rise / run of those ramps
SPAWN_CLIMB = 0.3         # spawn search may go this far above the reference floor (2nd pass)
MAX_DROP = 0.6            # ground() looks at most this far below the reference
HEADROOM = 0.28           # free height above a spawn spot (duck ~0.25 m tall)
FOOT_R = 0.09             # spawn footprint half-size (the feet span ~0.12 m)
STEP_TOL = 0.05           # a surface up to this far above the reference still counts as "below the feet"
FLAT_TOL = 0.02           # footprint height spread allowed for a spawn spot
STAND_H = 0.116           # trunk height above the feet when standing
RAY_GROUP = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
# wall-avoid steering
AV_RANGE = 0.40           # ray length (m)
AV_SLOW = 0.26            # start slowing when the wall ahead is closer than this (from the trunk centre)
AV_STOP = 0.13            # no forward command closer than this
AV_SIDE = 0.14            # side rays: nudge away when closer than this
AV_LOW = 0.035            # low ray height above the feet: anything taller than ~a carpet is an obstacle
STUCK_S = 2.5
STUCK_D = 0.03


class TerrainError(ValueError):
    pass


def _num(v, what):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise TerrainError(f"{what} must be a number")
    if not math.isfinite(f):
        raise TerrainError(f"{what} must be finite")
    return f


class Terrain:
    """Immutable set of axis-aligned boxes in sim metres: boxes[i] = (x0, y0, z0, x1, y1, z1)."""

    def __init__(self, boxes, region=None, meta=None):
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        lo, hi = np.minimum(b[:, :3], b[:, 3:]), np.maximum(b[:, :3], b[:, 3:])
        keep = np.all(hi - lo > 1e-4, axis=1)
        self.boxes = np.hstack([lo, hi])[keep]
        self.region = None if region is None else np.asarray(region, dtype=np.float64).reshape(6)
        self.meta = dict(meta or {})
        q = np.round(self.boxes, 4).astype(np.float32)
        q = q[np.lexsort(q.T[::-1])] if len(q) else q
        self.hash = hashlib.sha1(q.tobytes()).hexdigest()[:12]
        if self.region is not None:
            self.floor_z = float(self.region[2])
        elif len(self.boxes):
            self.floor_z = float(self.boxes[:, 2].min())
        else:
            self.floor_z = 0.0

    def __len__(self):
        return len(self.boxes)

    @classmethod
    def from_json(cls, body):
        if not isinstance(body, dict):
            raise TerrainError("body must be a JSON object")
        raw = body.get("boxes")
        if not isinstance(raw, list):
            raise TerrainError("need \"boxes\": [[x0,y0,z0,x1,y1,z1], ...] (sim metres)")
        if raw and not isinstance(raw[0], list):   # also accept a flat list of 6*N numbers
            if len(raw) % 6:
                raise TerrainError("flat boxes list length must be a multiple of 6")
            raw = [raw[i:i + 6] for i in range(0, len(raw), 6)]
        if len(raw) > MAX_BOXES:
            raise TerrainError(f"too many boxes ({len(raw)} > {MAX_BOXES})")
        boxes = []
        for i, bx in enumerate(raw):
            if not isinstance(bx, list) or len(bx) != 6:
                raise TerrainError(f"box {i} must be [x0,y0,z0,x1,y1,z1]")
            v = [_num(c, f"box {i}") for c in bx]
            if max(abs(c) for c in v) > MAX_COORD:
                raise TerrainError(f"box {i} outside +-{MAX_COORD} m")
            boxes.append(v)
        region = body.get("region")
        if region is not None:
            if not isinstance(region, list) or len(region) != 6:
                raise TerrainError("region must be [x0,y0,z0,x1,y1,z1]")
            region = [_num(c, "region") for c in region]
        meta = {k: body[k] for k in ("id", "anchor", "scale", "dim") if k in body}
        return cls(boxes or np.zeros((0, 6)), region, meta)

    def summary(self):
        b = self.boxes
        out = {"ok": True, "boxes": int(len(b)), "hash": self.hash, "floor_z": round(self.floor_z, 4)}
        out.update({k: v for k, v in self.meta.items()})
        if len(b):
            out["bounds"] = [round(float(x), 4) for x in np.concatenate([b[:, :3].min(0), b[:, 3:].max(0)])]
        if self.region is not None:
            out["region"] = [round(float(x), 4) for x in self.region]
        return out

    # ------------------------------------------------------------------ height queries (numpy, no MuJoCo)
    def _near(self, x, y, r):
        b = self.boxes
        if not len(b):
            return b
        m = (b[:, 0] < x + r) & (b[:, 3] > x - r) & (b[:, 1] < y + r) & (b[:, 4] > y - r)
        return b[m]

    def ground(self, x, y, ref, boxes=None):
        """Standable surface height under (x, y) for feet near `ref`: the highest box top (or the fallback floor)
        that is <= ref + STEP_TOL with HEADROOM of free space above it. None if (x, y) is solid there."""
        b = self._near(x, y, 1e-6) if boxes is None else boxes
        if len(b):
            m = (b[:, 0] <= x) & (b[:, 3] > x) & (b[:, 1] <= y) & (b[:, 4] > y)
            b = b[m]
        tops = [self.floor_z] + [float(t) for t in b[:, 5]] if len(b) else [self.floor_z]
        best = None
        for h in sorted(set(tops), reverse=True):
            if h > ref + STEP_TOL:
                continue
            if h < ref - MAX_DROP:
                break                          # nothing to stand on within reach below
            if len(b) and np.any((b[:, 2] < h + HEADROOM) & (b[:, 5] > h + 1e-4)):
                continue                       # something overlaps the space just above this surface
            best = h
            break
        return best

    def spot_ok(self, x, y, ref, boxes=None):
        """Height h if a duck can stand at (x, y): flat footprint, no box in the footprint x headroom volume."""
        b = self._near(x, y, FOOT_R + 0.05) if boxes is None else boxes
        h = self.ground(x, y, ref, b)
        if h is None:
            return None
        for dx, dy in ((FOOT_R, 0), (-FOOT_R, 0), (0, FOOT_R), (0, -FOOT_R),
                       (FOOT_R, FOOT_R), (-FOOT_R, FOOT_R), (FOOT_R, -FOOT_R), (-FOOT_R, -FOOT_R)):
            g = self.ground(x + dx, y + dy, h, b)
            if g is None or abs(g - h) > FLAT_TOL:
                return None
        if len(b):
            r = FOOT_R + 0.03                  # a little wall clearance
            hit = ((b[:, 0] < x + r) & (b[:, 3] > x - r) & (b[:, 1] < y + r) & (b[:, 4] > y - r)
                   & (b[:, 2] < h + HEADROOM) & (b[:, 5] > h + 0.004))
            if np.any(hit):
                return None
        return h

    def find_spawn(self, x, y, ref, max_r=1.5, limit_r=None, step=0.06, same_level=True):
        """Nearest safe standing spot to (x, y) (spiral search). Returns (x, y, ground_h); falls back to
        (x, y, ground or ref) if nothing is found."""
        local = self._near(x, y, max_r + 0.3)
        cands = [(0.0, 0.0)]
        n = int(max_r / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if (i or j) and math.hypot(i, j) * step <= max_r:
                    cands.append((i * step, j * step))
        cands.sort(key=lambda c: c[0] * c[0] + c[1] * c[1])
        fallback = None
        deadline = time.perf_counter() + 0.15   # runs on the sim thread: bounded
        # pass 1: same floor level as ref; pass 2: allow surfaces up to SPAWN_CLIMB higher (e.g. the anchor was set
        # standing on a slab floor, or blocks were placed under the duck); else the nearest spot at any level
        for r, good in ((ref, lambda h: (not same_level) or abs(h - ref) <= STEP_TOL),
                        (ref + SPAWN_CLIMB, lambda h: h >= ref - STEP_TOL)):
            if fallback is not None:
                break                           # a lower level exists: prefer it over climbing
            for dx, dy in cands:
                if time.perf_counter() > deadline:
                    break
                cx, cy = x + dx, y + dy
                if limit_r is not None and math.hypot(cx, cy) > limit_r:
                    continue
                b = local[(local[:, 0] < cx + 0.2) & (local[:, 3] > cx - 0.2)
                           & (local[:, 1] < cy + 0.2) & (local[:, 4] > cy - 0.2)] if len(local) else local
                h = self.spot_ok(cx, cy, r, b)
                if h is not None:
                    if good(h):
                        return cx, cy, h
                    if fallback is None:
                        fallback = (cx, cy, h)
        if fallback is not None:
            return fallback
        g = self.ground(x, y, ref)
        return x, y, (ref if g is None else g)


# ---------------------------------------------------------------------- MJCF
def apply(spec, terrain):
    """Add terrain boxes to the spec's worldbody (call before compile); move the flat floor to the fallback z."""
    if terrain is None or (not len(terrain) and terrain.region is None):
        return
    floor = None
    for g in spec.worldbody.geoms:
        if g.name == "floor":
            floor = g
            break
    fr = list(floor.friction) if floor is not None else [1.0, 0.005, 0.0001]
    solref = list(floor.solref) if floor is not None else [0.02, 1.0]
    solimp = list(floor.solimp) if floor is not None else [0.9, 0.95, 0.001, 0.5, 2.0]
    if floor is not None:
        floor.pos = [0.0, 0.0, terrain.floor_z]
    for i, (x0, y0, z0, x1, y1, z1) in enumerate(terrain.boxes):
        g = spec.worldbody.add_geom()
        g.name = f"terrain_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]
        g.pos = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
        g.group = TERRAIN_GROUP
        g.contype = 1
        g.conaffinity = 1
        g.condim = 3
        g.friction = fr
        g.solref = solref
        g.solimp = solimp
        g.rgba = [0.45, 0.42, 0.38, 1.0]
    for i, (size, pos, quat) in enumerate(ramps(terrain.boxes)):
        g = spec.worldbody.add_geom()
        g.name = f"terrain_ramp_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.quat = size, pos, quat
        g.group = TERRAIN_GROUP
        g.contype = 1
        g.conaffinity = 1
        g.condim = 3
        g.friction = fr
        g.solref = solref
        g.solimp = solimp
        g.rgba = [0.5, 0.3, 0.3, 1.0]


def ramps(boxes):
    """Thin tilted boxes along the exposed sides of very low boxes (carpets, snow layers): the policy trips over a
    sharp 1.6 cm edge, a RAMP_SLOPE ramp it walks over. Returns [(size3, pos3, quat4), ...]."""
    out = []
    t = 0.01
    for x0, y0, z0, x1, y1, z1 in boxes:
        h = z1 - z0
        if h > RAMP_MAX_H:
            continue
        run = h / RAMP_SLOPE
        th = math.atan2(h, run)
        half = math.hypot(run, h) / 2
        c, s = math.cos(th / 2), math.sin(th / 2)
        nz, nh = math.cos(th), math.sin(th)       # top-face normal: (nh outward, nz up)
        zc = (z0 + z1) / 2 - t * nz
        # +x side: rotate about +y by +th (going +x goes down); -x: about +y by -th; +y: about +x by -th; -y: +th
        out.append(([half, (y1 - y0) / 2, t], [x1 + run / 2 - t * nh, (y0 + y1) / 2, zc], [c, 0, s, 0]))
        out.append(([half, (y1 - y0) / 2, t], [x0 - run / 2 + t * nh, (y0 + y1) / 2, zc], [c, 0, -s, 0]))
        out.append(([(x1 - x0) / 2, half, t], [(x0 + x1) / 2, y1 + run / 2 - t * nh, zc], [c, -s, 0, 0]))
        out.append(([(x1 - x0) / 2, half, t], [(x0 + x1) / 2, y0 - run / 2 + t * nh, zc], [c, s, 0, 0]))
    return out


def terrain_geoms(model):
    return np.nonzero(model.geom_group == TERRAIN_GROUP)[0]


# ---------------------------------------------------------------------- steering
def _ray(sim, p, ang):
    vec = np.array([math.cos(ang), math.sin(ang), 0.0])
    gid = np.zeros(1, dtype=np.int32)
    d = mujoco.mj_ray(sim.model, sim.data, np.asarray(p, dtype=np.float64), vec, RAY_GROUP, 1, -1, gid)
    return AV_RANGE if d < 0 or d > AV_RANGE else d


DOWN_GROUP = np.array([1, 0, 0, 0, 1, 0], dtype=np.uint8)   # floor plane + terrain


def trunk_height(sim):
    """Trunk height above the surface directly below it (terrain or floor); pos z without terrain."""
    p = sim.pos()
    if getattr(sim, "terrain", None) is None:
        return float(p[2])
    gid = np.zeros(1, dtype=np.int32)
    d = mujoco.mj_ray(sim.model, sim.data, np.array(p, dtype=np.float64), np.array([0.0, 0.0, -1.0]),
                      DOWN_GROUP, 1, -1, gid)
    return float(p[2]) if d < 0 else float(d)


def clearance(sim, ang):
    """Free distance from the trunk centre along world heading `ang` (min of a low and a mid ray)."""
    p = sim.pos()
    feet = float(p[2]) - STAND_H
    lo = _ray(sim, (p[0], p[1], feet + AV_LOW), ang)
    mid = _ray(sim, (p[0], p[1], feet + 0.15), ang)
    return min(lo, mid)


def avoid(sim, v):
    """Adjust a body-frame velocity command [vx, vy, wz] so the duck turns away from walls instead of wedging.
    No-op without terrain. Ledges are NOT avoided (it falls off them, as the real one would)."""
    if getattr(sim, "terrain", None) is None or not len(sim.terrain) or sim.mode not in ("walk", "follow"):
        sim._av_sign = 0.0
        return v
    v = np.array(v, dtype=np.float32)
    vx = float(v[0])
    t = sim.t
    p = sim.pos()
    # stuck detector: trying to walk forward but not getting anywhere -> pick a new heading
    st = getattr(sim, "_av_stuck", None)
    if vx > 0.08:
        if st is None or math.hypot(p[0] - st[1], p[1] - st[2]) > STUCK_D:
            sim._av_stuck = (t, float(p[0]), float(p[1]))
        elif t - st[0] > STUCK_S:
            sim._av_stuck = (t, float(p[0]), float(p[1]))
            sign = getattr(sim, "_av_sign", 0.0) or (1.0 if np.random.random() < 0.5 else -1.0)
            if sim.mode == "walk":
                sim.walk_heading = wrap(sim.yaw() + sign * 2.0)
            sim._av_sign = sign
            sim._av_until = t + 2.0
    else:
        sim._av_stuck = None
    if vx <= 0.02:
        return v
    yaw = sim.yaw()
    head = yaw + math.atan2(float(v[1]), vx)
    d0 = clearance(sim, head)
    dl = clearance(sim, head + 0.6)
    dr = clearance(sim, head - 0.6)
    sign = getattr(sim, "_av_sign", 0.0)
    until = getattr(sim, "_av_until", 0.0)
    if d0 < AV_SLOW:
        if not sign:
            sign = 1.0 if dl >= dr else -1.0
        sim._av_until = t + 0.6
        k = max(0.0, min(1.0, (d0 - AV_STOP) / (AV_SLOW - AV_STOP)))
        v[0] = vx * k
        v[1] = 0.0
        v[2] = float(np.clip(sign * (1.2 - 0.6 * k), -1.2, 1.2))
        if sim.mode == "walk":
            sim.walk_heading = wrap(yaw + sign * 0.5)   # heading hold follows the turn instead of fighting it
    elif sign and t < until:
        v[2] = float(np.clip(v[2] + sign * 0.6, -1.2, 1.2))
        if sim.mode == "walk":
            sim.walk_heading = wrap(yaw + sign * 0.3)
    else:
        sign = 0.0
        if dl < AV_SIDE:
            v[2] = float(np.clip(v[2] - 0.6, -1.2, 1.2))
        elif dr < AV_SIDE:
            v[2] = float(np.clip(v[2] + 0.6, -1.2, 1.2))
    sim._av_sign = sign
    return v


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------- spawn / swap
def spawn_point(sim, x=0.0, y=0.0, ref=0.0):
    """(x, y, ground_h) for a reset: nearest safe spot to (x, y) inside the leash. Clears steering state."""
    sim._av_sign, sim._av_until, sim._av_stuck = 0.0, 0.0, None
    override = getattr(sim, "spawn_override", None)   # multi.py: a free spot away from the other ducks
    if override is not None:
        return override(sim, x, y, ref)
    tr = getattr(sim, "terrain", None)
    if tr is None or (not len(tr) and tr.region is None):
        return x, y, 0.0
    return tr.find_spawn(x, y, ref, limit_r=max(0.3, sim.leash - 0.1))


def build(sim, terrain):
    """Compile a new (model, data, bam) with `terrain` (heavy part; call WITHOUT the sim lock)."""
    import duck_sim as ds
    sp = getattr(sim, "_scene_path", None)        # brains.BrainSim: the robot this duck wears now (skates or not)
    scene = sp(getattr(sim, "variant", "default")) if sp is not None else None
    return ds.load_model(sim.use_bam, terrain, scene=scene)


def swap(sim, terrain, built, log=print):
    """Swap the freshly built model in (call WITH the sim lock). Keeps qpos/qvel/controller state; respawns
    standing on the new ground below the duck if the ground under it moved or it would be inside a block."""
    m, d, ctrl = built
    od = sim.data
    d.qpos[:] = od.qpos
    d.qvel[:] = od.qvel
    if m.na:
        d.act[:] = od.act
    d.ctrl[:] = od.ctrl
    d.qacc_warmstart[:] = od.qacc_warmstart
    d.time = od.time
    if ctrl is not None and sim.bam is not None:
        ctrl.q_target[:] = sim.bam.q_target
        ctrl.last_ts = sim.bam.last_ts
    if sim.backlash_j and sim._backlash_locked:
        m.jnt_range[sim.backlash_j] = sim.backlash_range * 0.02
    sim.model, sim.data, sim.bam = m, d, ctrl
    sim.terrain = terrain if (terrain is not None and (len(terrain) or terrain.region is not None)) else None
    mujoco.mj_forward(m, d)
    # inside a block now?
    tg = set(terrain_geoms(m).tolist())
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in tg) != (c.geom2 in tg):
            worst = min(worst, float(c.dist))
    p = sim.pos().copy()
    reason = None
    if worst < -0.01:
        reason = f"penetration {worst * 1000:.0f} mm"
    elif sim.terrain is not None and not sim.fallen and sim.policy == "velstand":
        feet = float(p[2]) - STAND_H
        g = sim.terrain.ground(float(p[0]), float(p[1]), feet + 0.02)
        if g is None or abs(g - feet) > 0.04:
            reason = f"ground moved (feet {feet:.3f} -> {g if g is None else round(g, 3)})"
    if reason:
        respawn(sim, float(p[0]), float(p[1]), float(p[2]) - STAND_H)
        q = sim.pos()
        log(f"[terrain] respawned at ({q[0]:.2f},{q[1]:.2f},{q[2]:.3f}): {reason}")
    return reason


def respawn(sim, x, y, ref):
    """Reset the duck standing at the nearest safe spot to (x, y), keeping yaw and the high-level mode."""
    mode, yaw = sim.mode, sim.yaw()
    keep = mode if mode in ("walk", "follow", "stand") else "stand"
    walk_cmd = sim.walk_cmd
    sim.reset()
    override = getattr(sim, "spawn_override", None)   # multi.py: also keep clear of the other ducks
    if override is not None:
        sx, sy, h = override(sim, x, y, ref, same_level=False)
    else:
        sx, sy, h = (sim.terrain.find_spawn(x, y, ref, limit_r=max(0.3, sim.leash - 0.1), same_level=False)
                     if sim.terrain else (x, y, 0.0))   # nearest spot at or below the feet, else climb up
    import duck_sim as ds
    d = sim.data
    d.qpos[sim.fq:sim.fq + 3] = [sx, sy, h + ds.SPAWN_Z]
    d.qpos[sim.fq + 3:sim.fq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(sim.model, d)
    sim.walk_cmd = walk_cmd
    sim._enter(keep)


class Manager:
    """Latest-wins terrain rebuild worker: POST /terrain hands a Terrain over; the worker compiles the new model
    without holding the sim lock (MuJoCo releases the GIL while compiling) and swaps it in under the lock."""

    def __init__(self, sim, lock, log=print):
        self.sim, self.lock, self.log = sim, lock, log
        self.cv = threading.Condition()
        self.pending = None
        self.current = None           # Terrain applied to the sim (None = flat floor)
        self.queued_hash = None
        self.builds = 0
        self.last_build_s = 0.0
        self.last_reason = None
        self.th = threading.Thread(target=self._run, name="terrain", daemon=True)
        self.th.start()

    def submit(self, body):
        t = Terrain.from_json(body)
        with self.cv:
            cur = self.pending or self.current
            if cur is not None and cur.hash == t.hash and cur.meta.get("id") == t.meta.get("id"):
                return {"ok": True, "hash": t.hash, "boxes": len(t), "changed": False}
            if cur is None and not len(t) and t.region is None:
                return {"ok": True, "hash": t.hash, "boxes": 0, "changed": False}
            if cur is not None and cur.hash == t.hash:       # same geometry, new id: just relabel
                cur.meta = t.meta
                return {"ok": True, "hash": t.hash, "boxes": len(t), "changed": False}
            self.pending = t
            self.cv.notify()
        return {"ok": True, "hash": t.hash, "boxes": len(t), "changed": True, "queued": True}

    def brief(self):
        cur = self.current
        return {"boxes": len(cur) if cur is not None else 0, "hash": cur.hash if cur is not None else None,
                "pending": self.pending is not None, "builds": self.builds}

    def summary(self):
        with self.cv:
            cur, pend = self.current, self.pending
        out = cur.summary() if cur is not None else {"ok": True, "boxes": 0, "hash": None, "floor_z": 0.0}
        out.update({"pending": pend is not None, "builds": self.builds,
                    "last_build_ms": round(self.last_build_s * 1000, 1), "last_respawn": self.last_reason})
        return out

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                t = self.pending
            time.sleep(0.2)                  # debounce bursts
            with self.cv:
                if self.pending is not t:
                    continue
            t0 = time.perf_counter()
            try:
                built = build(self.sim, t)
                with self.lock:
                    m0, m1 = self.sim.model, built[0]
                    if (m1.nq, m1.nv, m1.nbody, m1.nu) != (m0.nq, m0.nv, m0.nbody, m0.nu):
                        built = build(self.sim, t)     # robot variant switched meanwhile (brains): rebuild in lock
                    self.last_reason = swap(self.sim, t, built, self.log)
            except Exception as e:  # never kill the worker
                self.log(f"[terrain] rebuild failed: {e!r}")
                with self.cv:
                    if self.pending is t:
                        self.pending = None
                continue
            self.last_build_s = time.perf_counter() - t0
            self.builds += 1
            with self.cv:
                self.current = t if (len(t) or t.region is not None) else None
                if self.pending is t:
                    self.pending = None
            self.log(f"[terrain] applied {len(t)} boxes hash={t.hash} floor_z={t.floor_z:.3f} "
                     f"in {self.last_build_s * 1000:.0f} ms")
