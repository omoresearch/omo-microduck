"""Several Microducks in one sidecar (CONTRACT.md "Multiple ducks").

Design: one BrainSim (own MjModel/MjData, brains, tricks, talk, quack counter, falls/auto-reset) per duck, all stepped
by the one 50 Hz sim thread under one lock. The ducks bump into each other through mocap "ghost" capsules
(ghosts.py): before each control step duck i's ghosts are moved onto the other ducks' trunks, so collisions are real
MuJoCo contacts in each duck's world. Why not one shared MjModel with N robot instances: each duck must be able to
swap its own robot (roller skates = a different MJCF, ~0.35 s compile), reset alone, rebuild with terrain and keep
its own BAM actuator controller - in a shared world every skates on/off would recompile and re-index all N robots
and reset/respawn would need per-instance qpos surgery. Per-duck worlds keep all of that unchanged; the ghosts give
the physical bumping. Limitation: a duck feels the other as a rigid capsule (trunk-attached), not its exact legs.

Shared: terrain (one POST /terrain -> rebuilt for every duck), leash (around the anchor = sim origin), player.
Persisted: the roster (ducks.json: id, persona, last position) so ducks come back after a restart.
Personas: personas.json (shared with the voice backend voice-live/duck.mjs).
"""

import json
import math
import os
import re
import threading
import time

import numpy as np

import duck_sim as ds
import ghosts
import terrain as terrain_mod

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.environ.get("MICRODUCK_STATE", os.path.join(HERE, "ducks.json"))
PERSONAS_PATH = os.environ.get("MICRODUCK_PERSONAS", os.path.join(HERE, "personas.json"))
MAX_DUCKS = max(1, min(8, int(os.environ.get("MICRODUCK_MAX_DUCKS", "4"))))
VOICES = ("arbor", "breeze", "cove", "ember", "juniper", "maple", "sol", "spruce", "vale")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,19}$")
RESERVED = {"all", "list", "spawn", "remove", "help", "here", "scale", "brains", "brain", "trick", "walk", "stop",
            "sit", "stand", "follow", "quack", "left", "right", "reset", "turn", "talk", "default", "none"}
SEP = 0.40              # min trunk distance between ducks for a spawn spot (m)
HOME_R = 0.55           # home spots: ring around the anchor
FOLLOW_GAP = 0.30       # each extra follower stops this much further from the player
PEER_SLOW = 0.42        # peer avoid: slow + turn when another duck is this close ahead (m, trunk to trunk)
PEER_STOP = 0.24
SAVE_EVERY_S = 30.0

BUILTIN_PERSONA = {"id": "quackers", "name": "Quackers", "voice": "juniper", "color": "#F2C230", "brain": "default",
                   "trick": "groove", "quack": {"pitch": 1.0, "style": "bright"}, "aliases": [],
                   "personality": "bubbly, curious, cheeky little robot buddy"}


def log_default(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _hex(c, default="#F2C230"):
    s = str(c or "").strip()
    return s.upper() if re.match(r"^#[0-9A-Fa-f]{6}$", s) else default


def load_personas(path=PERSONAS_PATH, log=log_default):
    """[persona] from personas.json (validated; bad entries skipped). Falls back to the built-in Quackers."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return [dict(BUILTIN_PERSONA)]
    except Exception as e:  # noqa: BLE001
        log(f"[multi] {path}: {e!r}; using the built-in persona")
        return [dict(BUILTIN_PERSONA)]
    out, seen = [], set()
    for p in raw.get("personas", []) if isinstance(raw, dict) else []:
        try:
            name = str(p["name"]).strip()
            pid = str(p.get("id") or name).strip().lower()
            if not NAME_RE.match(name) or name.lower() in RESERVED or not re.match(r"^[a-z0-9-]{1,24}$", pid):
                raise ValueError(f"bad name/id {name!r}/{pid!r}")
            if pid in seen or name.lower() in seen:
                raise ValueError(f"duplicate {pid}")
            voice = str(p.get("voice", "juniper")).lower()
            if voice not in VOICES:
                raise ValueError(f"voice {voice!r} not in {VOICES}")
            q = p.get("quack") or {}
            e = {"id": pid, "name": name, "voice": voice, "color": _hex(p.get("color")),
                 "brain": str(p.get("brain") or "default").lower(), "trick": str(p.get("trick") or "").lower() or None,
                 "quack": {"pitch": float(min(2.0, max(0.5, float(q.get("pitch", 1.0))))), "style": str(q.get("style", ""))},
                 "aliases": [str(a).lower() for a in (p.get("aliases") or []) if str(a).strip()],
                 "personality": str(p.get("personality") or "")[:2000], "tagline": str(p.get("tagline") or "")[:160]}
            seen.update({pid, name.lower()})
            out.append(e)
        except Exception as ex:  # noqa: BLE001
            log(f"[multi] personas.json: skipping entry: {ex}")
    return out or [dict(BUILTIN_PERSONA)]


_OFFS = {}


def _offsets(r, step=0.07):
    """Grid offsets within radius r, nearest first (cached)."""
    key = (round(r, 2), step)
    if key not in _OFFS:
        n = int(r / step) + 1
        c = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1) if math.hypot(i, j) * step <= r]
        c.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
        _OFFS[key] = c
    return _OFFS[key]


class Duck:
    def __init__(self, did, persona, sim):
        self.id = did
        self.persona = persona
        self.name = persona["name"]
        self.sim = sim
        self.home = (0.0, 0.0)
        self.follow_target = None    # another duck's id ("follow <duck>"), else the player
        self.seq = 0
        self.created = time.time()

    def public(self, brief=False):
        s = self.sim
        p = s.pos()
        out = {"id": self.id, "name": self.name, "persona": self.persona["id"], "voice": self.persona["voice"],
               "color": self.persona["color"], "brain": s.slot, "trick": s.trick.id if s.trick else None,
               "mode": s.public_mode(), "pos": [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)],
               "yaw": round(float(s.yaw()), 3), "fallen": bool(s.fallen)}
        if not brief:
            out.update({"robot": s.variant, "quack": s.quack, "talking": ds.talk_mod.talking(s),
                        "quack_pitch": self.persona["quack"]["pitch"], "falls": s.falls, "resets": s.resets,
                        "follow": self.follow_target, "home": [round(self.home[0], 3), round(self.home[1], 3)]})
        return out


class Fleet:
    """All ducks. Every method that touches sims must be called with self.lock held unless noted."""

    def __init__(self, log=log_default, sim_factory=None, state_path=STATE_PATH, personas_path=PERSONAS_PATH,
                 max_ducks=MAX_DUCKS, autoload=True, leash=ds.LEASH_DEFAULT):
        self.log = log
        self.max = int(max_ducks)
        ds.GHOSTS = self.max - 1                     # every model built from now on carries the ghosts
        if sim_factory is None:
            import brains
            sim_factory = brains.BrainSim
        self.sim_factory = sim_factory
        self.state_path = state_path
        self.personas_path = personas_path
        self.personas = load_personas(personas_path, log)
        self.lock = threading.Lock()
        self.ducks = []                              # roster order; ducks[0] is the default duck (old API)
        self.next_n = 1
        self.leash = float(leash)
        self.player = None
        self.terrain_mgr = None                      # FleetTerrain (set by the service)
        self.peer_contacts = 0                       # robot-ghost contact steps (diagnostic)
        self.min_sep = 9.0                           # smallest trunk distance seen since start
        self._last_save = 0.0
        self._save_lock = threading.Lock()
        if autoload:
            self.load_roster()

    # ------------------------------------------------------------------ personas / lookup
    def persona(self, sel):
        s = str(sel or "").strip().lower()
        for p in self.personas:
            if s in (p["id"], p["name"].lower()):
                return p
        hits = [p for p in self.personas if p["id"].startswith(s) or p["name"].lower().startswith(s)] if s else []
        return hits[0] if len(hits) == 1 else None

    def default(self):
        return self.ducks[0] if self.ducks else None

    def resolve(self, sel):
        """'' / None -> [default]; 'all' -> every duck; id, name, persona id, alias or unambiguous name prefix.
        Returns (ducks, error)."""
        if sel is None or str(sel).strip() == "":
            d = self.default()
            return ([d], None) if d else ([], "no ducks spawned (!duck spawn)")
        s = str(sel).strip().lower()
        if s in ("all", "*", "everyone", "ducks"):
            return (list(self.ducks), None) if self.ducks else ([], "no ducks spawned (!duck spawn)")
        for d in self.ducks:
            if s in (d.id, d.name.lower(), d.persona["id"]) or s in d.persona.get("aliases", []):
                return [d], None
        hits = [d for d in self.ducks if d.name.lower().startswith(s) or d.persona["id"].startswith(s)]
        if len(hits) == 1:
            return hits, None
        if len(hits) > 1:
            return [], f"{sel!r} is ambiguous: {', '.join(d.name for d in hits)}"
        return [], f"no duck named {sel!r} (ducks: {', '.join(d.name for d in self.ducks) or 'none'})"

    def unused_personas(self):
        used = {d.persona["id"] for d in self.ducks}
        return [p for p in self.personas if p["id"] not in used]

    # ------------------------------------------------------------------ spawn spots
    def home_for(self, k):
        if k == 0:
            return (0.0, 0.0)
        a = (k - 1) * 2 * math.pi / max(3, self.max - 1) + math.pi / 2
        r = min(HOME_R, max(0.3, self.leash * 0.4))
        return (r * math.cos(a), r * math.sin(a))

    def free_spot(self, duck, x, y, ref=0.0, same_level=True, others=None):
        """Nearest standable spot to (x, y) inside the leash and >= SEP from every other duck. Bounded time
        (runs on the sim thread for auto-resets). -> (x, y, ground_h)."""
        if others is None:
            others = [d.sim.pos()[:2].copy() for d in self.ducks if d is not duck]
        tr = duck.sim.terrain if duck is not None else (self.terrain_mgr.current if self.terrain_mgr else None)
        lim = max(0.3, self.leash - 0.12)
        cands = _offsets(2 * lim)
        deadline = time.perf_counter() + 0.12
        fallback = None
        for dx, dy in cands:
            if time.perf_counter() > deadline:
                break
            cx, cy = x + dx, y + dy
            if math.hypot(cx, cy) > lim:
                continue
            if any(math.hypot(cx - o[0], cy - o[1]) < SEP for o in others):
                continue
            if tr is None or (not len(tr) and tr.region is None):
                return cx, cy, 0.0
            h = tr.spot_ok(cx, cy, ref)
            if h is None and not same_level:
                h = tr.spot_ok(cx, cy, ref + terrain_mod.SPAWN_CLIMB)
            if h is not None:
                if abs(h - ref) <= terrain_mod.STEP_TOL or not same_level:
                    return cx, cy, h
                if fallback is None:
                    fallback = (cx, cy, h)
        if fallback is not None:
            return fallback
        if tr is not None and (len(tr) or tr.region is not None):
            return tr.find_spawn(x, y, ref, limit_r=lim, same_level=same_level)
        return x, y, 0.0

    def _override_for(self, duck):
        def override(sim, x, y, ref, same_level=True):
            sim._av_sign, sim._av_until, sim._av_stuck = 0.0, 0.0, None
            if x == 0.0 and y == 0.0:          # DuckSim.reset(): respawn at this duck's home spot
                x, y = duck.home
            return self.free_spot(duck, x, y, ref, same_level=same_level)
        return override

    # ------------------------------------------------------------------ spawn / remove (call WITHOUT the lock)
    def _new_id(self):
        taken = {d.id for d in self.ducks}
        while f"d{self.next_n}" in taken:
            self.next_n += 1
        did = f"d{self.next_n}"
        self.next_n += 1
        return did

    def spawn(self, persona_sel=None, pos=None, yaw=None, did=None, save=True):
        """Create a duck (heavy part outside the lock). Returns (duck_public, error)."""
        with self.lock:
            if len(self.ducks) >= self.max:
                return None, f"max {self.max} ducks (remove one first)"
            if persona_sel:
                p = self.persona(persona_sel)
                if p is None:
                    return None, f"unknown persona {persona_sel!r} (personas: {', '.join(x['name'] for x in self.personas)})"
                if any(d.persona["id"] == p["id"] for d in self.ducks):
                    return None, f"{p['name']} is already here"
            else:
                free = self.unused_personas()
                if not free:
                    return None, "every persona is already spawned"
                p = free[0]
            did = did or self._new_id()
            k = len(self.ducks)
            terrain = self.terrain_mgr.current if self.terrain_mgr else None
        t0 = time.perf_counter()
        sim = self.sim_factory(log=self.log, leash=self.leash)      # compile + reset, no lock held
        sim.lock = self.lock
        if terrain is not None:
            built = terrain_mod.build(sim, terrain)
            terrain_mod.swap(sim, terrain, built, log=lambda *a: None)
            sim._fleet_terrain = terrain
        duck = Duck(did, p, sim)
        with self.lock:
            if len(self.ducks) >= self.max or any(d.persona["id"] == p["id"] for d in self.ducks):
                return None, "roster changed meanwhile; try again"
            used_homes = [d.home for d in self.ducks]
            homes = [self.home_for(i) for i in range(self.max)]
            free_homes = [h for h in homes if all(math.hypot(h[0] - u[0], h[1] - u[1]) > 0.05 for u in used_homes)]
            duck.home = free_homes[0] if free_homes else self.home_for(k)
            sim.spawn_override = self._override_for(duck)
            sim.peer_avoid = self._peer_avoid_for(duck)
            sim.leash = self.leash
            if self.player is not None:
                sim.set_player(*self.player)
            tx, ty = (pos[0], pos[1]) if pos else duck.home
            sx, sy, h = self.free_spot(duck, float(tx), float(ty), 0.0, same_level=False)
            d = sim.data                          # (the constructor already reset it; just place it)
            d.qpos[sim.fq:sim.fq + 3] = [sx, sy, h + ds.SPAWN_Z]
            yw = float(yaw) if yaw is not None else math.atan2(-sy, -sx) if math.hypot(sx, sy) > 0.05 else 0.0
            d.qpos[sim.fq + 3:sim.fq + 7] = ghosts.yaw_quat(yw)
            import mujoco
            mujoco.mj_forward(sim.model, d)
            self.ducks.append(duck)
            out = duck.public()
        self.log(f"[multi] spawned {duck.id} {duck.name} (voice {p['voice']}) at ({sx:.2f},{sy:.2f}) "
                 f"in {(time.perf_counter() - t0) * 1000:.0f} ms; {len(self.ducks)} duck(s)")
        if p.get("brain") and p["brain"] != "default":
            with self.lock:
                ok, err = sim.command("brain", name=p["brain"])
                if ok:
                    sim.command("stop")              # favourite brain loaded for `walk`, but start standing
            if not ok:
                self.log(f"[multi] {duck.name}: favourite brain {p['brain']!r}: {err}")
        if save:
            self.save_roster()
        return out, None

    def spawn_all(self):
        """Spawn every persona that isn't here yet, up to the max. -> (spawned list, errors)."""
        spawned, errs = [], []
        for p in list(self.unused_personas()):
            with self.lock:
                full = len(self.ducks) >= self.max
            if full:
                errs.append(f"max {self.max} ducks reached")
                break
            d, err = self.spawn(p["id"], save=False)
            if d:
                spawned.append(d)
            else:
                errs.append(err)
        self.save_roster()
        return spawned, errs

    def remove(self, sel):
        with self.lock:
            targets, err = self.resolve(sel)
            if err:
                return [], err
            for d in targets:
                self.ducks.remove(d)
                d.sim.spawn_override = None
            for d in self.ducks:
                if d.follow_target in {t.id for t in targets}:
                    d.follow_target = None
            names = [t.public(brief=True) for t in targets]
        self.log(f"[multi] removed {', '.join(t['name'] for t in names)}; {len(self.ducks)} duck(s) left")
        self.save_roster()
        return names, None

    # ------------------------------------------------------------------ persistence
    def save_roster(self, force=True):
        with self.lock:
            data = {"version": 1, "next": self.next_n, "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "ducks": [{"id": d.id, "persona": d.persona["id"], "name": d.name,
                               "pos": [round(float(x), 3) for x in d.sim.pos()[:2]], "yaw": round(float(d.sim.yaw()), 3),
                               "home": [round(d.home[0], 3), round(d.home[1], 3)]} for d in self.ducks]}
        with self._save_lock:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.state_path)
            self._last_save = time.time()

    def load_roster(self):
        try:
            with open(self.state_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = None
        except Exception as e:  # noqa: BLE001
            self.log(f"[multi] {self.state_path} unreadable ({e!r}); starting with the default duck")
            data = None
        if data is None:          # first start: the existing single duck becomes duck 1
            self.spawn(self.personas[0]["id"], did="d1")
            return
        self.next_n = int(data.get("next", 1))
        for e in data.get("ducks", [])[: self.max]:
            p = self.persona(e.get("persona")) or self.persona(e.get("name"))
            if p is None:
                self.log(f"[multi] roster: persona {e.get('persona')!r} no longer exists; skipped")
                continue
            d, err = self.spawn(p["id"], pos=e.get("pos"), yaw=e.get("yaw"), did=e.get("id"), save=False)
            if err:
                self.log(f"[multi] roster: {e.get('name')}: {err}")
        self.log(f"[multi] roster restored: {', '.join(d.name for d in self.ducks) or 'no ducks'}")
        self.save_roster()

    def maybe_save(self):
        """Called from the sim thread WITHOUT the lock: periodic position save."""
        if time.time() - self._last_save > SAVE_EVERY_S:
            try:
                self.save_roster()
            except Exception as e:  # noqa: BLE001
                self.log(f"[multi] roster save failed: {e!r}")
                self._last_save = time.time()

    # ------------------------------------------------------------------ commands
    def command(self, sel, cmd, **kw):
        """-> (ok, mode_of_first, results[], error). With the lock held."""
        targets, err = self.resolve(sel)
        if err:
            return False, None, [], err
        target = kw.pop("target", None)
        results = []
        for d in targets:
            if cmd == "follow":
                d.follow_target = None
                if target:
                    t, e2 = self.resolve(target)
                    if e2 or len(t) != 1 or t[0] is d:
                        results.append({"id": d.id, "name": d.name, "ok": False, "error": e2 or "bad follow target"})
                        continue
                    d.follow_target = t[0].id
            elif cmd in ("walk", "stop", "sit", "stand", "turn", "reset", "brain", "trick"):
                d.follow_target = None
            ok, e = d.sim.command(cmd, **dict(kw))
            r = {"id": d.id, "name": d.name, "ok": bool(ok), "mode": d.sim.public_mode()}
            if e:
                r["error"] = e
            results.append(r)
        ok = any(r["ok"] for r in results)
        first = results[0] if results else {}
        err = None if ok else (first.get("error") or "failed")
        return ok, first.get("mode"), results, err

    def set_player(self, x, y):
        self.player = (float(x), float(y))
        for d in self.ducks:
            d.sim.set_player(x, y)

    def set_leash(self, r):
        self.leash = float(r)
        for d in self.ducks:
            d.sim.leash = self.leash

    # ------------------------------------------------------------------ stepping (lock held)
    def _peer_avoid_for(self, duck):
        def hook(sim, v):
            others = getattr(sim, "_peer_xy", None)
            if not others or sim.mode not in ("walk", "follow"):
                return v
            vx = float(v[0])
            if vx <= 0.02:
                return v
            p = sim.pos()
            yaw = sim.yaw()
            worst, side = 9.0, 0.0
            for ox, oy in others:
                dx, dy = ox - p[0], oy - p[1]
                dist = math.hypot(dx, dy)
                rel = ds.wrap(math.atan2(dy, dx) - yaw)
                if dist < PEER_SLOW and abs(rel) < 1.0 and dist < worst:
                    worst, side = dist, (-1.0 if rel > 0 else 1.0)    # turn away from it
            if worst >= PEER_SLOW:
                return v
            v = np.array(v, dtype=np.float32)
            k = max(0.0, min(1.0, (worst - PEER_STOP) / (PEER_SLOW - PEER_STOP)))
            v[0] = vx * k
            v[2] = float(np.clip(v[2] + side * (1.0 - 0.5 * k), -1.2, 1.2))
            if sim.mode == "walk":
                sim.walk_heading = ds.wrap(yaw + side * 0.4)
            return v
        return hook

    def step(self):
        ducks = self.ducks
        if not ducks:
            return
        poses = []
        for d in ducks:
            s = d.sim
            poses.append((s.data.xpos[s.trunk_id].copy(), s.data.xquat[s.trunk_id].copy()))
        n = len(ducks)
        # follow slots: followers of the player stop further out the further back they are
        if self.player is not None:
            fol = [d for d in ducks if d.sim.mode == "follow" and d.follow_target is None]
            px, py = self.player
            fol.sort(key=lambda d: math.hypot(d.sim.pos()[0] - px, d.sim.pos()[1] - py))
            for r, d in enumerate(fol):
                d.sim.follow_stop = ds.FOLLOW_STOP + FOLLOW_GAP * r
        by_id = {d.id: i for i, d in enumerate(ducks)}
        for i, d in enumerate(ducks):
            s = d.sim
            if d.follow_target is not None:
                j = by_id.get(d.follow_target)
                if j is None:
                    d.follow_target = None
                else:
                    s.player = (float(poses[j][0][0]), float(poses[j][0][1]))
                    s.follow_stop = 0.36
            elif self.player is not None and s.player != self.player:
                s.player = self.player
            others = [poses[j] for j in range(n) if j != i]
            ghosts.update(s, others)
            s._peer_xy = [(float(p[0][0]), float(p[0][1])) for p in others]
            s.step()
            d.seq += 1
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    dd = float(np.linalg.norm(poses[i][0][:2] - poses[j][0][:2]))
                    if dd < self.min_sep:
                        self.min_sep = dd


class FleetTerrain(terrain_mod.Manager):
    """terrain.Manager for the fleet: one POST /terrain is compiled into every duck's model (off the lock, one duck
    at a time) and swapped in under the lock; ducks spawned meanwhile are caught up."""

    def __init__(self, fleet, log=log_default):
        self.fleet = fleet
        super().__init__(None, fleet.lock, log)

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                t = self.pending
            time.sleep(0.2)
            with self.cv:
                if self.pending is not t:
                    continue
            t0 = time.perf_counter()
            reasons = []
            try:
                for _ in range(3 * self.fleet.max):           # catch up ducks spawned during the rebuild
                    with self.lock:
                        todo = [d for d in self.fleet.ducks if getattr(d.sim, "_fleet_terrain", None) is not t]
                    if not todo:
                        break
                    d = todo[0]
                    built = terrain_mod.build(d.sim, t)
                    with self.lock:
                        if d not in self.fleet.ducks:
                            continue
                        m0, m1 = d.sim.model, built[0]
                        if (m1.nq, m1.nv, m1.nbody, m1.nu) != (m0.nq, m0.nv, m0.nbody, m0.nu):
                            built = terrain_mod.build(d.sim, t)    # skates on meanwhile: rebuild for that robot
                        r = terrain_mod.swap(d.sim, t, built, self.log)
                        d.sim._fleet_terrain = t
                        if r:
                            reasons.append(f"{d.name}: {r}")
            except Exception as e:  # noqa: BLE001
                self.log(f"[terrain] rebuild failed: {e!r}")
                with self.cv:
                    if self.pending is t:
                        self.pending = None
                continue
            self.last_reason = "; ".join(reasons) or None
            self.last_build_s = time.perf_counter() - t0
            self.builds += 1
            with self.cv:
                self.current = t if (len(t) or t.region is not None) else None
                if self.pending is t:
                    self.pending = None
            self.log(f"[terrain] applied {len(t)} boxes hash={t.hash} floor_z={t.floor_z:.3f} to "
                     f"{len(self.fleet.ducks)} duck(s) in {self.last_build_s * 1000:.0f} ms")
