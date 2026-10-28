#!/usr/bin/env python3
"""Community duck brains: catalogue, runtime hot-swap (BrainSim) and headless evaluator.

Brains are Microduck ONNX policies from Hugging Face (official pollen-robotics set + community repos),
downloaded by brains_fetch.py into brains/<owner>__<name>/ and treated as untrusted data (onnxruntime
only, I/O signature checked: obs [*,61] -> actions [*,14], optional LSTM h/c state).

  kind perpetual  -> replaces the walk slot (`brain <id>`): drives `walk` mode until stop/turn/follow/sit,
                     velstand still stands, turns, follows, gets up and brings it back inside the leash.
  kind episodic   -> a trick (`trick <id>`): plays once for duration_s, then hands back to velstand standing.
  variant rollers -> needs the roller-skate robot (robot_groundcontact_rollers.xml): the MjModel is swapped
                     at runtime and the duck respawns standing at the same x/y/yaw ("skates on/off").

    .venv/bin/python brains.py eval [ids...]   # nice -n 19; writes brains/catalogue.json
    .venv/bin/python brains.py list            # print the catalogue
"""

import inspect
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import duck_sim as ds  # noqa: E402

try:  # terrain module (another feature); optional
    import terrain as terrain_mod  # noqa: E402
except Exception:  # noqa: BLE001
    terrain_mod = None

BRAIN_DIR = os.path.join(HERE, "brains")
CATALOGUE_PATH = os.path.join(BRAIN_DIR, "catalogue.json")
ID_RE = r"^[a-z0-9-]{1,24}$"

SCENES = {
    "default": None,                      # ds.SCENE_XML at import (allcollisions + backlash; velstand's robot)
    "rollers": "scene_rollers.xml",       # robot_groundcontact_rollers.xml (4 passive wheels)
    "walk": "scene_walk.xml",             # robot_walk.xml (Arena sprint/hill robot) - eval only
    "groundcontact": "scene.xml",         # robot_groundcontact.xml (standup robot: kicks, roulade) - eval only
}
DEFAULT_SCENE = ds.SCENE_XML
# bodies the mod renders (model.json, default robot) -> name in the rollers robot
BODY_ALIASES = {"ankle_left": "ankle_l_v1", "ankle_right": "ankle_r_v1"}

TRICK_MAX_S = 12.0
ROLLER_STEER_GAIN = 1.0

# ----------------------------------------------------------------------------------------- specs
# Candidates evaluated by `brains.py eval`. `id` is the short chat id ([a-z0-9-]{1,24}).
# cat: walk (perpetual, walk slot) | roller (perpetual, rollers) | trick (episodic)
# cmd: constant 13-D command (missing = zeros, or the manifest's command.idle for tricks)
# expect: what "did the thing" means for the evaluator.
SPECS = [
    # --- sprinters / hill climbers (Microduck Arena events; Mjlab-Sprint2m / HillClimb1m5) ---
    dict(id="sprint", featured=True, dir="pollen-robotics__microduck-sprint-2m-reference", cat="walk", vx=0.55, vx_try=[1.0],
         blurb="Pollen's 2 m Sprint reference"),
    dict(id="sprint-fixed", dir="RemiFabre__microduck-sprint-fixed", cat="walk", vx=0.8,
         blurb="RemiFabre's 'unleashed-calm' sprinter"),
    dict(id="zoom", dir="jrubiosainz__microduck-game-sprinter", cat="walk", vx=1.0,
         blurb="PPO-tuned runner (HannesVonEssen's running policy, fine-tuned); ignores the command"),
    dict(id="zoom-remix", dir="jrubiosainz__microduck-game-sprinter-remix4", cat="walk", vx=1.15,
         blurb="Arena sprint remix with per-seed branch tables"),
    dict(id="dashi-remix", dir="dashi__microduck-sprint-remix-r856", cat="walk", vx=1.0,
         blurb="dashi's sprint remix"),
    dict(id="nugget", featured=True, dir="hofftv__microduck-nugget-sprint", cat="walk", vx=0.5,
         blurb="Nugget's walk (carry v3) on the 2 m Sprint"),
    dict(id="sprint-rev5", dir="austindixson__microduck-sprint-improver-rev5", cat="walk", vx=1.0,
         blurb="austindixson's sprint improver rev5"),
    dict(id="hill", featured=True, dir="pollen-robotics__microduck-hill-climb-1m5-reference", cat="walk", vx=0.5, vx_try=[1.0],
         blurb="Pollen's 1.5 m Hill Climb reference"),
    dict(id="hill-fixed", dir="RemiFabre__microduck-hill-fixed", cat="walk", vx=0.6,
         blurb="RemiFabre's hill climber"),
    dict(id="nugget-hill", dir="hofftv__microduck-nugget-hill", cat="walk", vx=0.4,
         blurb="Nugget's steady walk on the Hill Climb"),
    dict(id="nugget-trot", dir="hofftv__microduck-nugget-hill-walk", cat="walk", vx=0.4,
         blurb="Nugget's quick walk on the Hill Climb"),
    dict(id="climber", dir="jrubiosainz__microduck-game-hillclimber-remix5", cat="walk", vx=0.3,
         blurb="Arena hill-climb remix"),
    # --- roller skaters (Mjlab-RollerSprint2m / velocity rollers) ---
    dict(id="roller", featured=True, dir="pollen-robotics__microduck-policies__roller", cat="roller", vx=0.4,
         blurb="Pollen's official roller-skating gait"),
    dict(id="roller-sprint", dir="pollen-robotics__microduck-roller-sprint-2m-reference", cat="roller", vx=0.6,
         blurb="Pollen's 2 m Roller Sprint reference"),
    dict(id="roller-fixed", dir="RemiFabre__microduck-roller-fixed", cat="roller", vx=0.6,
         blurb="RemiFabre's 'free-calm' roller"),
    dict(id="skate", dir="jrubiosainz__microduck-game-roller", cat="roller", vx=1.4,
         blurb="PPO-tuned roller skater; ignores the command"),
    dict(id="skate-rev4", dir="austindixson__microduck-roller-improver-rev4", cat="roller", vx=1.0,
         blurb="Roller Sprint remix rev4"),
    dict(id="rhod", dir="RhodWeo__microduck-rhod-roller", cat="roller", vx=1.0,
         blurb="RhodWeo's ES-tuned roller sprinter"),
    dict(id="arena-skate", dir="namenu__microduck-arena", cat="roller", vx=1.0,
         blurb="namenu's Arena roller (PPO + ES)"),
    # --- tricks (episodic) ---
    dict(id="roulade", featured=True, dir="pollen-robotics__microduck-policies__roulade", cat="trick", expect="roll",
         duration=1.8,
         blurb="Pollen's forward roll"),
    dict(id="kick-left", dir="pollen-robotics__microduck-policies__ball_kick_left", cat="trick", expect="kick_left",
         blurb="Pollen's left-foot ball kick (no ball in the sim)"),
    dict(id="kick", featured=True, dir="pollen-robotics__microduck-policies__ball_kick_right", cat="trick", expect="kick_right",
         blurb="Pollen's right-foot ball kick (no ball in the sim)"),
    dict(id="pick", dir="pollen-robotics__microduck-policies__alpha_ground_pick", cat="trick", expect="bow",
         phase=(4.0, 0.7), blurb="Pollen's ground pick (beak to the floor)"),
    dict(id="crouch", dir="pollen-robotics__microduck-policies__roller_crouch", cat="trick", expect="crouch",
         variant="rollers", phase=(5.0, 0.7), blurb="Pollen's roller crouch (on skates)"),
    dict(id="flip", dir="witcheer__microduck-roulade", cat="trick", expect="roll", duration=3.0,
         blurb="witcheer's forward roll"),
    dict(id="kick-w", dir="witcheer__microduck-kick-right", cat="trick", expect="kick_right",
         blurb="witcheer's right-foot kick"),
    dict(id="cartwheel", featured=True, dir="guang384__microduck-side-roll-left", cat="trick", expect="roll",
         duration=3.5,
         blurb="guang384's lateral roll over the shoulder"),
    dict(id="hop", featured=True, dir="joanfox__microduck-happy-hop", cat="trick", expect="hop", act_delay=1,
         blurb="joanfox's happy two-foot hop"),
    dict(id="leap", dir="chelleboyer__microduck-hop-once", cat="trick", expect="hop",
         blurb="chelleboyer's forward hop"),
    dict(id="groove", featured=True, dir="kramp__microduck-groove", cat="trick", expect="dance",
         blurb="kramp's knee-bounce groove"),
    dict(id="bow", dir="neil-jo__microduck-polite-bow", cat="trick", expect="bow",
         blurb="neil-jo's polite bow"),
    dict(id="playdead", featured=True, dir="drfonz-indigo__microduck-playdead", cat="trick", expect="liedown",
         blurb="drfonz-indigo's bang-you're-dead flop"),
    dict(id="chatter", dir="microduckapp__microduck-chatterbox", cat="trick", expect="dance",
         blurb="microduckapp's chatterbox babble (LSTM)"),
    dict(id="hum", dir="microduckapp__microduck-humming-along", cat="trick", expect="dance",
         blurb="microduckapp's humming sway (LSTM)"),
    dict(id="flamingo", featured=True, dir="RemiFabre__microduck-flamingo-cycle", cat="trick", expect="lift_foot",
         duration=7.0, cmd_seq=[(0.0, [1.0, 1.0, 0.0]), (5.0, [0.0, 1.0, 0.0])],
         blurb="RemiFabre's one-foot flamingo stand"),
    dict(id="dash", featured=True, dir="RemiFabre__microduck-sprint-fixed", cat="trick", expect="dash",
         duration=1.5, cmd=[0.8, 0.0, 0.0], act_delay=1, jv_lag=1, travel=2.0,
         blurb="RemiFabre's 1.7 m/s Arena sprinter as a 1.5 s dash; ends in a tumble (velstand gets up)"),
]
SPEC_BY_ID = {s["id"]: s for s in SPECS}


def _read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def manifest_entry(d):
    """The policy's manifest entry (single-policy repo or its entry in the official set)."""
    info = _read_json(os.path.join(d, "hf_info.json"), {}) or {}
    m = _read_json(os.path.join(d, "manifest.json"), {}) or {}
    if isinstance(m.get("policies"), list):
        top = {k: v for k, v in m.items() if k != "policies"}
        for e in m["policies"]:
            if e.get("file") == info.get("file"):
                return {**top, **e}
        return top
    return m


def build_entry(spec, **over):
    """Spec + manifest + HF info -> runtime catalogue entry (plain JSON)."""
    d = os.path.join(BRAIN_DIR, spec["dir"])
    m = manifest_entry(d)
    info = _read_json(os.path.join(d, "hf_info.json"), {}) or {}
    cat = spec["cat"]
    kind = "episodic" if cat == "trick" else "perpetual"
    cmd = m.get("command") or {}
    idle = cmd.get("idle") or [0.0, 0.0, 0.0]
    if cat == "roller":
        variant = "rollers"
    else:
        variant = spec.get("variant", "default")
    if m.get("mode") == "roller":
        variant = "rollers"
    duration = spec.get("duration") or m.get("duration_s")
    if spec.get("phase"):
        duration = spec["phase"][0] * spec["phase"][1]
    e = {
        "id": spec["id"],
        "repo": info.get("id") or spec["dir"].replace("__", "/", 1),
        "file": info.get("file", "policy.onnx"),
        "dir": spec["dir"],
        "kind": kind,
        "category": cat,
        "variant": variant,
        "duration_s": float(min(duration, TRICK_MAX_S)) if kind == "episodic" and duration else None,
        "action_scale": float(spec.get("action_scale") or m.get("action_scale") or 1.0),
        "vx": float(spec.get("vx", 0.0)),
        "cmd": [float(x) for x in (spec.get("cmd") or list(idle))[:3]],
        "cmd_seq": spec.get("cmd_seq"),
        "phase": list(spec["phase"]) if spec.get("phase") else None,
        "jv_lag": int(spec.get("jv_lag", 0)),
        "act_delay": int(spec.get("act_delay", 0)),
        "recurrent": int(m.get("model_api", 1)) == 2,
        "expect": spec.get("expect"),
        "featured": bool(spec.get("featured")),
        "travel_m": spec.get("travel"),
        "name": m.get("name") or spec["id"],
        "description": (spec.get("blurb") or "")[:160],
        "manifest_description": (m.get("description") or "")[:300],
        "author": info.get("author") or spec["dir"].split("__")[0],
        "license": info.get("license") or "unspecified on the HF card",
        "hf_sha": info.get("sha"),
    }
    e.update(over)
    return e


# ----------------------------------------------------------------------------------------- brain
class Brain:
    """One loaded ONNX brain (feed-forward or LSTM) + its command semantics."""

    def __init__(self, entry):
        self.e = entry
        self.id = entry["id"]
        self.kind = entry["kind"]
        self.variant = entry["variant"]
        self.scale = float(entry.get("action_scale", 1.0))
        self.duration = entry.get("duration_s")
        self.jv_lag = int(entry.get("jv_lag", 0))
        self.act_delay = int(entry.get("act_delay", 0))
        path = os.path.join(BRAIN_DIR, entry["dir"], "policy.onnx")
        self.sess = ds._make_session(path)
        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = [o.name for o in self.sess.get_outputs()]
        self.recurrent = {"h_in", "c_in"} <= set(ins) and len(ins) == 3
        if self.recurrent:
            self.obs_name = (set(ins) - {"h_in", "c_in"}).pop()
            hs = ins["h_in"].shape
            self.h_shape = (int(hs[0]), 1, int(hs[2]))
            self.out_idx = outs.index("actions") if "actions" in outs else 0
            self.h_idx, self.c_idx = outs.index("h_out"), outs.index("c_out")
        else:
            if len(ins) != 1:
                raise ValueError(f"{self.id}: unsupported inputs {list(ins)}")
            self.obs_name = next(iter(ins))
            self.out_idx = 0
        if ins[self.obs_name].shape[-1] != 61:
            raise ValueError(f"{self.id}: obs input {ins[self.obs_name].shape}")
        self.broken = None
        self.reset()

    def reset(self):
        if self.recurrent:
            self.h = np.zeros(self.h_shape, dtype=np.float32)
            self.c = np.zeros(self.h_shape, dtype=np.float32)

    def act(self, obs):
        feed = {self.obs_name: obs.reshape(1, -1).astype(np.float32)}
        if self.recurrent:
            feed["h_in"], feed["c_in"] = self.h, self.c
        try:
            res = self.sess.run(None, feed)
        except Exception as ex:  # noqa: BLE001  (untrusted graph: never let it kill the sim thread)
            self.broken = repr(ex)[:200]
            self.reset()
            return np.zeros(14, dtype=np.float32)
        a = np.asarray(res[self.out_idx], dtype=np.float32).reshape(-1)
        if a.shape[0] != 14 or not np.all(np.isfinite(a)):
            self.broken = f"bad action {a.shape}"
            self.reset()
            return np.zeros(14, dtype=np.float32)
        if self.recurrent:
            self.h, self.c = res[self.h_idx].astype(np.float32), res[self.c_idx].astype(np.float32)
        return a

    def trick_command(self, el):
        cmd = np.zeros(13, dtype=np.float32)
        e = self.e
        if e.get("phase"):
            period = e["phase"][0]
            ph = 2 * math.pi * el / period
            cmd[0], cmd[1] = math.cos(ph), math.sin(ph)
        elif e.get("cmd_seq"):
            cur = e["cmd_seq"][0][1]
            for t0, c in e["cmd_seq"]:
                if el >= t0:
                    cur = c
            cmd[0:3] = cur
        else:
            cmd[0:3] = e.get("cmd") or [0, 0, 0]
        return cmd


# ----------------------------------------------------------------------------------------- catalogue
def load_catalogue(path=CATALOGUE_PATH, include_failed=False):
    """id -> entry for the featured brains the evaluator kept (status pass/wobbly)."""
    data = _read_json(path, {}) or {}
    out = {}
    for e in data.get("brains", []):
        featured = e.get("featured") or os.environ.get("MICRODUCK_BRAINS") == "all"
        if include_failed or (e.get("status") in ("pass", "wobbly") and featured):
            out[e["id"]] = e
    return out


# ----------------------------------------------------------------------------------------- sim
class BrainSim(ds.DuckSim):
    """DuckSim + hot-swappable brains. All public methods must be called with self.lock held (as DuckSim)."""

    def __init__(self, *a, catalogue=None, **kw):
        self._bs_ready = False
        super().__init__(*a, **kw)
        self.catalogue = load_catalogue() if catalogue is None else dict(catalogue)
        self._brains = {}
        self.variant = "default"
        self.slot = "default"
        self.trick = None
        self.trick_t0 = 0.0
        self.trick_queue_mode = "stand"
        self.tricks_done = 0
        self.brain_switches = 0
        self.model_gen = 0
        self._models = {}            # (variant, terrain id) -> (model, data, bam)
        self._spawn_z = {}
        self._jv_prev = None
        self._jv_prev_step = -2
        self._tgt_prev = None
        self._render_cache = (None, None)
        self._bound = self.model
        self._default_model = (self.model, self.data, self.bam)
        self._bs_ready = True

    # ------------------------------------------------------------ model variants
    def _rebind(self):
        """Recompute model indices (same code as DuckSim.__init__) after a model swap."""
        m = self.model
        self.imu_gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.fq = int(m.jnt_qposadr[fj])
        self.fv = int(m.jnt_dofadr[fj])
        self.jq = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.jv = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        assert m.nu == 14
        self.backlash_j = [j for j in range(m.njnt)
                           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_backlash")]
        self.backlash_range = m.jnt_range[self.backlash_j].copy() if self.backlash_j else None
        self._backlash_locked = False
        self._bound = m
        self.model_gen += 1

    def _scene_path(self, variant):
        rel = SCENES[variant]
        return DEFAULT_SCENE if rel is None else os.path.join(ds.ROBOT_DIR, rel)

    def _load_variant(self, variant):
        terrain = getattr(self, "terrain", None)
        key = (variant, id(terrain))
        if key not in self._models:
            # explicit scene (no global swap): several ducks (multi.py) may wear different robots
            built = ds.load_model(self.use_bam, terrain=terrain, scene=self._scene_path(variant))
            # keep at most the default + one other variant per terrain
            self._models = {k: v for k, v in self._models.items() if k[1] == id(terrain)}
            self._models[key] = built
        return self._models[key]

    def _spawn_height(self, variant):
        """Free-joint z that puts the lowest collision geom ~1 mm above a flat floor in the default pose."""
        if variant == "default":
            return ds.SPAWN_Z
        if variant not in self._spawn_z:
            m = mujoco.MjModel.from_xml_path(self._scene_path(variant))
            d = mujoco.MjData(m)
            fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
            fq = int(m.jnt_qposadr[fj])
            jq = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
            z = 0.25
            while z > 0.03:
                d.qpos[:] = 0
                d.qpos[fq:fq + 7] = [0, 0, z, 1, 0, 0, 0]
                d.qpos[jq] = ds.DEFAULT_POSE
                mujoco.mj_forward(m, d)
                if any(d.contact[i].dist < 0.001 for i in range(d.ncon)):
                    break
                z -= 0.001
            self._spawn_z[variant] = z + 0.002
        return self._spawn_z[variant]

    def _switch_variant(self, variant, respawn=True):
        """Swap the MjModel (skates on/off). Respawns standing at the same x/y/yaw."""
        if variant == self.variant and self.model is self._bound:
            return
        p = self.pos().copy()
        yaw = self.yaw()
        if variant == "default" and getattr(self, "terrain", None) is None and self._default_model is not None:
            built = self._default_model
        else:
            built = self._load_variant(variant)
        self.model, self.data, self.bam = built
        self.variant = variant                     # terrain.build() reads self.variant to rebuild the right robot
        self._rebind()
        if respawn:
            self._respawn(float(p[0]), float(p[1]), yaw)
        self.log(f"[brains] t={self.t:.2f} robot -> {variant} at ({p[0]:.2f},{p[1]:.2f}) yaw={math.degrees(yaw):.0f}")

    def _respawn(self, x, y, yaw):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        gh = 0.0
        if terrain_mod is not None and (getattr(self, "terrain", None) is not None
                                        or getattr(self, "spawn_override", None) is not None):
            try:
                x, y, gh = terrain_mod.spawn_point(self, x, y, 0.0)
            except Exception:  # noqa: BLE001
                gh = 0.0
        d.qpos[self.fq:self.fq + 3] = [x, y, gh + self._spawn_height(self.variant)]
        d.qpos[self.fq + 3:self.fq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, a in enumerate(self.jq):
            d.qpos[a] = ds.DEFAULT_POSE[i]
        if self.bam is not None:
            self.bam.last_ts = d.time
            self.bam.q_target[:] = ds.DEFAULT_POSE
        else:
            d.ctrl[:] = ds.DEFAULT_POSE
        mujoco.mj_forward(m, d)
        self.last_action = np.zeros(14, dtype=np.float32)
        self._tgt_prev = None
        self._jv_prev = None
        self.fallen = False
        self.fallen_since = None

    def render_ids(self, names, default_ids):
        """Body ids (in the current model) for the model.json body names (server /pose)."""
        if self.variant == "default":
            return default_ids
        key, ids = self._render_cache
        if key is self.model:
            return ids
        ids = []
        for n, did in zip(names, default_ids):
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BODY_ALIASES.get(n, n))
            ids.append(b if b >= 0 else self.trunk_id)
        self._render_cache = (self.model, ids)
        return ids

    # ------------------------------------------------------------ state
    def reset(self):
        if getattr(self, "_bs_ready", False):
            self.trick = None
            if self.slot != "default":
                self.log(f"[brains] t={self.t:.2f} reset -> default brain (was {self.slot})")
            self.slot = "default"
            if self.variant != "default":
                self._switch_variant("default", respawn=False)
        super().reset()
        if getattr(self, "_bs_ready", False):
            self._tgt_prev = None
            self._jv_prev = None

    def brain(self, bid):
        b = self._brains.get(bid)
        if b is None:
            b = Brain(self.catalogue[bid])
            self._brains[bid] = b
        return b

    def brain_info(self):
        return {"brain": self.slot, "trick": self.trick.id if self.trick else None, "robot": self.variant}

    def catalogue_public(self):
        keys = ("id", "kind", "category", "variant", "duration_s", "description", "repo", "author", "license",
                "status")
        return {"brain": self.slot, "trick": self.trick.id if self.trick else None, "robot": self.variant,
                "brains": [{k: e.get(k) for k in keys} | {"eval": e.get("eval_summary")}
                           for e in sorted(self.catalogue.values(), key=lambda e: (e["kind"], e["id"]))]}

    def public_mode(self):
        if self.trick is not None:
            return "trick"
        return super().public_mode()

    # ------------------------------------------------------------ commands
    def command(self, cmd, **kw):
        name = kw.pop("name", None)
        if cmd == "brain":
            return self.set_brain(name)
        if cmd == "trick":
            return self.start_trick(name)
        if cmd in ("quack", "talk") or cmd not in ("walk", "stop", "sit", "stand", "follow", "turn", "reset"):
            return super().command(cmd, **kw)
        if cmd == "reset":
            return super().command(cmd, **kw)
        if self.trick is not None:          # any motion command cancels a trick
            if cmd == "sit":
                left = max(0.0, (self.trick.duration or 0.0) - (self.t - self.trick_t0))
                return False, f"doing trick {self.trick.id}; sit in {left:.1f} s"
            self._end_trick(f"cancelled by {cmd}")
        if self.variant != "default":
            if cmd == "walk":                # keep skating
                self._enter("walk")
                return True, None
            self._skates_off(f"cmd {cmd}")
        return super().command(cmd, **kw)

    def set_brain(self, name):
        name = (name or "").strip().lower()
        if not name:
            return False, "brain needs a name (see GET /brains)"
        if name in ("default", "velstand", "normal"):
            if self.trick is not None:
                self._end_trick("brain default")
            if self.variant != "default":
                self._skates_off("brain default")
            changed = self.slot != "default"
            self.slot = "default"
            if changed:
                self.brain_switches += 1
                self.log(f"[brains] t={self.t:.2f} brain -> default")
            return True, None
        e = self.catalogue.get(name)
        if e is None:
            return False, f"unknown brain {name!r} (see GET /brains)"
        if e["kind"] != "perpetual":
            return False, f"{name!r} is a trick: use cmd trick"
        if self.fallen:
            return False, "duck is fallen; wait for it to get up"
        try:
            self.brain(name)
        except Exception as ex:  # noqa: BLE001
            return False, f"brain {name!r} failed to load: {ex}"
        if self.trick is not None:
            self._end_trick(f"brain {name}")
        self.slot = name
        self.brain_switches += 1
        self.brain(name).reset()
        self.log(f"[brains] t={self.t:.2f} brain -> {name} ({e['repo']}, robot {e['variant']})")
        if e["variant"] != self.variant:
            self._switch_variant(e["variant"])
            self.mode = "stand"
            self.pending = None
            self.stand_after_settle = False
            self.policy = "velstand"
        if e["variant"] == "rollers":
            self._enter("walk")
            return True, None
        return super().command("walk")

    def start_trick(self, name):
        name = (name or "").strip().lower()
        e = self.catalogue.get(name)
        if e is None:
            return False, f"unknown trick {name!r} (see GET /brains)"
        if e["kind"] != "episodic":
            return False, f"{name!r} is a brain, not a trick: use cmd brain"
        if self.fallen:
            return False, "duck is fallen; wait for it to get up"
        if self.mode in ("sit", "sitting_down", "standing_up") or self.policy == "sitstand":
            return False, "duck is sitting; stand up first"
        if e["variant"] == "rollers" and self.variant != "rollers":
            return False, f"{name!r} needs roller skates: pick a roller brain first"
        travel = float(e.get("travel_m") or (e.get("eval_summary") or {}).get("dist") or 0.0)
        if travel > 0.25:   # tricks that travel: keep the landing inside the leash
            p = self.pos()
            yaw = self.yaw()
            ax, ay = p[0] + travel * math.cos(yaw), p[1] + travel * math.sin(yaw)
            if math.hypot(ax, ay) > self.leash - 0.15:
                return False, f"no room for {name!r} ({travel:.1f} m ahead would cross the leash): turn the duck first"
        try:
            b = self.brain(name)
        except Exception as ex:  # noqa: BLE001
            return False, f"trick {name!r} failed to load: {ex}"
        if e["variant"] != "rollers" and self.variant == "rollers":
            self._skates_off(f"trick {name}")
        if self.trick is not None:
            self._end_trick(f"trick {name}")
        b.reset()
        self.trick = b
        self.trick_t0 = self.t
        self.trick_queue_mode = "walk" if (self.variant == "rollers") else "stand"
        self.returning = False
        self.log(f"[brains] t={self.t:.2f} trick {name} ({e['duration_s']} s, {e['repo']})")
        return True, None

    def _end_trick(self, why):
        b = self.trick
        self.trick = None
        self.tricks_done += 1
        self.mode = self.trick_queue_mode
        self.policy = "velstand"
        self.returning = False
        if self.mode == "walk":
            self._enter("walk")
        self.log(f"[brains] t={self.t:.2f} trick {b.id} done ({why})")

    def _skates_off(self, why):
        self.log(f"[brains] t={self.t:.2f} skates off ({why})")
        if self.slot != "default" and self.catalogue.get(self.slot, {}).get("variant") == "rollers":
            self.slot = "default"
        self._switch_variant("default")
        self.mode = "stand"
        self.policy = "velstand"

    # ------------------------------------------------------------ stepping
    def _bobs(self, command, lag):
        d = self.data
        ang = d.sensordata[self.imu_gyro_adr:self.imu_gyro_adr + 3].astype(np.float32)
        jp = d.qpos[self.jq].astype(np.float32) - ds.DEFAULT_POSE
        jv = d.qvel[self.jv].astype(np.float32)
        if lag:
            prev = self._jv_prev if self._jv_prev_step == self.steps - 1 else None
            self._jv_prev, self._jv_prev_step = jv, self.steps
            jv = jv if prev is None else prev
        else:
            self._jv_prev, self._jv_prev_step = jv, self.steps
        return np.concatenate([ang, self.proj_gravity(), jp, jv, self.last_action, command]).astype(np.float32)

    def _physics(self, target):
        if self.bam is not None:
            self.bam.q_target[:] = target
        else:
            self.data.ctrl[:] = target
        for _ in range(ds.DECIMATION):
            if self.bam is not None:
                self.bam.update()
            mujoco.mj_step(self.model, self.data)
        self.t += ds.CONTROL_DT
        self.steps += 1

    def _brain_step(self, b, command, check_fall=True):
        if self.backlash_j and self._backlash_locked:
            self.model.jnt_range[self.backlash_j] = self.backlash_range
            self._backlash_locked = False
        obs = self._bobs(command, b.jv_lag)
        a = b.act(obs)
        if b.broken:
            self.log(f"[brains] t={self.t:.2f} {b.id} broke ({b.broken}) -> removed, default brain")
            self.catalogue.pop(b.id, None)
            self._brains.pop(b.id, None)
            if self.trick is b:
                self.trick = None
            self.slot = "default"
            if self.variant != "default":
                self._switch_variant("default")
            self.mode = "stand"
            self.policy = "velstand"
            return
        self.last_action = a.copy()
        tgt = ds.DEFAULT_POSE + a * b.scale
        if b.act_delay:
            tgt, self._tgt_prev = (self._tgt_prev if self._tgt_prev is not None else tgt), tgt
        self._physics(tgt)
        if check_fall:
            n = self.falls
            self._update_fall()
            if self.falls > n and self.slot != "default" and self.variant == "default":
                self.log(f"[brains] t={self.t:.2f} {b.id} fell -> default brain (velstand gets up)")
                self.slot = "default"
                self.mode = "stand"
                self.policy = "velstand"

    def _leash_out(self, speed_pad=0.45):
        p = self.pos()
        r = math.hypot(p[0], p[1])
        v = self.data.qvel[self.fv:self.fv + 2]
        speed = math.hypot(v[0], v[1])
        outward = (p[0] * v[0] + p[1] * v[1]) > 0
        margin = min(ds.LEASH_MARGIN, 0.3 * self.leash) + speed_pad * speed
        return outward and r > self.leash - margin

    def _blocked_ahead(self):
        if terrain_mod is None or getattr(self, "terrain", None) is None:
            return False
        try:
            return terrain_mod.clearance(self, self.yaw()) < getattr(terrain_mod, "AV_SLOW", 0.26)
        except Exception:  # noqa: BLE001
            return False

    def step(self):
        if self.model is not self._bound:   # model swapped from outside (terrain rebuild): same robot, new ids
            self._rebind()
        if self.trick is not None:
            el = self.t - self.trick_t0
            if el >= (self.trick.duration or 0.0):
                self._end_trick("finished")
            else:
                self._brain_step(self.trick, self.trick.trick_command(el), check_fall=False)
                self.fallen = False      # rolls/flops are the point; fall logic resumes when it hands back
                self.fallen_since = None
                return
        if self.variant == "rollers":
            return self._roller_step()
        if (self.slot != "default" and self.mode == "walk" and self.policy == "velstand"
                and not self.fallen and not self.returning):
            b = self.brain(self.slot)
            if self._leash_out() or self._blocked_ahead():
                self.returning = True    # velstand walks it back inside the leash, then the brain resumes
                self.walk_heading = self.yaw()
            else:
                cmd = np.zeros(13, dtype=np.float32)
                cmd[0] = b.e.get("vx", 0.0)
                cmd[1:3] = (b.e.get("cmd") or [0, 0, 0])[1:3]
                self._brain_step(b, cmd)
                return
        super().step()

    def _roller_step(self):
        b = self.brain(self.slot) if self.slot != "default" else None
        if b is None or self.fallen:
            # nothing can drive the skates (or it fell): hold the default pose; auto-reset puts the shoes back on
            self._physics(ds.DEFAULT_POSE)
            n = self.falls
            self._update_fall()
            if b is None and not self.fallen and self.falls == n:
                self._skates_off("no roller brain")
            return
        cmd = np.zeros(13, dtype=np.float32)
        p = self.pos()
        r = math.hypot(p[0], p[1])
        if self.mode == "walk":
            cmd[0] = b.e.get("vx", 0.4)
            # steer back toward the origin when outside half the leash (heading-error slot, trained by the
            # roller recipes; command-blind brains skate straight and are handed back at the leash)
            if r > 0.5 * self.leash:
                err = ds.wrap(math.atan2(-p[1], -p[0]) - self.yaw())
                cmd[2] = float(np.clip(ROLLER_STEER_GAIN * err, -1.0, 1.0))
        if self._leash_out(speed_pad=0.6) and r > self.leash - 0.2:
            self._skates_off("leash")
            self.slot = "default"
            self._enter("walk")
            self.returning = True
            self.walk_heading = self.yaw()
            return
        self._brain_step(b, cmd)


# ----------------------------------------------------------------------------------------- evaluator
class Probe:
    """Per-step metrics while something runs."""

    def __init__(self, sim):
        self.sim = sim
        self.p0 = sim.pos().copy()
        self.yaw0 = sim.yaw()
        self.max_tilt = 0.0
        self.min_z = 9.0
        self.max_z = 0.0
        self.z0 = float(sim.pos()[2])
        self.inverted = False
        self.air = 0.0
        self.max_air = 0.0
        self.foot_speed = {"ankle_left": 0.0, "ankle_right": 0.0}
        self.foot_lift = {"ankle_left": 0.0, "ankle_right": 0.0}
        self.lift_time = 0.0
        self.jstd = []
        self.max_r = 0.0
        self.n = 0
        self._foot_ids = {}
        self._foot_prev = {}
        self._foot_z0 = {}

    def _foot(self, n):
        s = self.sim
        key = (n, id(s.model))
        if key not in self._foot_ids:
            b = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                b = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, BODY_ALIASES.get(n, n))
            self._foot_ids[key] = b
        return self._foot_ids[key]

    def tick(self):
        s = self.sim
        g = s.proj_gravity()
        tilt = math.degrees(math.acos(float(np.clip(-g[2], -1, 1))))
        self.max_tilt = max(self.max_tilt, tilt)
        if g[2] > 0.5:
            self.inverted = True
        p = s.pos()
        self.min_z = min(self.min_z, float(p[2]))
        self.max_z = max(self.max_z, float(p[2]))
        self.max_r = max(self.max_r, math.hypot(p[0], p[1]))
        floor_contact = False
        for i in range(s.data.ncon):
            c = s.data.contact[i]
            if s.model.geom_bodyid[c.geom1] == 0 or s.model.geom_bodyid[c.geom2] == 0:
                floor_contact = True
                break
        if floor_contact:
            self.air = 0.0
        else:
            self.air += ds.CONTROL_DT
            self.max_air = max(self.max_air, self.air)
        lifted = False
        for n in self.foot_speed:
            b = self._foot(n)
            x = s.data.xpos[b].copy()
            if n in self._foot_prev:
                v = np.linalg.norm(x - self._foot_prev[n]) / ds.CONTROL_DT
                self.foot_speed[n] = max(self.foot_speed[n], float(v))
            self._foot_prev[n] = x
            self._foot_z0.setdefault(n, float(x[2]))
            lift = float(x[2]) - self._foot_z0[n]
            self.foot_lift[n] = max(self.foot_lift[n], lift)
            if lift > 0.03:
                lifted = True
        if lifted and tilt < 45:
            self.lift_time += ds.CONTROL_DT
        self.jstd.append(s.data.qpos[s.jq].copy())
        self.n += 1

    def summary(self):
        s = self.sim
        p = s.pos()
        dxy = p[:2] - self.p0[:2]
        js = float(np.mean(np.std(np.array(self.jstd), axis=0))) if self.jstd else 0.0
        return {"dist": round(float(np.linalg.norm(dxy)), 3),
                "dx_heading": round(float(dxy[0] * math.cos(self.yaw0) + dxy[1] * math.sin(self.yaw0)), 3),
                "yaw_change_deg": round(math.degrees(ds.wrap(s.yaw() - self.yaw0)), 1),
                "max_tilt_deg": round(self.max_tilt, 1), "inverted": self.inverted,
                "min_z": round(self.min_z, 3), "rise": round(self.max_z - self.z0, 3),
                "max_air_s": round(self.max_air, 3),
                "foot_speed": {k[6:]: round(v, 2) for k, v in self.foot_speed.items()},
                "foot_lift": {k[6:]: round(v, 3) for k, v in self.foot_lift.items()},
                "lift_time_s": round(self.lift_time, 2),
                "joint_std": round(js, 3), "max_r": round(self.max_r, 2)}


def run_for(sim, seconds, probe=None):
    for _ in range(int(round(seconds / ds.CONTROL_DT))):
        sim.step()
        if probe is not None:
            probe.tick()


def perturb(sim, rng):
    d = sim.data
    for a in sim.jq:
        d.qpos[a] += rng.uniform(-0.01, 0.01)
    yaw = rng.uniform(-0.035, 0.035)
    d.qpos[sim.fq + 3:sim.fq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(sim.model, d)


def upright(sim):
    return (not sim.fallen) and float(sim.proj_gravity()[2]) < -0.8


def eval_once(sim, entry, seed, quick=False):
    """One trial of one brain config; returns (success, did_thing, metrics)."""
    rng = np.random.default_rng(seed)
    sim.catalogue = {entry["id"]: entry}
    sim._brains.pop(entry["id"], None)
    sim.leash = 100.0
    sim.command("reset")
    perturb(sim, rng)
    falls0, resets0 = sim.falls, sim.resets
    if entry["kind"] == "perpetual":
        if entry["variant"] != "rollers":
            run_for(sim, 1.0)                      # settle on velstand, like the Arena's untimed settle
        ok, err = sim.command("brain", name=entry["id"])
        if not ok:
            return False, False, {"error": err}
        if entry["variant"] == "rollers":
            run_for(sim, 0.3)
        pr = Probe(sim)
        T = 6.0
        run_for(sim, T, pr)
        m = pr.summary()
        m["speed"] = round(m["dist"] / T, 3)
        m["fell"] = sim.falls > falls0 or sim.resets > resets0 or sim.slot == "default"
        if entry["variant"] == "rollers" and not m["fell"]:
            # steering probe: does the heading-error slot turn it?  (target 90 deg left of current heading)
            y0 = sim.yaw()
            b = sim.brain(entry["id"])
            for _ in range(int(3.0 / ds.CONTROL_DT)):
                cmd = np.zeros(13, dtype=np.float32)
                cmd[0] = entry["vx"]
                cmd[2] = float(np.clip(ds.wrap(y0 + math.pi / 2 - sim.yaw()), -1, 1))
                sim._brain_step(b, cmd)
            m["steer_deg"] = round(math.degrees(ds.wrap(sim.yaw() - y0)), 1)
            m["fell"] = m["fell"] or sim.fallen or sim.slot == "default"
        # hand back: stop -> velstand stands 3 s
        sim.command("stop")
        run_for(sim, 3.0)
        m["handback_upright"] = upright(sim)
        did = (m["dx_heading"] > 0.3) if entry["variant"] != "rollers" else (m["dist"] > 0.5)
        if not quick and did:
            # runtime test: 20 s with the default leash (brain <-> velstand handovers / roller steering)
            sim.command("reset")
            sim.leash = 2.5
            f0, r0 = sim.falls, sim.resets
            if entry["variant"] != "rollers":
                run_for(sim, 1.0)
            sim.command("brain", name=entry["id"])
            pl = Probe(sim)
            kept = 0
            for _ in range(int(20.0 / ds.CONTROL_DT)):
                sim.step()
                pl.tick()
                kept += sim.slot == entry["id"]
            m["leash_falls"] = sim.falls - f0 + (sim.resets - r0)
            m["leash_max_r"] = round(pl.max_r, 2)
            m["leash_brain_kept"] = round(kept / (20.0 / ds.CONTROL_DT), 2)
            m["leash_ok"] = m["leash_falls"] == 0 and pl.max_r < 2.5 + 0.3 and upright(sim)
        success = (not m["fell"]) and m["handback_upright"] and did and (quick or m.get("leash_ok", False))
        if entry["variant"] == "rollers" and not quick:
            success = success and m.get("leash_brain_kept", 0) > 0.9
        return success, did, m
    # ---- trick
    if entry["variant"] == "rollers":
        return eval_roller_trick(sim, entry, seed)
    run_for(sim, 1.0)
    ok, err = sim.command("trick", name=entry["id"])
    if not ok:
        return False, False, {"error": err}
    pr = Probe(sim)
    T = entry["duration_s"]
    run_for(sim, T, pr)
    m = pr.summary()
    g = sim.proj_gravity()
    m["end_tilt_deg"] = round(math.degrees(math.acos(float(np.clip(-g[2], -1, 1)))), 1)
    m["upright_at_end"] = bool(g[2] < -0.8)
    f_end = sim.falls
    run_for(sim, 5.0)
    m["falls_after"] = sim.falls - f_end
    m["upright_after"] = upright(sim)
    m["resets"] = sim.resets - resets0
    did = thing_done(entry.get("expect"), m)
    expect = entry.get("expect")
    lands = (m["upright_at_end"] and m["falls_after"] == 0) or expect in ("liedown",)
    success = did and m["upright_after"] and m["resets"] == 0
    m["clean"] = bool(lands)
    return success, did, m


def eval_roller_trick(sim, entry, seed):
    # needs a roller brain to be on: use the best roller in the catalogue file, else the official one
    rid = "roller"
    rentry = build_entry(SPEC_BY_ID[rid])
    sim.catalogue[rid] = rentry
    sim.command("brain", name=rid)
    run_for(sim, 0.5)
    ok, err = sim.command("trick", name=entry["id"])
    if not ok:
        return False, False, {"error": err}
    pr = Probe(sim)
    run_for(sim, entry["duration_s"], pr)
    m = pr.summary()
    m["upright_at_end"] = bool(sim.proj_gravity()[2] < -0.8)
    run_for(sim, 2.0)
    m["upright_after"] = upright(sim) and sim.variant == "rollers"
    did = thing_done(entry.get("expect"), m)
    return did and m["upright_after"], did, m


def thing_done(expect, m):
    fs, fl = m["foot_speed"], m["foot_lift"]
    if expect == "roll":
        return m["inverted"] or m["max_tilt_deg"] > 120
    if expect == "kick_right":
        return fs["right"] > 0.6 and fs["right"] > 1.3 * fs["left"] and m["max_tilt_deg"] < 60
    if expect == "kick_left":
        return fs["left"] > 0.6 and fs["left"] > 1.3 * fs["right"] and m["max_tilt_deg"] < 60
    if expect == "hop":
        return m["max_air_s"] >= 0.04 or m["rise"] > 0.015
    if expect == "bow":
        return 12 < m["max_tilt_deg"] < 100
    if expect == "liedown":
        return m["end_tilt_deg"] > 75
    if expect == "dance":
        return m["joint_std"] > 0.03 and m["max_tilt_deg"] < 50
    if expect == "lift_foot":
        return m["lift_time_s"] > 1.5
    if expect == "crouch":
        return m["min_z"] < 0.11 and m["max_tilt_deg"] < 60
    if expect == "dash":
        return m["dx_heading"] > 1.2
    return True


CONFIGS_WALK = [dict(jv_lag=lag, act_delay=dl) for dl in (1, 0) for lag in (1, 0)]
CONFIGS_TRICK = [dict(jv_lag=0, act_delay=0), dict(jv_lag=1, act_delay=0), dict(jv_lag=0, act_delay=1),
                 dict(jv_lag=1, act_delay=1)]


def evaluate(ids=None, seeds=(1, 2, 3), log=print):
    sim = BrainSim(log=lambda *a: None, catalogue={})
    out = []
    t_wall = time.perf_counter()
    for spec in SPECS:
        if ids and spec["id"] not in ids:
            continue
        base = build_entry(spec)
        if base["kind"] == "episodic":
            configs = CONFIGS_TRICK
        else:
            vxs = [base["vx"]] + [v for v in spec.get("vx_try", []) if v != base["vx"]]
            configs = [dict(c, vx=v) for v in vxs for c in CONFIGS_WALK]
        if spec.get("act_delay") is not None or spec.get("jv_lag") is not None:
            configs = [dict(jv_lag=spec.get("jv_lag", 0), act_delay=spec.get("act_delay", 0))] + configs
        best = None
        tried = []
        for cfg in configs:
            e = dict(base, **cfg)
            trials = []
            for s in seeds:
                try:
                    trials.append(eval_once(sim, e, s, quick=False))
                except Exception as ex:  # noqa: BLE001
                    trials.append((False, False, {"error": repr(ex)}))
            n_ok = sum(1 for t in trials if t[0])
            n_did = sum(1 for t in trials if t[1])
            score = (n_ok, n_did, _perf(e, trials))
            tried.append({"cfg": cfg, "ok": n_ok, "did": n_did, "perf": round(score[2], 3)})
            if best is None or score > best[0]:
                best = (score, e, trials)
            if n_ok == len(seeds) and e["kind"] == "episodic" and cfg == configs[0]:
                break      # first config already clean
        score, e, trials = best
        n_ok, n_did, perf = score
        status = "pass" if n_ok == len(seeds) else ("wobbly" if n_ok >= 2 else "fail")
        if status == "pass" and e["kind"] == "episodic" and not all(t[2].get("clean", True) for t in trials):
            status = "wobbly"   # does the thing, velstand recovers, but it doesn't land on its feet by itself
        e = dict(e)
        e["status"] = status
        e["eval"] = {"trials": [t[2] for t in trials], "ok": n_ok, "did": n_did, "n": len(seeds),
                     "configs": tried}
        e["eval_summary"] = summarize(e, trials)
        out.append(e)
        log(f"{status:6s} {e['id']:12s} {e['repo']:50s} cfg(jv_lag={e['jv_lag']},delay={e['act_delay']},vx={e['vx']}) "
            f"ok={n_ok}/{len(seeds)} did={n_did} {e['eval_summary']}  [{time.perf_counter() - t_wall:.0f}s]")
    return out


def _perf(e, trials):
    ms = [t[2] for t in trials if "error" not in t[2]]
    if not ms:
        return 0.0
    if e["kind"] == "perpetual":
        return float(np.mean([m.get("speed", 0) for m in ms]))
    return -float(np.mean([m.get("max_tilt_deg", 0) for m in ms])) / 1000.0


def summarize(e, trials):
    ms = [t[2] for t in trials if "error" not in t[2]]
    if not ms:
        return {"error": trials[0][2].get("error")}
    def mean(k):
        return round(float(np.mean([m.get(k, 0) for m in ms])), 3)
    if e["kind"] == "perpetual":
        s = {"speed_mps": mean("speed"), "dist_6s": mean("dist"), "fell": sum(bool(m.get("fell")) for m in ms),
             "yaw_drift_deg": mean("yaw_change_deg"), "max_tilt_deg": mean("max_tilt_deg")}
        if "leash_falls" in ms[0]:
            s["leash_falls_20s"] = int(sum(m.get("leash_falls", 0) for m in ms))
            s["leash_max_r"] = max(m.get("leash_max_r", 0) for m in ms)
            s["leash_brain_kept"] = min(m.get("leash_brain_kept", 0) for m in ms)
        if "steer_deg" in ms[0]:
            s["steer_deg"] = mean("steer_deg")
        return s
    s = {"max_tilt_deg": mean("max_tilt_deg"), "rise_m": mean("rise"), "air_s": mean("max_air_s"),
         "joint_std": mean("joint_std"), "dist": mean("dist"),
         "upright_after": sum(bool(m.get("upright_after")) for m in ms)}
    if e.get("expect", "").startswith("kick"):
        s["foot_speed"] = ms[0]["foot_speed"]
    if e.get("expect") == "lift_foot":
        s["lift_time_s"] = mean("lift_time_s")
    if e.get("expect") == "liedown":
        s["end_tilt_deg"] = mean("end_tilt_deg")
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["eval", "list"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--out", default=CATALOGUE_PATH)
    args = ap.parse_args()
    if args.action == "list":
        for e in load_catalogue(args.out, include_failed=True).values():
            print(f"{e['status']:6s} {e['id']:12s} {e['kind']:9s} {e['variant']:8s} {e['repo']:50s} {e.get('eval_summary')}")
        return
    res = evaluate(args.ids or None)
    old = _read_json(args.out, {}) or {}
    keep = {e["id"]: e for e in old.get("brains", [])} if args.ids else {}
    for e in res:
        keep[e["id"]] = e
    data = {"version": 1, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "written by brains.py eval; runtime offers status pass/wobbly only",
            "brains": sorted(keep.values(), key=lambda e: (e["kind"], e["category"], e["id"]))}
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {sum(e['status'] != 'fail' for e in data['brains'])}/{len(data['brains'])} kept")


if __name__ == "__main__":
    main()
