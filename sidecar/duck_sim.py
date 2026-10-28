"""Microduck MuJoCo simulation core (no HTTP).

Mirrors pollen-robotics/microduck_rl scripts/infer_policy.py defaults:
  * scene.xml (robot_groundcontact.xml), timestep 0.005 s, decimation 4 -> 50 Hz policy
  * BAM M6 XL330 actuator model (better-actuator-models), kp_fw=200, vin=7.4 V,
    vin_drop_gain=0.1 V/Nm, vin_min=6.0 V, stiff friction constraint
  * 61-D obs = ang_vel(3, imu gyro) + projected_gravity(3) + joint_pos-default(14)
               + joint_vel(14) + last_action(14) + command(13)   (--new-cmd-obs layout)
  * target = DEFAULT_POSE + action * action_scale(1.0)
"""

import math
import os
import threading
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

import terrain as terrain_mod  # noqa: E402  (Minecraft blocks -> static box geoms)
import talk as talk_mod  # noqa: E402  (speaking animation: head nod while voice audio plays)
import ghosts as ghost_mod  # noqa: E402  (multi-duck: the other ducks as mocap capsules, see multi.py)

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(HERE, "upstream", "src", "mjlab_microduck", "robot", "microduck")
SCENE_XML = os.path.join(ROBOT_DIR, os.environ.get("MICRODUCK_SCENE", "scene_sim_allcollisions_backlash.xml"))
JOINT_VEL_LAG = int(os.environ.get("MICRODUCK_JV_LAG", "0"))
POLICY_DIR = os.path.join(HERE, "policies")

TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = TIMESTEP * DECIMATION  # 0.02 s -> 50 Hz
ACTION_SCALE = 1.0

# BAM constants (infer_policy.py)
BAM_KP_FW = 200.0
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,      # left leg
    0.3491, 0.3491, 0.0, 0.0,                     # neck_pitch, head_pitch, head_yaw, head_roll
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,         # right leg
], dtype=np.float32)

SPAWN_Z = 0.125

# behaviour constants
WALK_VX = 0.15
TURN_WZ = 1.0             # rad/s commanded for "turn" (velstand v8 tracks yaw)
TURN_TIMEOUT = 8.0
FOLLOW_STOP = 0.4
FOLLOW_VX = 0.3          # policy-command units (achieves ~0.1 m/s in sim)
LEASH_DEFAULT = 2.5
LEASH_MARGIN = 0.45       # start steering back this far inside the leash
FALL_RESET_S = 3.0
SIT_SETTLE_S = 2.0        # manifest ramp_s
STAND_UNWIND_S = 1.0      # manifest unwind_s (the rise); we keep sitstand a bit longer
STAND_HANDBACK_S = 2.5
SPEED_GAIN = float(os.environ.get("MICRODUCK_SPEED_GAIN", "2.667"))
KICK_S = 0.8
KICK_V = 0.3
RISE_WITH = os.environ.get("MICRODUCK_RISE_WITH", "sitstand")

# multi-duck (multi.py): number of ghost mocap bodies added to every model (0 = single duck, no ghosts)
GHOSTS = int(os.environ.get("MICRODUCK_GHOSTS", "0"))

MODES = ("stand", "walk", "follow", "turn", "sit", "sitting_down", "standing_up", "fallen")


_SESSIONS = {}


def _make_session(path):
    """onnxruntime session, shared per file: several ducks (multi.py) use the same velstand/brain graphs. Sessions
    are stateless (LSTM state is passed in and out by Brain), and the sim thread is the only caller."""
    key = os.path.realpath(path)
    s = _SESSIONS.get(key)
    if s is None:
        s = _SESSIONS[key] = _new_session(path)
    return s


def _new_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def quat_rotate_inverse(q, v):
    w = q[0]
    xyz = q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_model(use_bam=True, terrain=None, scene=None):
    """Load scene; with BAM, mirror infer_policy.load_mujoco_with_bam. `terrain`: terrain.Terrain boxes to add.
    `scene`: MJCF path (default SCENE_XML); pass it explicitly from threads (several ducks may wear different robots)."""
    scene = scene or SCENE_XML
    if not use_bam:
        spec = mujoco.MjSpec.from_file(scene)
        terrain_mod.apply(spec, terrain)
        if GHOSTS:
            ghost_mod.apply(spec, GHOSTS)
        model = spec.compile()
        model.opt.timestep = TIMESTEP
        return model, mujoco.MjData(model), None
    from bam.model import load_model as bam_load
    from bam.mujoco import MujocoController
    bam_model = bam_load(motor_name="xl330", model="m6")
    bam_model.actuator.kp = BAM_KP_FW
    bam_model.actuator.vin = BAM_VIN
    bam_model.actuator.max_current = None
    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R
    spec = mujoco.MjSpec.from_file(scene)
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break
    terrain_mod.apply(spec, terrain)
    if GHOSTS:
        ghost_mod.apply(spec, GHOSTS)
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    # this bam branch takes a resistance: V_drop = R * sum|tau|/kt  ==  gain * sum|tau|
    ctrl = MujocoController(bam_model, names, model, data,
                            vin_drop_resistance=BAM_VIN_DROP_GAIN * kt, vin_min=BAM_VIN_MIN)
    return model, data, ctrl


class DuckSim:
    def __init__(self, use_bam=True, leash=LEASH_DEFAULT, log=print):
        self.log = log
        self.use_bam = use_bam
        self.model, self.data, self.bam = load_model(use_bam)
        m = self.model
        self.velstand = _make_session(os.path.join(POLICY_DIR, os.environ.get("MICRODUCK_WALK_POLICY", "velstand.onnx")))
        self.sitstand = _make_session(os.path.join(POLICY_DIR, "alpha_sitstand.onnx"))
        for s in (self.velstand, self.sitstand):
            assert s.get_inputs()[0].shape[-1] == 61, s.get_inputs()[0].shape
        self.imu_gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.fq = int(m.jnt_qposadr[fj])
        self.fv = int(m.jnt_dofadr[fj])
        self.jq = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.jv = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        assert m.nu == 14
        # backlash joints (2 deg play) exist in the *_backlash robots: velstand was trained with them,
        # alpha_sitstand without (standup robot) -> clamp them shut while sitstand drives.
        self.backlash_j = [j for j in range(m.njnt)
                           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_backlash")]
        self.backlash_range = m.jnt_range[self.backlash_j].copy() if self.backlash_j else None
        self._backlash_locked = False
        self.lock = threading.Lock()
        self.leash = float(leash)
        self.player = None  # (x, y)
        self.player_t = 0.0
        self.quack = 0
        self.resets = 0
        self.falls = 0
        self.terrain = None  # terrain.Terrain (set by terrain.swap); None = flat floor at z=0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        sx, sy, sz = terrain_mod.spawn_point(self)  # safe floor spot nearest the origin
        d.qpos[self.fq:self.fq + 3] = [sx, sy, sz + SPAWN_Z]
        d.qpos[self.fq + 3:self.fq + 7] = [1, 0, 0, 0]
        for i, a in enumerate(self.jq):
            d.qpos[a] = DEFAULT_POSE[i]
        if self.bam is not None:
            self.bam.last_ts = d.time
            self.bam.q_target[:] = DEFAULT_POSE
        else:
            d.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        self.t = 0.0 if not hasattr(self, "t") else self.t
        self.steps = 0 if not hasattr(self, "steps") else self.steps
        self.last_action = np.zeros(14, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.head_cmd = np.zeros(4, dtype=np.float32)
        self.mode = "stand"
        self.walk_cmd = (WALK_VX, 0.0, 0.0)
        self.turn_target = None
        self.turn_t0 = 0.0
        self.policy = "velstand"
        self.sit_flag = 0.0
        self.sit_t0 = 0.0
        self.pending = None  # mode to enter after standing up
        self.stand_after_settle = False
        self.returning = False
        self.walk_heading = 0.0
        self.walk_t0 = self.t
        self.turn_sign = 1.0
        self.fallen = False
        self.fallen_since = None
        self.quack_until = 0.0

    # ------------------------------------------------------------ commands
    def command(self, cmd, **kw):
        """Apply a high-level command. Returns (ok, error_or_None)."""
        sitting = self.mode in ("sit", "sitting_down", "standing_up")
        if cmd == "talk":
            return talk_mod.command(self, **kw)
        if cmd == "quack":
            self.quack += 1
            self.quack_until = self.t + 0.35
            return True, None
        if cmd == "reset":
            self.resets += 1
            self.reset()
            return True, None
        if cmd == "sit":
            if self.mode in ("sit", "sitting_down"):
                return True, None
            if self.fallen:
                return False, "duck is fallen; wait for it to get up"
            self.mode = "sitting_down"
            self.policy = "sitstand"
            self.sit_flag = 1.0
            self.sit_t0 = self.t
            self.pending = None
            self.stand_after_settle = False
            self.vel_cmd[:] = 0
            return True, None
        if cmd == "stand":
            if sitting:
                self._begin_stand(None)
            else:
                self.mode = "stand"
            return True, None
        if cmd in ("walk", "stop", "follow", "turn"):
            if cmd == "walk":
                vx = float(kw.get("vx", WALK_VX))
                vy = float(kw.get("vy", 0.0))
                wz = float(kw.get("wz", 0.0))
                self.walk_cmd = (float(np.clip(vx, -0.3, 0.3)), float(np.clip(vy, -0.2, 0.2)),
                                 float(np.clip(wz, -1.5, 1.5)))
                target = "walk"
            elif cmd == "stop":
                target = "stand"
            elif cmd == "follow":
                target = "follow"
            else:
                d = kw.get("dir", "left")
                if d not in ("left", "right"):
                    return False, "dir must be left or right"
                self.turn_sign = 1.0 if d == "left" else -1.0
                target = "turn"
            if sitting:
                self._begin_stand(None if target == "stand" else target)
                if target == "turn":
                    self.turn_target = None  # computed when turn actually starts
                return True, None
            self._enter(target)
            return True, None
        return False, f"unknown cmd {cmd!r}"

    def _begin_stand(self, pending):
        if self.mode == "sitting_down" and self.t - self.sit_t0 < SIT_SETTLE_S:
            # flipping the flag mid-descent can tip it over: finish the sit, then rise
            self.stand_after_settle = True
            self.pending = pending
            return
        self.stand_after_settle = False
        if RISE_WITH == "velstand":
            # velstand was trained with sit spawns and gets up from a seat by itself
            self.sit_flag = 0.0
            self.pending = None
            self._enter(pending or "stand")
            return
        self.mode = "standing_up"
        self.policy = "sitstand"
        self.sit_flag = 0.0
        self.sit_t0 = self.t
        self.pending = pending

    def _enter(self, mode):
        self.mode = mode
        self.policy = "velstand"
        self.returning = False
        if mode == "walk":
            self.walk_heading = self.yaw()
            self.walk_t0 = self.t
        if mode == "turn":
            self.turn_target = wrap(self.yaw() + self.turn_sign * math.pi / 2)
            self.turn_t0 = self.t

    def set_player(self, x, y):
        self.player = (float(x), float(y))
        self.player_t = self.t

    # ------------------------------------------------------------ helpers
    def pos(self):
        return self.data.qpos[self.fq:self.fq + 3]

    def quat(self):
        return self.data.qpos[self.fq + 3:self.fq + 7]

    def yaw(self):
        return yaw_of(self.quat())

    def proj_gravity(self):
        q = self.data.xquat[self.trunk_id].astype(np.float32)
        return quat_rotate_inverse(q, np.array([0, 0, -1.0], dtype=np.float32))

    def _goto(self, tx, ty, speed, stop_r):
        """Velocity command (body frame) steering toward a world point."""
        p = self.pos()
        dx, dy = tx - p[0], ty - p[1]
        dist = math.hypot(dx, dy)
        if dist < stop_r:
            return np.zeros(3, dtype=np.float32), True
        err = wrap(math.atan2(dy, dx) - self.yaw())
        wz = float(np.clip(2.0 * err, -1.2, 1.2))
        vx = speed * max(0.0, math.cos(err)) if abs(err) < 0.9 else 0.0
        if dist < stop_r + 0.25:
            vx *= max(0.4, (dist - stop_r) / 0.25)
        return np.array([vx, 0.0, wz], dtype=np.float32), False

    def _high_level(self):
        """Compute the velocity command for velstand from the current mode."""
        mode = self.mode
        p = self.pos()
        r = math.hypot(p[0], p[1])
        if mode == "walk":
            ux, uy, uw = self.walk_cmd
            self.walk_heading = wrap(self.walk_heading + uw * CONTROL_DT)
            herr = wrap(self.walk_heading - self.yaw())
            wz = float(np.clip(uw + 1.5 * herr, -1.2, 1.2))  # heading hold (policy drifts in yaw)
            vx, vy = float(np.clip(ux * SPEED_GAIN, -0.4, 0.4)), float(np.clip(uy * SPEED_GAIN, -0.3, 0.3))
            if self.t - self.walk_t0 < KICK_S and math.hypot(vx, vy) < KICK_V:
                # velstand has a deadband: from a standstill it only starts stepping
                # for |v| >~ 0.2, so give it a short kick to get the gait going.
                n = math.hypot(vx, vy)
                if n > 1e-3:
                    vx, vy = vx / n * KICK_V, vy / n * KICK_V
            # leash: velocity direction in world
            yaw = self.yaw()
            vwx = vx * math.cos(yaw) - vy * math.sin(yaw)
            vwy = vx * math.sin(yaw) + vy * math.cos(yaw)
            outward = (p[0] * vwx + p[1] * vwy) > 0
            margin = min(LEASH_MARGIN, 0.3 * self.leash)
            if not self.returning and r > self.leash - margin and outward:
                self.returning = True
            if self.returning:
                v, _ = self._goto(0.0, 0.0, FOLLOW_VX, 0.0)
                if r < max(0.2, self.leash * 0.35):
                    self.returning = False
                    self.walk_heading = self.yaw()  # carry on in the new direction
                return v
            return np.array([vx, vy, wz], dtype=np.float32)
        if mode == "follow":
            if self.player is None:
                return np.zeros(3, dtype=np.float32)
            tx, ty = self.player
            tr = math.hypot(tx, ty)
            lim = max(0.0, self.leash - min(LEASH_MARGIN, 0.3 * self.leash))
            if tr > lim:  # clamp target inside the leash
                tx, ty = tx * lim / tr, ty * lim / tr
                v, _ = self._goto(tx, ty, FOLLOW_VX, 0.15)
                return v
            v, _ = self._goto(tx, ty, FOLLOW_VX, getattr(self, "follow_stop", FOLLOW_STOP))  # multi: per-duck slot
            return v
        if mode == "turn":
            if self.turn_target is None:
                self.turn_target = wrap(self.yaw() + self.turn_sign * math.pi / 2)
                self.turn_t0 = self.t
            err = wrap(self.turn_target - self.yaw())
            if abs(err) < 0.08 or self.t - self.turn_t0 > TURN_TIMEOUT:
                self.mode = "stand"
                return np.zeros(3, dtype=np.float32)
            wz = float(np.clip(2.5 * err, -TURN_WZ, TURN_WZ))
            if abs(wz) < 0.5:
                wz = math.copysign(0.5, wz)
            return np.array([0.0, 0.0, wz], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------ obs/step
    def _obs(self, command):
        d = self.data
        ang = d.sensordata[self.imu_gyro_adr:self.imu_gyro_adr + 3].astype(np.float32)
        jp = d.qpos[self.jq].astype(np.float32) - DEFAULT_POSE
        jv = d.qvel[self.jv].astype(np.float32)
        if JOINT_VEL_LAG:
            prev = getattr(self, "_jv_prev", None)
            self._jv_prev = jv
            jv = jv if prev is None else prev
        return np.concatenate([ang, self.proj_gravity(), jp, jv, self.last_action, command]).astype(np.float32)

    def _update_fall(self):
        g = self.proj_gravity()
        z = terrain_mod.trunk_height(self)  # trunk height above the ground below it (= pos z on the flat floor)
        sitting = self.policy == "sitstand"
        tilted = g[2] > -0.5  # > 60 deg from upright
        low = z < (0.035 if sitting else 0.06)
        fallen = bool(tilted or low)
        if fallen and not self.fallen:
            self.falls += 1
            self.fallen_since = self.t
            p = self.pos()
            self.log(f"[sim] t={self.t:.2f} FALL detected (mode={self.mode} policy={self.policy} z={z:.3f} gz={g[2]:.2f} pos=({p[0]:.2f},{p[1]:.2f}))")
            if self.policy == "sitstand":  # velstand is the one that gets up
                self.policy = "velstand"
                self.mode = "stand"
        if not fallen:
            self.fallen_since = None
        self.fallen = fallen
        if fallen and self.fallen_since is not None and self.t - self.fallen_since > FALL_RESET_S:
            self.log(f"[sim] t={self.t:.1f} on the ground > {FALL_RESET_S}s -> auto reset")
            self.resets += 1
            self.reset()

    def step(self):
        """One 50 Hz control step (policy + 4 physics substeps)."""
        command = np.zeros(13, dtype=np.float32)
        if self.policy == "sitstand":
            # transitions
            el = self.t - self.sit_t0
            if self.mode == "sitting_down" and el > SIT_SETTLE_S:
                self.mode = "sit"
                if self.stand_after_settle:
                    self._begin_stand(self.pending)
            if self.mode == "standing_up" and el > STAND_HANDBACK_S:
                g = self.proj_gravity()
                if g[2] < -0.9 or el > STAND_HANDBACK_S + 3:
                    nxt = self.pending or "stand"
                    self.pending = None
                    self.policy = "velstand"
                    self._enter(nxt)
            command[0] = self.sit_flag
            self.vel_cmd[:] = 0
            sess = self.sitstand if self.policy == "sitstand" else self.velstand
        else:
            sess = self.velstand
        if self.policy == "velstand":
            self.vel_cmd = terrain_mod.avoid(self, self._high_level()) if not self.fallen else np.zeros(3, dtype=np.float32)
            hook = getattr(self, "peer_avoid", None)   # multi.py: steer around the other ducks
            if hook is not None and not self.fallen:
                self.vel_cmd = hook(self, self.vel_cmd)
            command[0:3] = self.vel_cmd
        if self.t < self.quack_until and not self.fallen:
            command[4] = -0.35  # quick head-pitch nod while quacking (head_pose command)
        talk_mod.apply(self, command, CONTROL_DT)
        obs = self._obs(command)
        lock = self.policy == "sitstand"
        if self.backlash_j and lock != self._backlash_locked:
            self.model.jnt_range[self.backlash_j] = (self.backlash_range * 0.02) if lock else self.backlash_range
            self._backlash_locked = lock
        action = sess.run(None, {sess.get_inputs()[0].name: obs.reshape(1, -1)})[0][0].astype(np.float32)
        self.last_action = action.copy()
        target = DEFAULT_POSE + action * ACTION_SCALE
        if self.bam is not None:
            self.bam.q_target[:] = target
        else:
            self.data.ctrl[:] = target
        for _ in range(DECIMATION):
            if self.bam is not None:
                self.bam.update()
            mujoco.mj_step(self.model, self.data)
        self.t += CONTROL_DT
        self.steps += 1
        self._update_fall()

    # ------------------------------------------------------------ public mode
    def public_mode(self):
        m = self.mode
        if m == "sitting_down":
            if self.stand_after_settle:
                return self.pending or "stand"
            return "sit"
        if m == "standing_up":
            return self.pending or "stand"
        return m
