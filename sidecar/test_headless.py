#!/usr/bin/env python3
"""Headless (no HTTP) test of the Microduck sim, faster than real time.

Checks: 60 s of walk / turn / follow without falling or resetting, trunk height, upright,
path length > 1 m, leash respected, sit -> stand, fall recovery, model.json sanity.
"""

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
import model_export  # noqa: E402

Z_MIN = 0.09        # standing trunk height is ~0.116 m; fallen is < 0.06
UPRIGHT_GZ = -0.8   # projected gravity z; -1 = perfectly upright (< ~37 deg tilt)

results = []


def check(name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


class Tracker:
    def __init__(self, sim):
        self.sim = sim
        self.min_z = 9.0
        self.max_gz = -1.0
        self.max_r = 0.0
        self.path = 0.0
        self.last = sim.pos()[:2].copy()
        self.n = 0

    def step(self, seconds):
        s = self.sim
        for _ in range(int(round(seconds / ds.CONTROL_DT))):
            s.step()
            self.n += 1
            p = s.pos()
            self.min_z = min(self.min_z, float(p[2]))
            self.max_gz = max(self.max_gz, float(s.proj_gravity()[2]))
            self.max_r = max(self.max_r, math.hypot(p[0], p[1]))
            if self.n % 25 == 0:  # sample path every 0.5 s (ignores in-step trunk sway)
                self.path += math.hypot(*(p[:2] - self.last))
                self.last = p[:2].copy()


def main():
    t_wall = time.perf_counter()
    sim = ds.DuckSim(log=lambda *a: print("   ", *a))
    print(f"scene={os.path.basename(ds.SCENE_XML)} bam={sim.bam is not None} speed_gain={ds.SPEED_GAIN}")

    # ---------------------------------------------------------------- 60 s locomotion
    sim.leash = 2.5
    for _ in range(100):  # settle 2 s
        sim.step()
    tr = Tracker(sim)
    t0 = sim.t
    falls0, resets0 = sim.falls, sim.resets

    sim.command("walk")                       # default ~0.15 m/s forward
    tr.step(20.0)
    p_after_walk = sim.pos().copy()
    print(f"    after 20 s walk: pos=({p_after_walk[0]:.2f},{p_after_walk[1]:.2f}) path={tr.path:.2f} m returning={sim.returning}")

    y0 = sim.yaw()
    sim.command("turn", dir="left")
    tr.step(5.0)
    dl = math.degrees(ds.wrap(sim.yaw() - y0))
    mode_after_left = sim.mode
    y0 = sim.yaw()
    sim.command("turn", dir="right")
    tr.step(5.0)
    dr = math.degrees(ds.wrap(sim.yaw() - y0))
    check("turn left ~90deg then stop", 60 <= dl <= 120 and mode_after_left == "stand", f"dyaw={dl:+.1f} deg, mode after={mode_after_left}")
    check("turn right ~-90deg then stop", -120 <= dr <= -60 and sim.mode == "stand", f"dyaw={dr:+.1f} deg, mode after={sim.mode}")

    sim.command("walk", vx=0.12, wz=0.4)      # arc walk with yaw override
    tr.step(10.0)

    sim.set_player(1.2, -0.8)
    sim.command("follow")
    tr.step(18.0)
    p = sim.pos()
    dfol = math.hypot(p[0] - 1.2, p[1] + 0.8)
    check("follow reaches player (stop within ~0.4 m)", dfol < 0.65, f"dist to player={dfol:.2f} m, vel_cmd={np.round(sim.vel_cmd, 2).tolist()}")
    sim.command("stop")
    tr.step(2.0)

    loco_t = sim.t - t0
    check("60 s locomotion: no fall", sim.falls == falls0, f"falls={sim.falls - falls0} over {loco_t:.1f} s sim")
    check("60 s locomotion: no reset", sim.resets == resets0, f"resets={sim.resets - resets0}")
    check("trunk height stays above threshold", tr.min_z > Z_MIN, f"min trunk z={tr.min_z:.3f} m (threshold {Z_MIN})")
    check("stays upright", tr.max_gz < UPRIGHT_GZ, f"max projected-gravity z={tr.max_gz:.3f} (tilt<={math.degrees(math.acos(-tr.max_gz)):.1f} deg)")
    check("actually moved (> 1 m path)", tr.path > 1.0, f"path length={tr.path:.2f} m")
    check("leash respected (default 2.5 m)", tr.max_r <= sim.leash, f"max dist from origin={tr.max_r:.2f} m (leash {sim.leash})")

    # ---------------------------------------------------------------- tighter leash, long straight walk
    sim.command("reset")
    for _ in range(50):
        sim.step()
    sim.leash = 1.0
    tr2 = Tracker(sim)
    f0 = sim.falls
    sim.command("walk")
    tr2.step(40.0)
    check("leash 1.0 m: steers back on a 40 s straight walk", tr2.max_r <= 1.0 and sim.falls == f0,
          f"max dist={tr2.max_r:.2f} m, path={tr2.path:.2f} m, falls={sim.falls - f0}")
    sim.set_player(5.0, 0.0)
    sim.command("follow")
    tr2.step(20.0)
    check("leash 1.0 m: follow player outside leash stays inside", tr2.max_r <= 1.0, f"max dist={tr2.max_r:.2f} m")
    sim.leash = ds.LEASH_DEFAULT

    # ---------------------------------------------------------------- sit / stand
    sim.command("reset")
    for _ in range(100):
        sim.step()
    z_stand = float(sim.pos()[2])
    f0, r0 = sim.falls, sim.resets
    sim.command("sit")
    zmin = 9.0
    for _ in range(int(4.0 / ds.CONTROL_DT)):
        sim.step()
        zmin = min(zmin, float(sim.pos()[2]))
    z_sit = float(sim.pos()[2])
    gz_sit = float(sim.proj_gravity()[2])
    check("sit (alpha_sitstand flag=1)", sim.mode == "sit" and z_sit < z_stand - 0.03 and gz_sit < -0.7 and sim.falls == f0,
          f"trunk z {z_stand:.3f} -> {z_sit:.3f} m, gz={gz_sit:.2f}, mode={sim.mode}, policy={sim.policy}")
    sim.command("stand")
    for _ in range(int(5.0 / ds.CONTROL_DT)):
        sim.step()
    z_up = float(sim.pos()[2])
    check("stand back up (flag=0, hand back to velstand)",
          sim.mode == "stand" and sim.policy == "velstand" and z_up > 0.105 and sim.falls == f0 and sim.resets == r0,
          f"trunk z={z_up:.3f} m, gz={sim.proj_gravity()[2]:.2f}, mode={sim.mode}, policy={sim.policy}")
    sim.command("sit")
    for _ in range(int(3.0 / ds.CONTROL_DT)):
        sim.step()
    sim.command("walk")  # walk while seated -> stand up first, then walk
    tr3 = Tracker(sim)
    tr3.step(8.0)
    check("walk while seated: stands up then walks", sim.mode == "walk" and tr3.path > 0.3 and sim.falls == f0,
          f"mode={sim.mode}, path={tr3.path:.2f} m, falls={sim.falls - f0}")

    # ---------------------------------------------------------------- quick sit -> stand flip (mid-descent)
    f0 = sim.falls
    for wait in (0.3, 0.8, 1.5):
        sim.command("stop")
        for _ in range(100):
            sim.step()
        sim.command("sit")
        for _ in range(int(wait / ds.CONTROL_DT)):
            sim.step()
        sim.command("stand")
        sim.command("follow")
        for _ in range(int(6.0 / ds.CONTROL_DT)):
            sim.step()
    check("sit then immediate stand/follow (0.3/0.8/1.5 s) doesn't fall", sim.falls == f0 and sim.mode == "follow",
          f"falls={sim.falls - f0}, mode={sim.mode}")

    # ---------------------------------------------------------------- quack
    q0 = sim.quack
    sim.command("quack")
    sim.command("quack")
    check("quack counter increments", sim.quack == q0 + 2, f"quack {q0} -> {sim.quack}")

    # ---------------------------------------------------------------- talk (speaking head nod) doesn't fall
    import talk as talk_mod
    f0, r0 = sim.falls, sim.resets
    head_seen = []
    for setup in ("stop", "walk", "sit"):
        sim.command(setup)
        for _ in range(int(3.0 / ds.CONTROL_DT)):
            sim.step()
        sim.command("talk", on=True)
        z_min = 9.0
        hp = []
        for _ in range(int(8.0 / ds.CONTROL_DT)):
            sim.step()
            z_min = min(z_min, float(sim.pos()[2]))
            hp.append(float(sim.data.qpos[sim.jq[6]]))  # head_pitch joint
        head_seen.append(max(hp[100:]) - min(hp[100:]))
        talking = talk_mod.talking(sim)
        sim.command("talk", on=False)
        for _ in range(int(1.0 / ds.CONTROL_DT)):
            sim.step()
        print(f"    talk during {setup}: talking={talking} min z={z_min:.3f} head_pitch swing={head_seen[-1]:.3f} rad mode={sim.mode}")
    sim.command("talk", duration=1.0)
    for _ in range(int(1.5 / ds.CONTROL_DT)):
        sim.step()
    check("talk on/off during stand/walk/sit doesn't fall; duration expires",
          sim.falls == f0 and sim.resets == r0 and not talk_mod.talking(sim) and talk_mod.count(sim) >= 4,
          f"falls={sim.falls - f0} resets={sim.resets - r0} talks={talk_mod.count(sim)}")
    check("talk visibly moves the head (head_pitch swing > 0.05 rad standing)", head_seen[0] > 0.05,
          f"swing stand/walk/sit = {[round(h, 3) for h in head_seen]} rad")
    sim.command("stand")
    for _ in range(int(3.0 / ds.CONTROL_DT)):
        sim.step()

    # ---------------------------------------------------------------- fall recovery
    sim.command("reset")
    sim.data.qpos[sim.fq + 2] = 0.06
    sim.data.qpos[sim.fq + 3:sim.fq + 7] = [0.7071, 0, -0.7071, 0]  # on its back
    mujoco.mj_forward(sim.model, sim.data)
    f0, r0 = sim.falls, sim.resets
    t_up = None
    for i in range(int(6.0 / ds.CONTROL_DT)):
        sim.step()
        if t_up is None and not sim.fallen and sim.proj_gravity()[2] < -0.9 and sim.pos()[2] > 0.1:
            t_up = (i + 1) * ds.CONTROL_DT
    check("fall recovery (velstand gets up, or auto-reset after 3 s)",
          (t_up is not None) or sim.resets > r0, f"upright after {t_up} s, auto-resets={sim.resets - r0}")

    # forced auto-reset path: pin it on its back by disabling the policy targets
    sim.command("reset")
    sim.data.qpos[sim.fq + 2] = 0.05
    sim.data.qpos[sim.fq + 3:sim.fq + 7] = [0.7071, 0, -0.7071, 0]
    mujoco.mj_forward(sim.model, sim.data)
    r0 = sim.resets
    t_fall = sim.t
    real_step = sim.step
    reset_at = None
    for i in range(int(5.0 / ds.CONTROL_DT)):
        # hold it down: re-pin trunk orientation each step to simulate "stuck"
        if sim.resets == r0:
            sim.data.qpos[sim.fq + 3:sim.fq + 7] = [0.7071, 0, -0.7071, 0]
            sim.data.qpos[sim.fq + 2] = 0.05
        real_step()
        if sim.resets > r0 and reset_at is None:
            reset_at = sim.t - t_fall
    check("auto-reset after ~3 s stuck on the ground", reset_at is not None and 2.5 <= reset_at <= 3.6,
          f"reset after {reset_at if reset_at is None else round(reset_at, 2)} s; pos now {np.round(sim.pos(), 3).tolist()}")

    # ---------------------------------------------------------------- model.json sanity
    sim.command("reset")
    for _ in range(50):
        sim.step()
    mj, ids, ntris = model_export.build_model_json(sim.model, ao=False)   # geometry only (AO bake is slow)
    pts = []
    for b, body in zip(ids, mj["bodies"]):
        R = sim.data.xmat[b].reshape(3, 3)
        for m in body["meshes"]:
            v = np.array(m["verts"]).reshape(-1, 3)
            pts.append(v @ R.T + sim.data.xpos[b])
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    check("model.json <= 26k tris and posed duck sits on the floor",
          ntris <= 26000 and -0.01 < lo[2] < 0.02 and 0.18 < hi[2] < 0.35,
          f"{ntris} tris, {len(ids)} bodies, posed bbox z=[{lo[2]:.3f},{hi[2]:.3f}] x=[{lo[0]:.3f},{hi[0]:.3f}] y=[{lo[1]:.3f},{hi[1]:.3f}]")

    wall = time.perf_counter() - t_wall
    n_fail = sum(1 for r in results if not r[1])
    print(f"\nsim steps={sim.steps} ({sim.steps * ds.CONTROL_DT:.0f} s sim) in {wall:.1f} s wall "
          f"({sim.steps * ds.CONTROL_DT / wall:.0f}x real time)")
    print(f"SUMMARY: {len(results) - n_fail}/{len(results)} passed -> {'PASS' if n_fail == 0 else 'FAIL'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
