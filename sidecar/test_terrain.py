#!/usr/bin/env python3
"""Headless terrain test (no HTTP): synthetic Minecraft-like room at scale 4 (1 block = 0.25 m).

Room: 1-block-thick floor (top z=0), 2-block walls around x,y in [-1.25, 1.25], a slab step (0.125 m) and a
carpet (1/16 block) inside; variant B adds a 1-block ledge (floor drops to z=-0.25 for x > 0.75).
Checks: runtime swap flat -> room keeps the duck standing (no NaN/fall); walking into a wall (avoidance off)
never passes through; 60 s of walking/turning inside the room with avoidance stays upright and inside;
carpet is walkable; swap A -> B at runtime; walking off the ledge falls and recovers/resets onto a floor;
spawn search never picks a spot inside a wall; rebuild timing.
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
import terrain as tm  # noqa: E402

B = 0.25
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def room(ledge=False):
    boxes = []
    if ledge:
        boxes.append([-1.5, -1.5, -B, 0.75, 1.5, 0.0])          # upper floor
        boxes.append([0.75, -1.5, -2 * B, 1.5, 1.5, -B])         # lower floor (one block down)
    else:
        boxes.append([-1.5, -1.5, -B, 1.5, 1.5, 0.0])
    zlo = -2 * B if ledge else -B
    boxes += [[-1.5, -1.5, zlo, -1.25, 1.5, 2 * B],             # west wall
              [1.25, -1.5, zlo, 1.5, 1.5, 2 * B],               # east wall
              [-1.25, -1.5, zlo, 1.25, -1.25, 2 * B],           # south wall (sim -y)
              [-1.25, 1.25, zlo, 1.25, 1.5, 2 * B],             # north wall
              [-1.0, 0.5, 0.0, -0.75, 1.0, B / 2],              # bottom slab: a 12.5 cm step
              [-0.5, -1.0, 0.0, 0.0, -0.5, B / 16]]             # carpet: 1.6 cm
    region = [-1.75, -1.75, zlo, 1.75, 1.75, 0.75]
    return tm.Terrain(boxes, region, {"id": "test-ledge" if ledge else "test-room"})


def apply(sim, terr):
    t0 = time.perf_counter()
    built = tm.build(sim, terr)
    reason = tm.swap(sim, terr, built, log=lambda *a: print("   ", *a))
    return time.perf_counter() - t0, reason


def place(sim, x, y, yaw):
    sim.reset()
    sx, sy, h = sim.terrain.find_spawn(x, y, 0.0) if sim.terrain else (x, y, 0.0)
    sim.data.qpos[sim.fq:sim.fq + 3] = [sx, sy, h + ds.SPAWN_Z]
    sim.data.qpos[sim.fq + 3:sim.fq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(sim.model, sim.data)


def run(sim, seconds, cb=None):
    bad = False
    for i in range(int(round(seconds / ds.CONTROL_DT))):
        sim.step()
        if not np.all(np.isfinite(sim.data.qpos)):
            bad = True
            break
        if cb:
            cb(sim)
    return not bad


class Stats:
    def __init__(self):
        self.min_gz_up = -1.0
        self.max_abs_x = 0.0
        self.max_abs_y = 0.0
        self.min_y = 9.0

    def __call__(self, sim):
        p = sim.pos()
        self.min_gz_up = max(self.min_gz_up, float(sim.proj_gravity()[2]))
        self.max_abs_x = max(self.max_abs_x, abs(float(p[0])))
        self.max_abs_y = max(self.max_abs_y, abs(float(p[1])))
        self.min_y = min(self.min_y, float(p[1]))


def main():
    t_wall = time.perf_counter()
    sim = ds.DuckSim(log=lambda *a: print("   ", *a))
    run(sim, 2.0)
    A = room(False)
    Bt = room(True)

    # --- spawn search sanity (pure numpy)
    bad = 0
    for x in np.linspace(-1.6, 1.6, 17):
        for y in np.linspace(-1.6, 1.6, 17):
            sx, sy, h = A.find_spawn(float(x), float(y), 0.0, max_r=1.5)
            inside = abs(sx) < 1.25 - tm.FOOT_R and abs(sy) < 1.25 - tm.FOOT_R
            on_slab = (-1.0 - tm.FOOT_R < sx < -0.75 + tm.FOOT_R) and (0.5 - tm.FOOT_R < sy < 1.0 + tm.FOOT_R)
            if not inside or on_slab or abs(h) > 0.02:
                bad += 1
    check("find_spawn: 289 query points -> safe floor spot (inside room, not in wall/slab, h=0)", bad == 0, f"bad={bad}")
    check("ground(): carpet top 1.6 cm, slab 12.5 cm, inside wall -> None",
          abs(A.ground(-0.25, -0.75, 0.0) - B / 16) < 1e-9 and abs(A.ground(-0.9, 0.7, 0.1) - B / 2) < 1e-9
          and A.ground(1.4, 0.0, 0.0) is None, "")

    # --- runtime swap flat -> room A while standing at the origin
    st = Stats()
    dt, reason = apply(sim, A)
    ok = run(sim, 3.0, st)
    p = sim.pos()
    check("swap flat->room keeps the duck standing (no respawn, no NaN, no fall)",
          ok and reason is None and sim.falls == 0 and st.min_gz_up < -0.85 and abs(p[2] - 0.116) < 0.02,
          f"build+swap {dt * 1000:.0f} ms, reason={reason}, z={p[2]:.3f}, worst gz={st.min_gz_up:.2f}, ngeom={sim.model.ngeom}")

    # --- wall: avoidance OFF, walk straight south (-y) into the wall at y=-1.25 for 12 s
    real_avoid = tm.avoid
    ds.terrain_mod.avoid = lambda s, v: v
    place(sim, 0.6, -0.4, -math.pi / 2)
    sim.command("walk", vx=0.15)
    st = Stats()
    ok = run(sim, 12.0, st)
    ds.terrain_mod.avoid = real_avoid
    # trunk centre can get to ~0.05-0.1 m from the face; anything below -1.25 + 0.03 means it went through
    check("walk into wall (no avoidance): never passes through", ok and st.min_y > -1.25 + 0.03,
          f"min trunk y={st.min_y:.3f} (wall face -1.25), falls={sim.falls}")

    # --- 60 s inside room A with avoidance: walk, turns, follow toward a target behind a wall
    place(sim, 0.0, 0.0, 0.0)
    falls0, resets0 = sim.falls, sim.resets
    st = Stats()
    seq = [("walk", {"vx": 0.15}, 15), ("turn", {"dir": "left"}, 4), ("walk", {"vx": 0.15}, 12),
           ("turn", {"dir": "right"}, 4), ("walk", {"vx": 0.12, "wz": 0.3}, 10), ("follow", {}, 15)]
    sim.set_player(2.5, 0.3)    # outside the east wall: follow must not tunnel
    ok = True
    path = 0.0
    last = sim.pos()[:2].copy()

    def cb(s):
        nonlocal path, last
        st(s)
        if s.steps % 25 == 0:
            path += float(np.hypot(*(s.pos()[:2] - last)))
            last = s.pos()[:2].copy()
    for cmd, kw, secs in seq:
        sim.command(cmd, **kw)
        ok = run(sim, secs, cb) and ok
    check("60 s walk/turn/follow in the room: upright, no falls/resets",
          ok and sim.falls == falls0 and sim.resets == resets0 and st.min_gz_up < -0.7,
          f"falls={sim.falls - falls0} resets={sim.resets - resets0} worst gz={st.min_gz_up:.2f} path={path:.2f} m")
    check("60 s: stayed inside the walls", st.max_abs_x < 1.25 and st.max_abs_y < 1.25,
          f"max|x|={st.max_abs_x:.3f} max|y|={st.max_abs_y:.3f}")
    check("60 s: actually moved (not wedged)", path > 1.0, f"path={path:.2f} m")

    # --- carpet: walk across it (it's 1.6 cm)
    place(sim, -0.25, -0.2, -math.pi / 2)
    sim.command("walk", vx=0.15)
    f0 = sim.falls
    zs = []
    ok = run(sim, 6.0, lambda s: zs.append(float(s.pos()[2])))
    p = sim.pos()
    on = -0.5 < p[0] < 0.0 and -1.0 < p[1] < -0.5
    h = tm.trunk_height(sim)
    check("carpet (1/16 block, ramped edges) is walkable: walked onto it, no fall",
          ok and sim.falls == f0 and on and abs((p[2] - h) - B / 16) < 0.003,
          f"end ({p[0]:.2f},{p[1]:.2f}) carpet x[-0.5,0] y[-1,-0.5], trunk z={p[2]:.3f}, surface below z={p[2] - h:.4f}")

    # --- runtime swap room A -> B (ledge appears) while walking
    place(sim, 0.0, 0.0, 0.0)
    sim.command("walk", vx=0.1)
    run(sim, 2.0)
    f0 = sim.falls
    dt, reason = apply(sim, Bt)
    st = Stats()
    ok = run(sim, 3.0, st)
    check("swap room->ledge at runtime while walking: no NaN, no fall", ok and sim.falls == f0 and st.min_gz_up < -0.7,
          f"build+swap {dt * 1000:.0f} ms reason={reason} worst gz={st.min_gz_up:.2f}")

    # --- ledge: walk east off the 1-block ledge at x=0.75
    place(sim, 0.45, 0.0, 0.0)
    sim.command("walk", vx=0.15)
    f0, r0 = sim.falls, sim.resets
    xs, zs = [], []
    ok = run(sim, 16.0, lambda s: (xs.append(float(s.pos()[0])), zs.append(float(s.pos()[2]))))
    p = sim.pos()
    went_over = max(xs) > 0.8
    dropped = min(zs) < 0.0
    h = tm.trunk_height(sim)
    upright = sim.proj_gravity()[2] < -0.8
    check("ledge: walked over the edge and dropped", ok and went_over and dropped,
          f"max x={max(xs):.2f} min trunk z={min(zs):.3f} falls={sim.falls - f0} resets={sim.resets - r0}")
    check("ledge: standing again on a floor afterwards (recovered or reset)",
          upright and 0.09 < h < 0.14 and not sim.fallen,
          f"end pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.3f}) height above ground={h:.3f} fallen={sim.fallen}")

    # --- reset with terrain spawns on the floor at the origin; clear terrain back to flat
    sim.command("reset")
    p = sim.pos().copy()
    run(sim, 2.0)
    check("reset with terrain -> spawn on the floor at the origin", abs(p[0]) < 1e-6 and abs(p[2] - ds.SPAWN_Z) < 1e-6
          and not sim.fallen, f"spawn={p.round(3).tolist()}")
    dt, reason = apply(sim, tm.Terrain(np.zeros((0, 6))))
    ok = run(sim, 2.0)
    check("clear terrain -> flat floor again", ok and sim.terrain is None and not sim.fallen,
          f"ngeom={sim.model.ngeom} reason={reason}")

    # --- anchor set while standing on a slab floor: sim z=0 is the slab BOTTOM, the floor top is at 0.125
    slab_floor = tm.Terrain([[-1.5, -1.5, -B, 1.5, 1.5, 0.0], [-1.5, -1.5, 0.0, 1.5, 1.5, B / 2]],
                            [-1.75, -1.75, -1.0, 1.75, 1.75, 0.75])
    dt, reason = apply(sim, slab_floor)
    ok = run(sim, 3.0)
    p = sim.pos()
    check("slab floor under the anchor: duck (inside the slab) respawns on top and stands",
          ok and reason is not None and abs(p[2] - (B / 2 + 0.116)) < 0.02 and not sim.fallen,
          f"reason={reason}, trunk z={p[2]:.3f} (want {B / 2 + 0.116:.3f})")
    sim.command("reset")
    p = sim.pos().copy()
    check("reset on a slab floor spawns on the slab top", abs(p[2] - (B / 2 + ds.SPAWN_Z)) < 1e-6, f"spawn z={p[2]:.3f}")

    # --- step cost with big terrains
    for n in (500, 2000):
        rng = np.random.default_rng(1)
        boxes = [[-3, -3, -0.25, 3, 3, 0.0]]
        for _ in range(n - 1):
            x, y = rng.uniform(-3, 3, 2)
            if math.hypot(x, y) < 0.6:
                continue
            w = rng.choice([0.25, 0.5])
            boxes.append([x, y, 0.0, x + w, y + 0.25, rng.choice([0.0156, 0.125, 0.25, 0.5])])
        dt, reason = apply(sim, tm.Terrain(boxes))
        place(sim, 0.0, 0.0, 0.0)
        sim.command("walk", vx=0.15)
        t0 = time.perf_counter()
        nst = 250
        run(sim, nst * ds.CONTROL_DT)
        per = (time.perf_counter() - t0) / nst * 1000
        check(f"{len(boxes)} boxes: control step well under 20 ms", per < 8.0,
              f"{per:.2f} ms/step, build+swap {dt * 1000:.0f} ms")

    print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed  (wall {time.perf_counter() - t_wall:.0f} s)")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
