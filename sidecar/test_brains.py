#!/usr/bin/env python3
"""Headless tests for community brains (brains.py): one check per catalogued brain + hot-swap invariants.

    .venv/bin/python test_brains.py          # ~20-40 s wall, nice it: nice -n 19
"""

import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import brains as B  # noqa: E402
import duck_sim as ds  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def run(sim, s):
    for _ in range(int(round(s / ds.CONTROL_DT))):
        sim.step()


def main():
    t_wall = time.perf_counter()
    cat = B.load_catalogue()
    check("catalogue has 6-12 featured brains", 6 <= len(cat) <= 12, f"{len(cat)}: {sorted(cat)}")
    import re
    check("ids are [a-z0-9-]{1,24}", all(re.match(B.ID_RE, i) for i in cat), "")
    check("every brain has repo/licence/kind/eval", all(e.get("repo") and e.get("kind") in ("perpetual", "episodic")
                                                       and e.get("eval_summary") for e in cat.values()), "")

    sim = B.BrainSim(log=lambda *a: None)
    # ---- one check per catalogued brain, through the runtime command path (leash 2.5 like the service)
    for bid, e in sorted(cat.items(), key=lambda kv: (kv[1]["kind"], kv[0])):
        sim.command("reset")
        sim.leash = 2.5
        f0, r0 = sim.falls, sim.resets
        run(sim, 0.6)
        if e["kind"] == "perpetual":
            ok, err = sim.command("brain", name=bid)
            p0 = sim.pos().copy()
            kept = 0
            max_r = 0.0
            path, last = 0.0, p0[:2].copy()
            n = int(8.0 / ds.CONTROL_DT)
            for k in range(n):
                sim.step()
                kept += sim.slot == bid
                max_r = max(max_r, math.hypot(*sim.pos()[:2]))
                if k % 25 == 24:
                    path += float(np.linalg.norm(sim.pos()[:2] - last))
                    last = sim.pos()[:2].copy()
            dist = path
            falls = sim.falls - f0 + sim.resets - r0
            good = ok and kept == n and falls == 0 and dist > 1.0 and max_r < 2.8
            check(f"brain {bid:10s} ({e['variant']})", good,
                  f"ok={ok} {err or ''} path={dist:.2f} m in 8 s, max_r={max_r:.2f}, falls={falls}, health brain={sim.brain_info()}")
            sim.command("brain", name="default")
            run(sim, 2.0)
            check(f"brain {bid:10s} -> default", sim.slot == "default" and sim.variant == "default" and not sim.fallen,
                  f"{sim.brain_info()} fallen={sim.fallen}")
        else:
            if e["variant"] == "rollers":
                rid = next((k for k, v in cat.items() if v["variant"] == "rollers" and v["kind"] == "perpetual"), None)
                if rid is None:
                    check(f"trick {bid}", False, "no roller brain to put skates on")
                    continue
                sim.command("brain", name=rid)
                run(sim, 0.5)
            ok, err = sim.command("trick", name=bid)
            mode_during = sim.public_mode()
            pr = B.Probe(sim)
            for _ in range(int((e["duration_s"] or 1.0) / ds.CONTROL_DT) + 2):
                sim.step()
                pr.tick()
            m = pr.summary()
            ended = sim.trick is None
            run(sim, 5.0)
            m["end_tilt_deg"] = 0.0
            did = B.thing_done(e.get("expect"), {**m, "end_tilt_deg": e.get("eval_summary", {}).get("end_tilt_deg", 90)})
            up = (not sim.fallen) and sim.proj_gravity()[2] < -0.8
            resets = sim.resets - r0
            check(f"trick {bid:11s}", ok and mode_during == "trick" and ended and did and up and resets == 0,
                  f"ok={ok} {err or ''} mode={mode_during} ended={ended} did={did} upright_after={up} "
                  f"max_tilt={m['max_tilt_deg']} air={m['max_air_s']} dist={m['dist']}")
            if e["variant"] == "rollers":
                sim.command("stop")

    # ---- hot-swap invariants
    sim.command("reset")
    sim.leash = 2.5
    ok, err = sim.command("brain", name="nope-nope")
    check("unknown brain rejected", not ok and "unknown" in (err or ""), err)
    trick = next((k for k, v in cat.items() if v["kind"] == "episodic"), None)
    perp = next((k for k, v in cat.items() if v["kind"] == "perpetual" and v["variant"] == "default"), None)
    if trick:
        ok, err = sim.command("brain", name=trick)
        check("trick id as brain rejected", not ok, err)
    if perp:
        ok, err = sim.command("trick", name=perp)
        check("brain id as trick rejected", not ok, err)
    roller = next((k for k, v in cat.items() if v["variant"] == "rollers" and v["kind"] == "perpetual"), None)
    if roller:
        sim.command("reset")
        run(sim, 0.5)
        sim.command("turn", dir="left")
        run(sim, 3.0)
        p, yaw = sim.pos().copy(), sim.yaw()
        sim.command("brain", name=roller)
        p2, yaw2 = sim.pos().copy(), sim.yaw()
        check("roller swap keeps x/y/yaw", np.linalg.norm(p2[:2] - p[:2]) < 0.02 and abs(ds.wrap(yaw2 - yaw)) < 0.05
              and sim.variant == "rollers" and sim.model.nq != sim._default_model[0].nq,
              f"dxy={np.linalg.norm(p2[:2] - p[:2]):.3f} dyaw={math.degrees(ds.wrap(yaw2 - yaw)):.1f} nq={sim.model.nq}")
        ids = sim.render_ids(["trunk_base", "ankle_left", "ankle_right"], [1, 6, 15])
        check("render ids map onto the roller robot", len(set(ids)) == 3 and min(ids) > 0, str(ids))
        run(sim, 2.0)
        sim.command("sit")
        check("sit while skating -> skates off, sitting", sim.variant == "default" and sim.slot == "default"
              and sim.public_mode() == "sit", f"{sim.brain_info()} mode={sim.public_mode()}")
        run(sim, 3.0)
        sim.command("brain", name=roller)
        run(sim, 1.0)
        sim.command("reset")
        check("reset on skates -> default robot at origin", sim.variant == "default" and sim.slot == "default"
              and math.hypot(*sim.pos()[:2]) < 0.05, f"{sim.brain_info()} pos={sim.pos()[:2]}")
    if perp:
        # fall while a brain drives -> default brain, velstand gets up, nothing left in the slot
        sim.command("reset")
        run(sim, 0.5)
        sim.command("brain", name=perp)
        run(sim, 1.0)
        sim.data.qvel[sim.fv:sim.fv + 3] = [0.0, 1.6, 0.0]   # shove sideways
        sim.data.qvel[sim.fv + 3:sim.fv + 6] = [9.0, 0.0, 0.0]
        f0 = sim.falls
        run(sim, 8.0)
        check("brain fall -> default brain + recovered", sim.falls > f0 and sim.slot == "default" and not sim.fallen,
              f"falls+{sim.falls - f0} slot={sim.slot} fallen={sim.fallen} resets={sim.resets}")
        # leash: a brain never takes it past the leash
        sim.command("reset")
        sim.leash = 1.0
        run(sim, 0.5)
        sim.command("brain", name=perp)
        mx = 0.0
        for _ in range(int(20 / ds.CONTROL_DT)):
            sim.step()
            mx = max(mx, math.hypot(*sim.pos()[:2]))
        check("leash 1.0 holds with a brain", mx < 1.3 and sim.slot == perp, f"max_r={mx:.2f} slot={sim.slot}")
        sim.leash = 2.5
    if trick:
        # walk/stop during a trick cancels it and hands back to velstand
        sim.command("reset")
        run(sim, 0.5)
        sim.command("trick", name=trick)
        run(sim, 0.2)
        ok_sit, err_sit = sim.command("sit")
        check("sit during a trick refused (would sit down mid-roll)", not ok_sit and sim.trick is not None, err_sit)
        sim.command("stop")
        check("stop cancels a trick", sim.trick is None and sim.public_mode() == "stand", sim.public_mode())
        run(sim, 4.0)
        # sitting: trick refused
        sim.command("sit")
        run(sim, 2.5)
        ok, err = sim.command("trick", name=trick)
        check("trick refused while sitting", not ok and "sitting" in (err or ""), err)
        sim.command("stand")
        run(sim, 4.0)
    # auto-reset path: a roller brain can't get up -> on the ground > 3 s -> reset -> default brain + robot
    if roller:
        sim.command("reset")
        run(sim, 0.3)
        sim.command("brain", name=roller)
        run(sim, 0.5)
        d = sim.data
        d.qpos[sim.fq + 2] = 0.08
        d.qpos[sim.fq + 3:sim.fq + 7] = [0.0, 1.0, 0.0, 0.0]   # upside down
        r0 = sim.resets
        run(sim, ds.FALL_RESET_S + 1.0)
        check("roller fall -> auto-reset -> default brain/robot", sim.resets == r0 + 1 and sim.slot == "default"
              and sim.variant == "default" and not sim.fallen, f"resets+{sim.resets - r0} {sim.brain_info()}")
    wall = time.perf_counter() - t_wall
    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} checks passed in {wall:.1f} s wall")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
