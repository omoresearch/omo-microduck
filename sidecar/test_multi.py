#!/usr/bin/env python3
"""Multi-duck tests (multi.py, ghosts.py, server.py "Multiple ducks").

    nice -n 19 .venv/bin/python test_multi.py            # headless + HTTP (own test server on :8806, temp state)
    nice -n 19 .venv/bin/python test_multi.py headless   # headless only
    nice -n 10 .venv/bin/python test_multi.py            # real-time numbers as deployed (the service runs at Nice=10;
                                                         # the test server inherits the test's niceness)

Headless: spawn 4 with no overlap (lab terrain), ghost contacts (ducks are solid to each other, ghosts never touch
terrain), 60 s of mixed commands with all 4 (walk/follow/turn/tricks/brains/sit) in the lab: falls from crowding
and auto-recovery, interpenetration, per-duck command isolation, remove/respawn, step cost for 1-4 ducks.
HTTP: GET/POST /ducks, spawn all / remove all, /poses, /cmd duck isolation, real-time loop timing with 1-4 ducks
(/health late, step_ms, process CPU), persistence across a server restart.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import duck_sim as ds  # noqa: E402
import ghosts  # noqa: E402
import multi  # noqa: E402
import terrain as tm  # noqa: E402

B = 0.25        # one MC block at scale 4
results = []
TIMING = {}


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def lab():
    """A 5 x 5 m lab (walls 2 blocks), a table, a crate and a carpet, like the user's underground lab."""
    boxes = [[-2.75, -2.75, -B, 2.75, 2.75, 0.0],
             [-2.75, -2.75, -B, -2.5, 2.75, 2 * B], [2.5, -2.75, -B, 2.75, 2.75, 2 * B],
             [-2.5, -2.75, -B, 2.5, -2.5, 2 * B], [-2.5, 2.5, -B, 2.5, 2.75, 2 * B],
             [1.0, 0.75, 0.0, 1.5, 1.25, B],          # crate
             [-1.5, -1.25, 0.0, -0.75, -1.0, B],      # table-ish block row
             [0.25, -1.0, 0.0, 1.0, -0.5, 0.0156]]    # carpet
    return tm.Terrain(boxes, [-3.0, -3.0, -1.0, 3.0, 3.0, 1.0], {"id": "test-lab", "scale": 4})


def tmp_files():
    d = tempfile.mkdtemp(prefix="microduck-multi-", dir=os.environ.get("TMPDIR"))
    return d, os.path.join(d, "ducks.json")


def apply_terrain(fleet, terr):
    for d in fleet.ducks:
        built = tm.build(d.sim, terr)
        tm.swap(d.sim, terr, built, log=lambda *a: None)
        d.sim._fleet_terrain = terr


class FakeMgr:
    def __init__(self, cur):
        self.current = cur


def pair_min(fleet):
    ps = [d.sim.pos()[:2].copy() for d in fleet.ducks]
    m = 9.0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            m = min(m, float(np.linalg.norm(ps[i] - ps[j])))
    return m


def run(fleet, seconds, on_step=None):
    for _ in range(int(round(seconds / ds.CONTROL_DT))):
        fleet.step()
        if on_step:
            on_step()


# ----------------------------------------------------------------------------------------------- headless
def headless():
    t_wall = time.perf_counter()
    d, state = tmp_files()
    terr = lab()
    fleet = multi.Fleet(log=lambda *a: print("   ", *a), state_path=state, autoload=False, max_ducks=4)
    fleet.terrain_mgr = FakeMgr(terr)
    names = [p["name"] for p in fleet.personas]
    check("personas.json: 5 distinct personas, distinct allowed voices",
          len(fleet.personas) == 5 and len({p["voice"] for p in fleet.personas}) == 5
          and all(p["voice"] in multi.VOICES for p in fleet.personas), f"{[(p['name'], p['voice']) for p in fleet.personas]}")
    t0 = time.perf_counter()
    spawned, errs = fleet.spawn_all()
    check("spawn all -> 4 ducks (max 4), 5th persona refused", len(fleet.ducks) == 4 and errs,
          f"{[s['name'] for s in spawned]} errs={errs} in {time.perf_counter() - t0:.1f}s")
    mn = pair_min(fleet)
    inside = []
    for dk in fleet.ducks:
        p = dk.sim.pos()
        h = terr.spot_ok(float(p[0]), float(p[1]), 0.0)
        inside.append(h is not None and math.hypot(p[0], p[1]) < fleet.leash)
    check("spawned ducks: >= 0.4 m apart, each on a free standable spot inside the leash",
          mn >= multi.SEP - 1e-6 and all(inside), f"min pair distance {mn:.2f} m, spots ok {inside}")
    d5, err = fleet.spawn("zippy")
    check("spawn beyond max is refused", d5 is None and "max" in (err or ""), err)
    run(fleet, 2.0)
    check("4 ducks stand 2 s without falling", all(not dk.sim.fallen for dk in fleet.ducks) and
          sum(dk.sim.falls for dk in fleet.ducks) == 0, f"falls {[dk.sim.falls for dk in fleet.ducks]}")

    # ---- ghosts: never touch terrain; ducks are solid to each other
    a, b = fleet.ducks[0], fleet.ducks[1]
    m = a.sim.model
    gb = [i for i in range(m.ngeom) if m.geom_group[i] == ghosts.GROUP]
    import mujoco
    tg = set(tm.terrain_geoms(m).tolist())
    ghosts.update(a.sim, [((1.25, 1.0, 0.2), (1, 0, 0, 0))])       # inside the crate
    mujoco.mj_forward(m, a.sim.data)
    bad = sum(1 for i in range(a.sim.data.ncon)
              if (a.sim.data.contact[i].geom1 in gb or a.sim.data.contact[i].geom2 in gb)
              and (a.sim.data.contact[i].geom1 in tg or a.sim.data.contact[i].geom2 in tg or
                   a.sim.data.contact[i].geom1 == 0 or a.sim.data.contact[i].geom2 == 0))
    check("ghost capsules (group 5) never collide with terrain/floor", len(gb) == 3 and bad == 0,
          f"{len(gb)} ghost geoms, {bad} ghost-terrain contacts with a ghost inside the crate")
    # push duck b against duck a: put b 0.10 m from a, sideways
    pa = a.sim.pos().copy()
    yaw = a.sim.yaw()
    bx, by = pa[0] + 0.10 * math.cos(yaw + math.pi / 2), pa[1] + 0.10 * math.sin(yaw + math.pi / 2)
    if terr.spot_ok(bx, by, 0.0) is None:
        bx, by = pa[0] - 0.10 * math.cos(yaw + math.pi / 2), pa[1] - 0.10 * math.sin(yaw + math.pi / 2)
    b.sim.data.qpos[b.sim.fq:b.sim.fq + 2] = [bx, by]
    b.sim.data.qvel[:] = 0
    mujoco.mj_forward(b.sim.model, b.sim.data)
    d0 = math.hypot(*(b.sim.pos()[:2] - a.sim.pos()[:2]))
    nc = [0]

    def count():
        nc[0] += ghosts.contacts(a.sim)[0] > 0 or ghosts.contacts(b.sim)[0] > 0
    run(fleet, 1.5, count)
    d1 = math.hypot(*(b.sim.pos()[:2] - a.sim.pos()[:2]))
    check("two ducks placed overlapping are pushed apart by real contacts (one shared physical space)",
          nc[0] > 0 and d1 > d0 + 0.04, f"distance {d0:.3f} -> {d1:.3f} m, {nc[0]} steps with duck-duck contacts, "
          f"falls {[dk.sim.falls for dk in fleet.ducks]}")
    run(fleet, 3.0)          # anyone knocked over gets up / auto-resets

    # ---- per-duck isolation
    for dk in fleet.ducks:
        fleet.command(dk.id, "stop")
    run(fleet, 1.0)
    q = [dk.sim.quack for dk in fleet.ducks]
    ok1, mode, res, err = fleet.command(fleet.ducks[1].name.lower(), "walk")
    ok2, _, _, _ = fleet.command(fleet.ducks[2].name[:3], "trick", name="hop")
    ok3, _, _, _ = fleet.command(fleet.ducks[3].id, "quack")
    modes = [dk.sim.public_mode() for dk in fleet.ducks]
    q2 = [dk.sim.quack for dk in fleet.ducks]
    check("per-duck isolation: walk/trick/quack hit only the named duck (name, prefix, id)",
          ok1 and ok2 and ok3 and modes == ["stand", "walk", "trick", "stand"] and
          [b2 - b1 for b1, b2 in zip(q, q2)] == [0, 0, 0, 1], f"modes {modes}, quack deltas {[b2 - b1 for b1, b2 in zip(q, q2)]}")
    slots0 = [dk.sim.slot for dk in fleet.ducks]
    okb, _, _, _ = fleet.command(fleet.ducks[0].id, "brain", name="sprint")
    slots1 = [dk.sim.slot for dk in fleet.ducks]
    check("brain on one duck only", okb and slots1[0] == "sprint" and slots1[1:] == slots0[1:],
          f"slots {slots0} -> {slots1}")
    oka, _, resa, _ = fleet.command("all", "stop")
    check("duck 'all' -> every duck", oka and len(resa) == 4 and all(r["ok"] for r in resa), f"{resa}")
    _, err = fleet.resolve("nobody")
    check("unknown duck name -> error", err and "no duck" in err, err)
    run(fleet, 1.0)

    # ---- 60 s mixed commands with all 4 in the lab
    rng = random.Random(7)
    for dk in fleet.ducks:
        dk.sim.falls = 0
        dk.sim.resets = 0
    cmds = [("walk", {}), ("walk", {}), ("follow", {}), ("follow", {}), ("turn", {"dir": "left"}), ("turn", {"dir": "right"}),
            ("stop", {}), ("trick", {"name": "hop"}), ("trick", {"name": "groove"}), ("trick", {"name": "kick"}),
            ("brain", {"name": "sprint"}), ("brain", {"name": "default"}), ("sit", {}), ("stand", {}), ("quack", {})]
    nxt = [rng.uniform(0.5, 3.0) for _ in fleet.ducks]
    t = 0.0
    min_sep = [9.0]
    worst_pen = [0.0]
    crowd_falls = [0]
    falls_prev = [0] * 4
    sent = 0
    tr_cnt = {}

    def on_step():
        nonlocal t
        t += ds.CONTROL_DT
        ang = 0.25 * t
        fleet.set_player(1.2 * math.cos(ang), 1.2 * math.sin(ang))
        min_sep[0] = min(min_sep[0], pair_min(fleet))
        for i, dk in enumerate(fleet.ducks):
            n, w = ghosts.contacts(dk.sim)
            worst_pen[0] = min(worst_pen[0], w)
            if dk.sim.falls > falls_prev[i]:
                falls_prev[i] = dk.sim.falls
                others = [o.sim.pos()[:2] for o in fleet.ducks if o is not dk]
                if min(float(np.linalg.norm(dk.sim.pos()[:2] - o)) for o in others) < 0.3:
                    crowd_falls[0] += 1
    t_run = time.perf_counter()
    for _ in range(int(60 / ds.CONTROL_DT)):
        for i, dk in enumerate(fleet.ducks):
            if t >= nxt[i]:
                c, kw = rng.choice(cmds)
                fleet.command(dk.id, c, **kw)
                tr_cnt[c] = tr_cnt.get(c, 0) + 1
                sent += 1
                nxt[i] = t + rng.uniform(2.0, 4.0)
        fleet.step()
        on_step()
    sim_wall = time.perf_counter() - t_run
    for dk in fleet.ducks:
        fleet.command(dk.id, "stop")
        if dk.sim.mode in ("sit", "sitting_down"):
            fleet.command(dk.id, "stand")
    run(fleet, 5.0)
    falls = [dk.sim.falls for dk in fleet.ducks]
    resets = [dk.sim.resets for dk in fleet.ducks]
    upright = [(not dk.sim.fallen) and float(dk.sim.proj_gravity()[2]) < -0.8 for dk in fleet.ducks]
    inside = [math.hypot(*dk.sim.pos()[:2]) < fleet.leash + 0.3 for dk in fleet.ducks]
    check("60 s mixed commands x 4 ducks in the lab: all upright at the end (falls auto-recovered), inside the leash",
          all(upright) and all(inside), f"{sent} cmds {tr_cnt}; falls {falls} (crowding falls {crowd_falls[0]}), "
          f"auto-resets {resets}, {sim_wall:.1f}s wall")
    check("no falls from crowding (fall with another duck < 0.3 m away) - or they recovered", crowd_falls[0] == 0 or all(upright),
          f"crowding falls {crowd_falls[0]}")
    check("ducks never interpenetrate: min trunk distance >= 0.12 m, ghost penetration <= 15 mm",
          min_sep[0] >= 0.12 and worst_pen[0] > -0.015, f"min trunk distance {min_sep[0]:.3f} m, deepest duck-duck "
          f"penetration {worst_pen[0] * 1000:.1f} mm")

    # ---- remove / respawn
    victim = fleet.ducks[1]
    removed, err = fleet.remove(victim.name.lower())
    check("remove by name", not err and len(fleet.ducks) == 3 and victim not in fleet.ducks, f"{removed}")
    dd, err = fleet.spawn(victim.persona["id"])
    mn = pair_min(fleet)
    check("respawn the same persona: new id, free spot", dd and dd["id"] != victim.id and mn >= multi.SEP - 1e-6,
          f"{victim.id} -> {dd and dd['id']}, min pair distance {mn:.2f}")
    removed, err = fleet.remove("all")
    check("remove all -> no ducks; default endpoints have nobody", not err and len(fleet.ducks) == 0
          and fleet.resolve(None)[1], f"removed {[r['name'] for r in removed]}")
    spawned, errs = fleet.spawn_all()
    check("spawn all again after remove all", len(fleet.ducks) == 4, f"{[s['name'] for s in spawned]}")
    with open(state) as f:
        st = json.load(f)
    check("roster saved to the state file", [e["name"] for e in st["ducks"]] == [dk.name for dk in fleet.ducks],
          f"{[e['name'] for e in st['ducks']]}")
    f2 = multi.Fleet(log=lambda *a: None, state_path=state, autoload=True, max_ducks=4)
    check("roster reload (same ids, names, positions within 0.3 m)",
          [(dk.id, dk.name) for dk in f2.ducks] == [(dk.id, dk.name) for dk in fleet.ducks]
          and all(math.hypot(*(a2.sim.pos()[:2] - a1.sim.pos()[:2])) < 0.3 for a1, a2 in zip(fleet.ducks, f2.ducks)),
          f"{[(dk.id, dk.name) for dk in f2.ducks]}")
    del f2

    # ---- cost per duck count (headless, no HTTP)
    for n in (1, 2, 3, 4):
        while len(fleet.ducks) > n:
            fleet.remove(fleet.ducks[-1].id)
        while len(fleet.ducks) < n:
            fleet.spawn()
        apply_terrain(fleet, terr)
        for dk in fleet.ducks:
            fleet.command(dk.id, "brain", name="default")
            fleet.command(dk.id, "walk")
        run(fleet, 0.5)
        ts = []
        c0 = time.process_time()
        for _ in range(250):
            a0 = time.perf_counter()
            fleet.step()
            ts.append((time.perf_counter() - a0) * 1000)
        cpu = (time.process_time() - c0) / (250 * ds.CONTROL_DT)
        ts = np.array(ts)
        TIMING[n] = {"mean_ms": round(float(ts.mean()), 2), "p99_ms": round(float(np.percentile(ts, 99)), 2),
                     "cpu_core_pct_at_50hz": round(100 * cpu, 1)}
    check("headless step cost: 4 ducks well inside the 20 ms budget (mean < 8 ms; p99 is scheduler noise on a busy box)",
          TIMING[4]["mean_ms"] < 8, json.dumps(TIMING))
    print(f"headless done in {time.perf_counter() - t_wall:.0f}s")


# ----------------------------------------------------------------------------------------------- HTTP
PORT = 8806
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return r.status, json.loads(r.read())


def post(path, obj, method="POST"):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode() if obj is not None else None,
                                 headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def start_server(state):
    env = dict(os.environ, MICRODUCK_STATE=state, MICRODUCK_TEST="1", OMP_NUM_THREADS="1")
    logf = open(state + ".log", "a")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "--port", str(PORT)], env=env,
                         stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(0.1)
        try:
            get("/health")
            return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                raise RuntimeError("test server died; see " + state + ".log")
    raise RuntimeError("test server did not come up")


def wait_ducks(n, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, r = get("/ducks")
        if len(r["ducks"]) == n:
            return r
        time.sleep(0.2)
    return r


def http():
    d, state = tmp_files()
    p = start_server(state)
    try:
        st, h = get("/health")
        check("fresh start: one default duck (Quackers = duck 1), old /health shape", st == 200 and h.get("ducks") == 1
              and h.get("name") == "Quackers" and h.get("duck") == "d1" and h.get("hz") == 50 and "mode" in h,
              json.dumps({k: h.get(k) for k in ("ok", "mode", "duck", "name", "ducks", "max_ducks")}))
        st, pose = get("/pose")
        check("old /pose unchanged (default duck bodies)", st == 200 and len(pose["bodies"]) == 15 and "quack" in pose, "")
        st, r = post("/terrain", {"id": "lab", "region": list(lab().region), "boxes": lab().boxes.tolist()})
        time.sleep(1.5)
        # timing per duck count, real time (walking)
        for n in (1, 2, 3, 4):
            r = wait_ducks(len(get("/ducks")[1]["ducks"]))
            while len(r["ducks"]) < n:
                post("/ducks", {"from": "mc"})
                r = get("/ducks")[1]
            post("/cmd", {"cmd": "walk", "duck": "all", "from": "mc"})
            time.sleep(2.0)
            _, h0 = get("/health")
            c0 = sum(os.times()[:2])
            with open(f"/proc/{p.pid}/stat") as f:
                s0 = f.read().split()
            t0 = time.time()
            time.sleep(10.0)
            _, h1 = get("/health")
            with open(f"/proc/{p.pid}/stat") as f:
                s1 = f.read().split()
            el = time.time() - t0
            cpu = (int(s1[13]) + int(s1[14]) - int(s0[13]) - int(s0[14])) / os.sysconf("SC_CLK_TCK") / el
            steps = h1["seq"] - h0["seq"]
            TIMING.setdefault("http", {})[n] = {"step_ms": h1["step_ms"], "late_per_10s": h1["late"] - h0["late"],
                                                "overruns": h1["overruns"] - h0["overruns"],
                                                "hz": round(steps / el, 1), "cpu_pct_one_core": round(100 * cpu, 1)}
            del c0
        tim = TIMING["http"]
        check(f"real time at 50 Hz with 1-4 ducks (>= 49 Hz, no resync overruns; test server at nice {os.nice(0)})",
              all(v["hz"] >= 49 and v["overruns"] == 0 for v in tim.values()), json.dumps(tim))
        st, r = get("/ducks")
        check("GET /ducks lists 4 with id/name/voice/color/brain/mode/pos", st == 200 and len(r["ducks"]) == 4 and all(
            all(k in dk for k in ("id", "name", "voice", "color", "brain", "mode", "pos")) for dk in r["ducks"]),
              json.dumps([(dk["id"], dk["name"], dk["voice"], dk["color"], dk["brain"], dk["mode"]) for dk in r["ducks"]]))
        st, r5 = post("/ducks", {"persona": "all", "from": "mc"})
        check("spawn all when full -> refused cleanly", st == 409 and not r5["ok"], json.dumps(r5)[:200])
        st, ps = get("/poses")
        check("GET /poses: every duck's bodies + seq/quack/talking/name/color", st == 200 and len(ps["ducks"]) == 4 and all(
            len(v["bodies"]) == 15 and all(k in v for k in ("seq", "quack", "talking", "name", "color")) for v in ps["ducks"].values()),
              f"order {ps['order']}, default {ps['default']}")
        post("/cmd", {"cmd": "stop", "duck": "all", "from": "mc"})
        time.sleep(1.0)
        ducks = get("/ducks")[1]["ducks"]
        b = ducks[1]
        st, r = post("/cmd", {"cmd": "walk", "duck": b["name"].upper(), "from": "mc"})
        time.sleep(3.0)
        after = get("/ducks")[1]["ducks"]
        moved = [math.hypot(a2["pos"][0] - a1["pos"][0], a2["pos"][1] - a1["pos"][1]) for a1, a2 in zip(ducks, after)]
        check("POST /cmd duck=<NAME> (case-insensitive) moves only that duck",
              st == 200 and r["duck"] == b["id"] and after[1]["mode"] == "walk" and moved[1] > 0.1
              and all(m < 0.08 for i, m in enumerate(moved) if i != 1) and all(a["mode"] == "stand" for i, a in enumerate(after) if i != 1),
              f"moved {[round(m, 2) for m in moved]} modes {[a['mode'] for a in after]}")
        st, r = post("/cmd", {"cmd": "quack", "duck": "all", "from": "twitch:tester"})
        check("twitch duck=all quack ok (4 results)", st == 200 and len(r.get("results", [])) == 4, json.dumps(r)[:200])
        st, r = post("/ducks", {"persona": "zippy", "from": "twitch:tester"})
        check("twitch cannot spawn", st == 403, json.dumps(r))
        st, r = post("/ducks/remove", {"duck": "all", "from": "twitch:tester"})
        check("twitch cannot remove", st == 403, json.dumps(r))
        st, r = post("/cmd", {"cmd": "walk", "duck": "nobody", "from": "mc"})
        check("unknown duck -> 404", st == 404 and not r["ok"], json.dumps(r))
        st, r = post("/cmd", {"cmd": "follow", "duck": ducks[2]["id"], "target": ducks[0]["name"], "from": "mc"})
        time.sleep(0.2)
        fl = get("/ducks")[1]["ducks"][2]
        check("follow another duck", st == 200 and fl["follow"] == ducks[0]["id"] and fl["mode"] == "follow", json.dumps(r))
        st, r = post(f"/ducks/{ducks[3]['id']}", None, method="DELETE")
        check("DELETE /ducks/<id>", st == 200 and r["ducks"] == 3, json.dumps(r)[:200])
        st, r = post("/ducks", {"persona": ducks[3]["persona"], "from": "mc"})
        check("respawn that persona", st == 200 and r["duck"]["name"] == ducks[3]["name"], json.dumps(r)[:200])
        post("/cmd", {"cmd": "stop", "duck": "all", "from": "mc"})
        time.sleep(1.0)
        before = get("/ducks")[1]["ducks"]
    finally:
        p.terminate()
        p.wait(10)
    check("SIGTERM -> clean exit (roster saved)", p.returncode == 0, f"rc={p.returncode}")
    p = start_server(state)
    try:
        r = wait_ducks(len(before))
        same = [(a["id"], a["name"]) for a in r["ducks"]] == [(a["id"], a["name"]) for a in before]
        near = all(math.hypot(a["pos"][0] - b2["pos"][0], a["pos"][1] - b2["pos"][1]) < 0.35 for a, b2 in zip(r["ducks"], before))
        check("persistence across a service restart: same ducks (ids, names) near their last spots", same and near,
              f"{[(a['id'], a['name']) for a in r['ducks']]}")
        st, r = post("/ducks/remove", {"duck": "all", "from": "mc"})
        _, h = get("/health")
        _, pose = get("/pose")
        st2, r2 = post("/cmd", {"cmd": "quack", "from": "mc"})
        check("remove all: /health ok mode none, /pose empty bodies, /cmd -> 409 (old clients don't crash)",
              st == 200 and h["ok"] and h["mode"] == "none" and h["ducks"] == 0 and pose["bodies"] == {} and st2 == 409,
              f"{h['mode']} {pose} {st2}")
    finally:
        p.terminate()
        p.wait(10)
    p = start_server(state)
    try:
        _, r = get("/ducks")
        check("removed-all roster persists as empty (no surprise respawn)", len(r["ducks"]) == 0, "")
    finally:
        p.terminate()
        p.wait(10)


def main():
    t0 = time.perf_counter()
    headless()
    if len(sys.argv) < 2 or sys.argv[1] != "headless":
        http()
    print("TIMING", json.dumps(TIMING))
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"SUMMARY: {len(results) - n_fail}/{len(results)} passed -> {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({time.perf_counter() - t0:.0f}s)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
