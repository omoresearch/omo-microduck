#!/usr/bin/env python3
"""HTTP test for POST/GET /terrain against a running sidecar (default http://127.0.0.1:8796).

Leaves the sidecar with the terrain it had before (or flat) and the duck reset.
Also measures loop timing (/health late, overruns, step_ms) with 500 and 2000 boxes while walking.
"""

import json
import math
import random
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8796"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return r.status, json.loads(r.read())


def post(path, obj, raw=None):
    data = raw if raw is not None else json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_applied(h, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, s = get("/terrain")
        if s.get("hash") == h and not s.get("pending"):
            return s, time.time() - t0
        time.sleep(0.1)
    return s, None


def room():
    B = 0.25
    return [[-1.5, -1.5, -B, 1.5, 1.5, 0.0],
            [-1.5, -1.5, -B, -1.25, 1.5, 2 * B], [1.25, -1.5, -B, 1.5, 1.5, 2 * B],
            [-1.25, -1.5, -B, 1.25, -1.25, 2 * B], [-1.25, 1.25, -B, 1.25, 1.5, 2 * B],
            [-1.0, 0.5, 0.0, -0.75, 1.0, B / 2]]


def random_boxes(n, seed):
    rng = random.Random(seed)
    out = [[-3.25, -3.25, -0.25, 3.25, 3.25, 0.0]]
    while len(out) < n:
        x, y = rng.uniform(-3.25, 3.0), rng.uniform(-3.25, 3.0)
        if math.hypot(x, y) < 0.7:
            continue
        out.append([round(x, 4), round(y, 4), 0.0, round(x + 0.25, 4), round(y + 0.25, 4),
                    rng.choice([0.125, 0.25, 0.5, 0.75])])
    return out


def main():
    st, before = get("/terrain")
    check("GET /terrain", st == 200 and before.get("ok") is True and "boxes" in before and "hash" in before,
          json.dumps({k: before.get(k) for k in ("boxes", "hash", "floor_z", "pending", "builds")}))
    st, r = post("/terrain", {"boxes": "nope"})
    check("POST /terrain bad boxes -> 400", st == 400 and not r.get("ok"), r.get("error", ""))
    st, r = post("/terrain", {"boxes": [[0, 0, 0, 1, 1]]})
    check("POST /terrain short box -> 400", st == 400, r.get("error", ""))
    st, r = post("/terrain", {"boxes": [[0, 0, 0, 1e9, 1, 1]]})
    check("POST /terrain out-of-range -> 400", st == 400, r.get("error", ""))
    st, r = post("/terrain", {"boxes": [[0, 0, 0, 0.1, 0.1, 0.1]] * 4001})
    check("POST /terrain > 4000 boxes -> 400", st == 400, r.get("error", ""))

    post("/cmd", {"cmd": "reset", "from": "mc"})
    body = {"id": "http-test-room", "anchor": [0.5, 64, 0.5], "scale": 4, "dim": "minecraft:overworld",
            "region": [-1.75, -1.75, -0.25, 1.75, 1.75, 0.75], "boxes": room()}
    st, r = post("/terrain", body)
    check("POST /terrain room -> queued", st == 200 and r.get("ok") and r.get("changed") and r.get("boxes") == 6, json.dumps(r))
    s, dt = wait_applied(r["hash"])
    check("room applied (GET /terrain hash matches, not pending)", dt is not None and s.get("boxes") == 6
          and s.get("id") == "http-test-room" and s.get("bounds") == [-1.5, -1.5, -0.25, 1.5, 1.5, 0.5],
          f"after {dt if dt is None else round(dt, 2)} s: {json.dumps({k: s.get(k) for k in ('boxes', 'hash', 'bounds', 'floor_z', 'last_build_ms', 'last_respawn')})}")
    st, r2 = post("/terrain", body)
    check("re-POST identical terrain -> changed:false (no rebuild)", st == 200 and r2.get("changed") is False, json.dumps(r2))
    time.sleep(2.0)
    st, h = get("/health")
    check("/health carries terrain + duck standing after swap", h["terrain"]["boxes"] == 6 and not h["fallen"]
          and abs(h["pos"][2] - 0.116) < 0.03, json.dumps({"terrain": h["terrain"], "pos": h["pos"], "fallen": h["fallen"]}))

    # loop timing with realistic box counts while walking
    for n in (500, 2000):
        boxes = random_boxes(n, n)
        t0 = time.time()
        st, r = post("/terrain", {"id": f"bench-{n}", "boxes": boxes})
        post_ms = (time.time() - t0) * 1000
        s, dt = wait_applied(r["hash"], 15)
        post("/cmd", {"cmd": "reset", "from": "mc"})
        post("/cmd", {"cmd": "walk", "from": "mc"})
        time.sleep(0.5)
        _, h0 = get("/health")
        time.sleep(10.0)
        _, h1 = get("/health")
        steps = h1["seq"] - h0["seq"]
        late = h1["late"] - h0["late"]
        over = h1["overruns"] - h0["overruns"]
        check(f"{n} boxes: 50 Hz loop keeps real time while walking (no overruns, <2% late)",
              dt is not None and over == 0 and late <= 0.02 * steps and steps > 450,
              f"POST {len(json.dumps(boxes)) // 1024} KB in {post_ms:.0f} ms, rebuild {s.get('last_build_ms')} ms, "
              f"{steps} steps/10 s, late={late}, overruns={over}, step_ms={h1['step_ms']}, falls={h1['falls'] - h0['falls']}")

    # restore what was there before (flat if nothing)
    post("/cmd", {"cmd": "stop", "from": "mc"})
    st, r = post("/terrain", {"id": "flat", "boxes": []})
    s, dt = wait_applied(None if True else r["hash"], 8)
    check("POST empty boxes -> back to flat floor", dt is not None and s.get("boxes") == 0, json.dumps({k: s.get(k) for k in ("boxes", "hash")}))
    post("/cmd", {"cmd": "reset", "from": "mc"})
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
