#!/usr/bin/env python3
"""HTTP smoke test against the running sidecar (default http://127.0.0.1:8796)."""

import json
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


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multi_ducks(names):
    """Multiple ducks (CONTRACT.md "Multiple ducks"). Read-only checks always; spawn/remove only against a test
    instance (/health "test": true, MICRODUCK_TEST=1) or with --multi, so the live lab isn't changed."""
    st, h = get("/health")
    st, r = get("/ducks")
    check("GET /ducks", st == 200 and r.get("ok") and len(r["ducks"]) == h.get("ducks") and r["default"] == h.get("duck")
          and all(k in r["ducks"][0] for k in ("id", "name", "voice", "color", "brain", "mode", "pos")),
          json.dumps([(d["id"], d["name"], d["voice"], d["mode"]) for d in r["ducks"]]))
    st, ps = get("/poses")
    check("GET /poses (every duck, same body format as /pose)", st == 200 and set(ps["ducks"]) == {d["id"] for d in r["ducks"]}
          and all(set(v["bodies"]) == set(names) and "quack" in v and "talking" in v and "seq" in v for v in ps["ducks"].values()),
          f"{len(ps['ducks'])} ducks, order {ps['order']}")
    st, rr = post("/cmd", {"cmd": "quack", "duck": r["ducks"][0]["name"].lower(), "from": "mc"})
    check("POST /cmd with \"duck\": <name>", st == 200 and rr.get("duck") == r["ducks"][0]["id"], json.dumps(rr))
    if not (h.get("test") or "--multi" in sys.argv):
        print("      (live sidecar: spawn/remove checks skipped; run against a MICRODUCK_TEST=1 instance or pass --multi)")
        return
    n0 = len(r["ducks"])
    if n0 >= r["max"]:  # full roster (e.g. left by another test): make room so spawn can be tested
        post("/ducks/remove", {"duck": r["ducks"][-1]["id"], "from": "mc"})
        n0 -= 1
    st, sp = post("/ducks", {"from": "mc"})
    ok_sp = st == 200 and sp.get("ok")
    check("POST /ducks spawns the next unused persona", ok_sp and sp["ducks"] == n0 + 1, json.dumps(sp)[:240])
    if not ok_sp:
        return
    new = sp["duck"]
    time.sleep(1.0)
    q_before = {k: v["quack"] for k, v in get("/poses")[1]["ducks"].items()}
    st, rr = post("/cmd", {"cmd": "quack", "duck": new["id"], "from": "twitch:tester"})
    time.sleep(0.2)
    q_after = {k: v["quack"] for k, v in get("/poses")[1]["ducks"].items()}
    check("quack on the new duck only", st == 200 and q_after[new["id"]] == q_before[new["id"]] + 1
          and all(q_after[k] == q_before[k] for k in q_before if k != new["id"]), f"{q_before} -> {q_after}")
    st, rr = post("/cmd", {"cmd": "stop", "duck": "all", "from": "mc"})
    check("POST /cmd duck=all", st == 200 and len(rr.get("results", [])) == n0 + 1, json.dumps(rr)[:200])
    st, rr = post("/ducks", {"from": "twitch:tester"})
    check("twitch cannot spawn (403)", st == 403, json.dumps(rr))
    st, rr = post("/ducks/remove", {"duck": new["name"], "from": "mc"})
    check("POST /ducks/remove", st == 200 and rr["ducks"] == n0, json.dumps(rr)[:200])
    st, rr = post("/cmd", {"cmd": "quack", "duck": new["id"], "from": "mc"})
    check("removed duck is gone (404)", st == 404, json.dumps(rr))


def main():
    st, h = get("/health")
    check("GET /health", st == 200 and h.get("ok") is True and h.get("hz") == 50 and "mode" in h and "policy" in h,
          json.dumps({k: h[k] for k in ("ok", "mode", "fallen", "t", "policy", "hz")}))

    t0 = time.time()
    st, m = get("/model.json")
    names = [b["name"] for b in m["bodies"]]
    ntris = sum(len(me["tris"]) // 3 for b in m["bodies"] for me in b["meshes"])
    nverts = sum(len(me["verts"]) // 3 for b in m["bodies"] for me in b["meshes"])
    idx_ok = all(max(me["tris"]) < len(me["verts"]) // 3 for b in m["bodies"] for me in b["meshes"])
    check("GET /model.json", st == 200 and m.get("version") == 1 and 0 < ntris <= 26000 and idx_ok,
          f"{len(names)} bodies, {ntris} tris, {nverts} verts, fetched in {time.time() - t0:.2f}s")
    meshes = [me for b in m["bodies"] for me in b["meshes"]]
    ext_ok = all(len(me.get("normals", me["verts"])) == len(me["verts"]) and len(me.get("colors", me["verts"])) == len(me["verts"])
                 for me in meshes)
    unit = [sum(x * x for x in me["normals"][i:i + 3]) for me in meshes if "normals" in me for i in range(0, len(me["normals"]), 3)]
    n_ext = sum("normals" in me and "colors" in me for me in meshes)
    check("model.json optional normals/colors per vertex", ext_ok and all(0.98 < u < 1.02 for u in unit),
          f"{n_ext}/{len(meshes)} meshes with normals+colors, {sum(bool(me.get('cull')) for me in meshes)} cull")
    print("      bodies:", ", ".join(names))

    st, p1 = get("/pose")
    time.sleep(1.0)
    st2, p2 = get("/pose")
    dseq = p2["seq"] - p1["seq"]
    dt = p2["t"] - p1["t"]
    check("GET /pose has every model body", set(p1["bodies"]) == set(names) and all(len(v) == 7 for v in p1["bodies"].values()),
          f"{len(p1['bodies'])} bodies, trunk_base={p1['bodies'].get('trunk_base')}")
    check("pose seq advances in real time (~50/s)", 40 <= dseq <= 60 and 0.8 <= dt <= 1.2,
          f"seq +{dseq} and sim t +{dt:.2f}s over 1.0 s wall")

    q0 = p2["quack"]
    cmds = [
        ({"cmd": "reset", "from": "mc"}, "stand"),
        ({"cmd": "walk", "from": "mc"}, "walk"),
        ({"cmd": "walk", "vx": 0.1, "wz": 0.3, "from": "mc"}, "walk"),
        ({"cmd": "stop", "from": "twitch:tester"}, "stand"),
        ({"cmd": "turn", "dir": "left", "from": "mc"}, "turn"),
        ({"cmd": "turn", "dir": "right", "from": "twitch:tester"}, "turn"),
        ({"cmd": "quack", "from": "twitch:tester"}, None),
        ({"cmd": "sit", "from": "mc"}, "sit"),
        ({"cmd": "stand", "from": "mc"}, "stand"),
        ({"cmd": "follow", "from": "mc"}, "follow"),
    ]
    for body, want in cmds:
        st, r = post("/cmd", body)
        ok = st == 200 and r.get("ok") is True and (want is None or r.get("mode") == want)
        check(f"POST /cmd {body['cmd']}{' ' + body.get('dir', '') if 'dir' in body else ''}", ok, json.dumps(r))
        time.sleep(1.5 if body["cmd"] in ("sit", "walk", "turn") else 0.3)
    st, r = post("/player", {"x": 0.5, "y": 0.3})
    check("POST /player", st == 200 and r.get("ok"), json.dumps(r))
    st, r = post("/config", {"leash": 2.0})
    check("POST /config leash", st == 200 and r.get("leash") == 2.0, json.dumps(r))
    st, r = post("/config", {"leash": 2.5})
    st, r = post("/cmd", {"cmd": "reset", "from": "twitch:tester"})
    check("twitch cannot reset", st == 403 and r.get("ok") is False, json.dumps(r))
    st, r = post("/cmd", {"cmd": "fly", "from": "mc"})
    check("unknown cmd rejected", st == 400 and r.get("ok") is False, json.dumps(r))
    st, r = post("/cmd", {"cmd": "turn", "from": "mc"})
    check("turn without dir rejected", st == 400 and r.get("ok") is False, json.dumps(r))
    st, r = post("/cmd", {"cmd": "talk", "from": "twitch:tester"})
    check("twitch cannot talk", st == 403, json.dumps(r))
    st, r = post("/cmd", {"cmd": "talk", "on": True, "from": "voice"})
    time.sleep(0.3)
    _, ht = get("/health")
    check("POST /cmd talk on -> health.talking", st == 200 and r.get("ok") and ht.get("talking") is True, json.dumps(r))
    time.sleep(2.0)
    st, r = post("/cmd", {"cmd": "talk", "on": False, "from": "voice"})
    time.sleep(0.3)
    _, ht = get("/health")
    check("POST /cmd talk off; still upright", st == 200 and ht.get("talking") is False and ht["fallen"] is False,
          json.dumps({k: ht.get(k) for k in ("talking", "talks", "fallen", "mode")}))
    time.sleep(0.2)
    st, p3 = get("/pose")
    check("quack counter increased", p3["quack"] == q0 + 1, f"{q0} -> {p3['quack']}")

    # ---- community brains (brains.py)
    post("/cmd", {"cmd": "reset", "from": "mc"})
    time.sleep(1.0)
    st, cat = get("/brains")
    ids = {b["id"]: b for b in cat.get("brains", [])}
    check("GET /brains", st == 200 and cat.get("brain") == "default" and 6 <= len(ids) <= 12
          and all(b.get("repo") and b.get("kind") in ("perpetual", "episodic") for b in ids.values()),
          f"{len(ids)}: {sorted(ids)}")
    _, h = get("/health")
    check("/health has brain/trick/robot", h.get("brain") == "default" and "trick" in h and h.get("robot") == "default",
          json.dumps({k: h.get(k) for k in ("brain", "trick", "robot")}))
    walker = next((i for i, b in ids.items() if b["kind"] == "perpetual" and b["variant"] == "default"), None)
    roller = next((i for i, b in ids.items() if b["kind"] == "perpetual" and b["variant"] == "rollers"), None)
    trick = "roulade" if "roulade" in ids else next((i for i, b in ids.items() if b["kind"] == "episodic"), None)
    if walker:
        st, r = post("/cmd", {"cmd": "brain", "name": walker, "from": "twitch:tester"})
        time.sleep(3.0)
        _, h = get("/health")
        check(f"twitch brain {walker} -> walking with it", st == 200 and r.get("mode") == "walk" and h.get("brain") == walker
              and not h["fallen"], json.dumps({k: h.get(k) for k in ("brain", "mode", "pos", "fallen")}))
    if roller:
        _, p_before = get("/pose")
        st, r = post("/cmd", {"cmd": "brain", "name": roller, "from": "mc"})
        time.sleep(0.15)   # health/pose snapshots are published by the sim thread on its next step
        _, h = get("/health")
        _, p_after = get("/pose")
        tb0, tb1 = p_before["bodies"]["trunk_base"], p_after["bodies"]["trunk_base"]
        check(f"brain {roller}: robot swapped to rollers, pose continuous", st == 200 and h.get("robot") == "rollers"
              and abs(tb0[0] - tb1[0]) < 0.15 and abs(tb0[1] - tb1[1]) < 0.15
              and set(p_after["bodies"]) == set(names), f"{tb0[:3]} -> {tb1[:3]} robot={h.get('robot')}")
        time.sleep(3.0)
        _, h = get("/health")
        check(f"{roller} skating, upright", h.get("brain") == roller and not h["fallen"],
              json.dumps({k: h.get(k) for k in ("brain", "robot", "mode", "pos")}))
    if trick:
        st, r = post("/cmd", {"cmd": "trick", "name": trick, "from": "twitch:tester"})
        time.sleep(0.15)
        _, h = get("/health")
        check(f"twitch trick {trick} -> mode trick (skates off)", st == 200 and r.get("mode") == "trick"
              and h.get("trick") == trick and h.get("robot") == "default", json.dumps(r))
        time.sleep(float(ids[trick].get("duration_s") or 1.0) + 4.0)
        _, h = get("/health")
        check(f"trick {trick} hands back to velstand standing", h.get("trick") is None and h["mode"] == "stand"
              and not h["fallen"], json.dumps({k: h.get(k) for k in ("trick", "mode", "fallen", "falls")}))
    st, r = post("/cmd", {"cmd": "brain", "name": "nope-x", "from": "mc"})
    check("unknown brain -> 400", st == 400 and "unknown" in r.get("error", ""), json.dumps(r))
    st, r = post("/cmd", {"cmd": "trick", "name": "../../etc", "from": "twitch:tester"})
    check("bad brain id -> 400", st == 400 and r.get("ok") is False, json.dumps(r))
    st, r = post("/cmd", {"cmd": "brain", "from": "mc"})
    check("brain without name -> 400", st == 400, json.dumps(r))
    st, r = post("/cmd", {"cmd": "brain", "name": "default", "from": "mc"})
    time.sleep(0.15)
    _, h = get("/health")
    check("brain default", st == 200 and h.get("brain") == "default" and h.get("robot") == "default", json.dumps(r))
    time.sleep(3)
    st, h = get("/health")
    post("/cmd", {"cmd": "stop", "from": "mc"})
    check("still upright after the command tour", h["fallen"] is False, json.dumps(h))
    multi_ducks(names)
    n_fail = results.count(False)
    print(f"SUMMARY: {len(results) - n_fail}/{len(results)} passed -> {'PASS' if n_fail == 0 else 'FAIL'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
