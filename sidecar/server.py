#!/usr/bin/env python3
"""Microduck MuJoCo sim sidecar — HTTP API on 127.0.0.1:8796 (see CONTRACT.md, incl. "Multiple ducks").

Several ducks (multi.py Fleet). Every pre-multi endpoint keeps its shape and acts on the default duck (the first one in
the roster), so an older mod keeps working; new: GET /ducks, POST /ducks, POST /ducks/remove, DELETE /ducks/<id>,
GET /poses and an optional "duck" field on POST /cmd.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import duck_sim as ds  # noqa: E402
import model_export  # noqa: E402
import terrain as terrain_mod  # noqa: E402
import brains  # noqa: E402  (community brains: hot-swappable policies, see brains.py)
import multi  # noqa: E402  (several ducks: Fleet, personas, roster)

HOST = "127.0.0.1"


LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")


def foreign_request(headers):
    """Why a request must be refused, or None. Only local programs (the mod, the tools) may drive the
    sim: a web page in any browser on this machine, including the in-game web walls, can also reach
    127.0.0.1. Browsers always send Origin on cross-site POSTs, and a DNS-rebinding page carries its
    own host name in Host; the mod and the Python tools send neither."""
    if headers.get("Origin"):
        return "requests from web pages are refused"
    host = (headers.get("Host") or "").strip()
    name = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0]
    if name.lower() not in LOOPBACK_NAMES:
        return "only 127.0.0.1 / localhost may be used as the Host"
    return None
PORT = 8796
TWITCH_ALLOWED = {"walk", "stop", "sit", "stand", "follow", "quack", "turn", "brain", "trick"}
POLICY_FILES = {"velstand": "velstand.onnx", "sitstand": "alpha_sitstand.onnx"}
DUCK_SEL_RE = re.compile(r"^[A-Za-z0-9 _*-]{1,24}$")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r5(x):
    return round(float(x), 5)


class Service:
    def __init__(self):
        self.fleet = multi.Fleet(log=log, autoload=False)
        self.lock = self.fleet.lock
        t = time.time()
        g = ds.GHOSTS                      # /model.json from a ghost-free robot (same cache signature as before)
        ds.GHOSTS = 0
        try:
            m0, _, _ = ds.load_model(False)
        finally:
            ds.GHOSTS = g
        model, ids, ntris = model_export.load_or_build(m0, os.path.join(HERE, "model_cache.json"))
        del m0
        self.model_bytes = json.dumps(model, separators=(",", ":")).encode()
        self.body_ids = ids
        self.body_names = [b["name"] for b in model["bodies"]]
        self.ntris = ntris
        log(f"model.json: {len(self.body_names)} bodies, {ntris} tris, {len(self.model_bytes)} bytes ({time.time() - t:.1f}s)")
        self.snap_lock = threading.Lock()
        self.seq = 0
        self.pose_bytes = b"{}"
        self.poses_bytes = b"{}"
        self.health = {}
        self.overruns = 0
        self.stop = threading.Event()
        self.late = 0          # control steps that finished after their 20 ms deadline
        self.step_ms = 0.0     # EMA of one fleet step (all ducks) wall time
        self.test = os.environ.get("MICRODUCK_TEST") == "1"
        self.terrain = multi.FleetTerrain(self.fleet, log)   # POST /terrain rebuild worker (every duck)
        self.fleet.terrain_mgr = self.terrain
        self.fleet.load_roster()
        with self.lock:
            self._snapshot()

    @property
    def sim(self):  # the default duck's sim (old single-duck code paths)
        d = self.fleet.default()
        return d.sim if d else None

    # -- called with self.lock held
    def _bodies(self, s):
        d = s.data
        ids = s.render_ids(self.body_names, self.body_ids)
        arr = np.round(np.hstack([d.xpos[ids], d.xquat[ids]]), 5).tolist()   # vectorised: ~4x cheaper than per-float
        return dict(zip(self.body_names, arr))

    def _snapshot(self):
        self.seq += 1
        ducks = self.fleet.ducks
        per = {}
        for d in ducks:
            s = d.sim
            per[d.id] = {"name": d.name, "color": d.persona["color"], "seq": d.seq, "t": round(s.t, 3),
                         "mode": s.public_mode(), "fallen": bool(s.fallen), "quack": s.quack,
                         "quack_pitch": d.persona["quack"]["pitch"], "talking": ds.talk_mod.talking(s),
                         "bodies": self._bodies(s)}
        common = {"late": self.late, "step_ms": round(self.step_ms, 3), "overruns": self.overruns, "seq": self.seq,
                  "terrain": self.terrain.brief(), "ducks": len(ducks), "max_ducks": self.fleet.max,
                  "names": [d.name for d in ducks], "player": self.fleet.player}
        if self.test:
            common["test"] = True
        if ducks:
            d0 = ducks[0]
            s = d0.sim
            p0 = per[d0.id]
            pose = {"seq": self.seq, "t": p0["t"], "mode": p0["mode"], "fallen": p0["fallen"], "quack": s.quack,
                    "duck": d0.id, "bodies": p0["bodies"]}
            health = {"ok": True, "mode": p0["mode"], "fallen": p0["fallen"], "t": p0["t"],
                      "policy": POLICY_FILES[s.policy], "hz": 50, "leash": s.leash, "quack": s.quack,
                      "resets": s.resets, "falls": s.falls, "pos": [round(float(x), 3) for x in s.pos()],
                      "talking": p0["talking"], "talks": ds.talk_mod.count(s), "duck": d0.id, "name": d0.name,
                      **s.brain_info(), **common}
        else:
            pose = {"seq": self.seq, "t": 0.0, "mode": "none", "fallen": False, "quack": 0, "duck": None, "bodies": {}}
            health = {"ok": True, "mode": "none", "fallen": False, "t": 0.0, "policy": None, "hz": 50,
                      "leash": self.fleet.leash, "quack": 0, "resets": 0, "falls": 0, "pos": None, "talking": False,
                      "talks": 0, "duck": None, "name": None, "brain": None, "trick": None, "robot": None, **common}
        poses = {"seq": self.seq, "default": ducks[0].id if ducks else None, "order": [d.id for d in ducks],
                 "ducks": per}
        b = json.dumps(pose, separators=(",", ":")).encode()
        bb = json.dumps(poses, separators=(",", ":")).encode()
        with self.snap_lock:
            self.pose_bytes = b
            self.poses_bytes = bb
            self.health = health

    def run(self):
        dt = ds.CONTROL_DT
        nxt = time.monotonic()
        last_save_check = 0.0
        while not self.stop.is_set():
            with self.lock:
                t0 = time.perf_counter()
                try:
                    self.fleet.step()
                except Exception as e:  # noqa: BLE001  (never let one duck kill the loop)
                    log(f"[sim] step failed: {e!r}")
                self.step_ms += 0.02 * ((time.perf_counter() - t0) * 1000 - self.step_ms)
                self._snapshot()
            nxt += dt
            now = time.monotonic()
            if now - last_save_check > 1.0:
                last_save_check = now
                self.fleet.maybe_save()
            delay = nxt - time.monotonic()
            if delay < 0:
                self.late += 1
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:  # fell far behind (suspend/CPU starvation): resync, don't spiral
                self.overruns += 1
                nxt = time.monotonic()

    def get_pose(self):
        with self.snap_lock:
            return self.pose_bytes

    def get_poses(self):
        with self.snap_lock:
            return self.poses_bytes

    def get_health(self):
        with self.snap_lock:
            return dict(self.health)

    def ducks_public(self):
        with self.lock:
            f = self.fleet
            used = {d.persona["id"] for d in f.ducks}
            return {"ok": True, "default": f.ducks[0].id if f.ducks else None, "max": f.max,
                    "ducks": [d.public() for d in f.ducks], "player": f.player, "leash": f.leash,
                    "scale": self.terrain.current.meta.get("scale") if self.terrain.current is not None else None,
                    "personas": [{"id": p["id"], "name": p["name"], "voice": p["voice"], "color": p["color"],
                                  "brain": p["brain"], "trick": p["trick"], "tagline": p.get("tagline", ""),
                                  "used": p["id"] in used} for p in f.personas]}


SERVICE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "microduck-sim/1.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the journal quiet (pose is polled at 20+ Hz)
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self, limit=65536):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
        obj = json.loads(raw or b"{}")
        if not isinstance(obj, dict):
            raise ValueError("body must be a JSON object")
        return obj

    def do_GET(self):
        why = foreign_request(self.headers)
        if why:
            return self._send(403, {"ok": False, "error": why})
        u = urllib.parse.urlsplit(self.path)
        path, qs = u.path, urllib.parse.parse_qs(u.query)
        if path == "/health":
            return self._send(200, SERVICE.get_health())
        if path == "/pose":
            return self._send(200, SERVICE.get_pose())
        if path == "/poses":
            return self._send(200, SERVICE.get_poses())
        if path == "/ducks":
            return self._send(200, SERVICE.ducks_public())
        if path == "/model.json":
            return self._send(200, SERVICE.model_bytes)
        if path == "/terrain":
            return self._send(200, SERVICE.terrain.summary())
        if path == "/brains":  # community brains catalogue + that duck's current brain (brains.py)
            with SERVICE.lock:
                ducks, err = SERVICE.fleet.resolve((qs.get("duck") or [None])[0])
                if err or len(ducks) != 1:
                    return self._send(404, {"ok": False, "error": err or "pick one duck"})
                cat = ducks[0].sim.catalogue_public()
                cat["duck"] = ducks[0].id
            return self._send(200, cat)
        return self._send(404, {"ok": False, "error": "not found"})

    def do_DELETE(self):
        why = foreign_request(self.headers)
        if why:
            return self._send(403, {"ok": False, "error": why})
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith("/ducks/"):
            sel = urllib.parse.unquote(path[len("/ducks/"):])
            return self._remove(sel, "?")
        return self._send(404, {"ok": False, "error": "not found"})

    def _remove(self, sel, src):
        if not sel or not DUCK_SEL_RE.match(sel):
            return self._send(400, {"ok": False, "error": "need \"duck\": id, name or \"all\""})
        if src.startswith("twitch:"):
            return self._send(403, {"ok": False, "error": "remove is not allowed from twitch"})
        removed, err = SERVICE.fleet.remove(sel)
        if err:
            return self._send(404, {"ok": False, "error": err})
        log(f"ducks remove {sel!r} from={src} -> {[d['name'] for d in removed]}")
        return self._send(200, {"ok": True, "removed": removed, "ducks": len(SERVICE.fleet.ducks)})

    def do_POST(self):
        why = foreign_request(self.headers)
        if why:
            return self._send(403, {"ok": False, "error": why})
        path = urllib.parse.urlsplit(self.path).path
        try:
            body = self._json_body(4 << 20 if path == "/terrain" else 65536)
        except Exception as e:
            return self._send(400, {"ok": False, "error": f"bad json: {e}"})
        fleet = SERVICE.fleet
        src = str(body.get("from", "?"))
        if path == "/ducks":       # spawn: {"persona": id|name|"all"} (or "name"); MC/voice only
            if src.startswith("twitch:"):
                return self._send(403, {"ok": False, "error": "spawn is not allowed from twitch"})
            sel = body.get("persona", body.get("name"))
            sel = None if sel is None else str(sel).strip()
            if sel and not DUCK_SEL_RE.match(sel):
                return self._send(400, {"ok": False, "error": "bad persona"})
            if sel and sel.lower() == "all":
                spawned, errs = fleet.spawn_all()
                log(f"ducks spawn all from={src} -> {[d['name'] for d in spawned]} {errs if errs else ''}")
                return self._send(200 if spawned or not errs else 409,
                                  {"ok": bool(spawned) or not errs, "spawned": spawned, "errors": errs,
                                   "ducks": len(fleet.ducks), **({"error": "; ".join(errs)} if errs and not spawned else {})})
            d, err = fleet.spawn(sel or None)
            if err:
                return self._send(409, {"ok": False, "error": err})
            log(f"ducks spawn {sel!r} from={src} -> {d['id']} {d['name']}")
            return self._send(200, {"ok": True, "duck": d, "spawned": [d], "ducks": len(fleet.ducks)})
        if path == "/ducks/remove":
            return self._remove(str(body.get("duck", body.get("name", ""))).strip(), src)
        if path == "/cmd":
            cmd = str(body.get("cmd", "")).strip().lower()
            if cmd in ("left", "right"):  # chat aliases
                body["dir"] = cmd
                cmd = "turn"
            if src.startswith("twitch:") and cmd not in TWITCH_ALLOWED:
                return self._send(403, {"ok": False, "error": f"cmd {cmd!r} not allowed from twitch"})
            sel = body.get("duck")
            if sel is not None:
                sel = str(sel).strip()
                if sel and not DUCK_SEL_RE.match(sel):
                    return self._send(400, {"ok": False, "error": "bad \"duck\" (id, name or \"all\")"})
            kw = {}
            try:
                for k in ("vx", "vy", "wz"):
                    if k in body and body[k] is not None:
                        v = float(body[k])
                        if not math.isfinite(v):
                            raise ValueError(k)
                        kw[k] = v
            except (TypeError, ValueError):
                return self._send(400, {"ok": False, "error": "vx/vy/wz must be numbers"})
            if "dir" in body:
                kw["dir"] = str(body["dir"]).lower()
            if cmd == "talk":  # speaking animation (talk.py): {"on":bool} or {"duration":s}
                for k in ("on", "duration"):
                    if k in body:
                        kw[k] = body[k]
            if cmd == "turn" and kw.get("dir") not in ("left", "right"):
                return self._send(400, {"ok": False, "error": "turn needs \"dir\":\"left\"|\"right\""})
            if cmd in ("brain", "trick"):  # community brains (brains.py): {"name": id}
                name = str(body.get("name", "")).strip().lower()
                if not re.match(brains.ID_RE, name):
                    return self._send(400, {"ok": False, "error": f"{cmd} needs \"name\": [a-z0-9-]{{1,24}} (see GET /brains)"})
                kw["name"] = name
            if cmd == "follow" and body.get("target"):
                t = str(body["target"]).strip()
                if not DUCK_SEL_RE.match(t):
                    return self._send(400, {"ok": False, "error": "bad follow target"})
                kw["target"] = t
            if cmd not in ("walk", "stop", "sit", "stand", "follow", "quack", "turn", "reset", "brain", "trick", "talk"):
                return self._send(400, {"ok": False, "error": f"unknown cmd {cmd!r}"})
            with SERVICE.lock:
                targets, rerr = fleet.resolve(sel)
                if rerr:
                    code = 409 if not fleet.ducks else 404
                    return self._send(code, {"ok": False, "error": rerr})
                ok, mode, results, err = fleet.command(sel, cmd, **kw)
            who = ",".join(r["name"] for r in results)
            if cmd != "talk" or not ok:
                log(f"cmd {cmd} {kw if kw else ''} duck={who} from={src} -> ok={ok} mode={mode}{' err=' + err if err else ''}")
            out = {"ok": ok, "mode": mode, "duck": results[0]["id"] if len(results) == 1 else "all",
                   "name": results[0]["name"] if len(results) == 1 else "all"}
            if len(results) > 1 or (sel or "").lower() in ("all", "*", "everyone", "ducks"):
                out["results"] = results
            if ok:
                return self._send(200, out)
            out["error"] = err
            return self._send(400, out)
        if path == "/player":
            try:
                x, y = float(body["x"]), float(body["y"])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return self._send(400, {"ok": False, "error": "need numeric x and y"})
            with SERVICE.lock:
                fleet.set_player(x, y)
            return self._send(200, {"ok": True})
        if path == "/terrain":  # Minecraft blocks around the anchor as boxes (terrain.py, CONTRACT.md)
            try:
                r = SERVICE.terrain.submit(body)
            except terrain_mod.TerrainError as e:
                return self._send(400, {"ok": False, "error": str(e)})
            if r.get("changed"):
                log(f"terrain {r['boxes']} boxes hash={r['hash']} id={body.get('id')} queued")
            return self._send(200, r)
        if path == "/config":
            if "leash" in body:
                try:
                    r = float(body["leash"])
                    if not (math.isfinite(r) and 0.3 <= r <= 100):
                        raise ValueError
                except (TypeError, ValueError):
                    return self._send(400, {"ok": False, "error": "leash must be a number in [0.3, 100]"})
                with SERVICE.lock:
                    fleet.set_leash(r)
                log(f"config leash={r}")
            return self._send(200, {"ok": True, "leash": fleet.leash})
        return self._send(404, {"ok": False, "error": "not found"})


def main():
    global SERVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    SERVICE = Service()
    th = threading.Thread(target=SERVICE.run, name="sim", daemon=True)
    th.start()
    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    httpd.daemon_threads = True
    import signal
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))   # systemctl stop/restart: save the roster (finally)
    log(f"listening on http://{HOST}:{args.port}  scene={os.path.basename(ds.SCENE_XML)} ducks={len(SERVICE.fleet.ducks)}"
        f"/{SERVICE.fleet.max} state={SERVICE.fleet.state_path}")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        SERVICE.stop.set()
        try:
            SERVICE.fleet.save_roster()
        except Exception:  # noqa: BLE001
            pass
        httpd.server_close()


if __name__ == "__main__":
    main()
