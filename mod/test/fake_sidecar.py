#!/usr/bin/env python3
"""Tiny stand-in for the microduck-sim sidecar (CONTRACT.md API) used only by the mod's test harness.

Serves a static 2-body model (a box "trunk" and a tetra "head"), a pose whose trunk yaws slowly and whose
quack counter increments every 2 s, and records the last POST /cmd and /player bodies (GET /_last).
Usage: python3 fake_sidecar.py [port]   (default 8797, so it never collides with the real sidecar on 8796)
"""
import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
T0 = time.time()
LAST = {"cmd": None, "player": None, "player_count": 0}

B = 0.05
BOX_V = [-B, -B, 0, B, -B, 0, B, B, 0, -B, B, 0, -B, -B, 2 * B, B, -B, 2 * B, B, B, 2 * B, -B, B, 2 * B]
BOX_T = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
TET_V = [0, 0, 0, 0.04, 0, 0, 0, 0.04, 0, 0, 0, 0.04]
TET_T = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
MODEL = {"version": 1, "bodies": [
    {"name": "trunk", "meshes": [{"rgba": [0.95, 0.85, 0.2, 1], "verts": BOX_V, "tris": BOX_T}]},
    {"name": "head", "meshes": [{"rgba": [1, 0.5, 0.1, 1], "verts": TET_V, "tris": TET_T}]},
]}


def pose():
    t = time.time() - T0
    yaw = 0.5 * t
    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return {"seq": int(t * 50), "t": round(t, 3), "mode": "stand", "fallen": False, "quack": int(t // 2),
            "bodies": {"trunk": [0.0, 0.0, 0.1] + q, "head": [0.03, 0.0, 0.2] + q}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._send({"ok": True, "mode": "stand", "fallen": False, "t": time.time() - T0, "policy": "fake", "hz": 50})
        elif self.path == "/model.json":
            self._send(MODEL)
        elif self.path == "/pose":
            self._send(pose())
        elif self.path == "/_last":
            self._send(LAST)
        else:
            self._send({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/cmd":
            LAST["cmd"] = body
            c = body.get("cmd")
            if c in ("walk", "stop", "sit", "stand", "follow", "quack", "turn", "reset"):
                self._send({"ok": True, "mode": c})
            else:
                self._send({"ok": False, "error": "unknown cmd"})
        elif self.path == "/player":
            LAST["player"] = body
            LAST["player_count"] += 1
            self._send({"ok": True})
        else:
            self._send({"ok": False, "error": "not found"}, 404)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
