#!/usr/bin/env python3
"""Download + statically validate candidate Microduck brains from Hugging Face.

Only small files are fetched (one .onnx, manifest.json, README.md, timeline.json if any) into
brains/<owner>__<name>/. Everything downloaded is treated as untrusted DATA: no code from a repo is
ever executed; the ONNX file is only opened with onnxruntime and its I/O signature checked
(obs [*,61] -> action [*,14], optional LSTM h/c state per docs/recurrent-policies.md).

    .venv/bin/python brains_fetch.py            # fetch every candidate in CANDIDATES (skips present files)
    .venv/bin/python brains_fetch.py --check    # re-validate what is on disk only
"""

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = os.path.join(HERE, "brains")
OFFICIAL = "pollen-robotics/microduck-policies"
MAX_ONNX_BYTES = 16 * 1024 * 1024  # these MLPs are ~1-3 MB; refuse anything big

# (repo, onnx file in the repo). Official set: one entry per file.
CANDIDATES = [
    (OFFICIAL, "roulade.onnx"),
    (OFFICIAL, "ball_kick_left.onnx"),
    (OFFICIAL, "ball_kick_right.onnx"),
    (OFFICIAL, "alpha_ground_pick.onnx"),
    (OFFICIAL, "roller.onnx"),
    (OFFICIAL, "roller_crouch.onnx"),
    ("pollen-robotics/microduck-sprint-2m-reference", "policy.onnx"),
    ("pollen-robotics/microduck-hill-climb-1m5-reference", "policy.onnx"),
    ("pollen-robotics/microduck-roller-sprint-2m-reference", "policy.onnx"),
    ("RemiFabre/microduck-sprint-fixed", "policy.onnx"),
    ("RemiFabre/microduck-hill-fixed", "policy.onnx"),
    ("RemiFabre/microduck-roller-fixed", "policy.onnx"),
    ("RemiFabre/microduck-flamingo-cycle", "policy.onnx"),
    ("hofftv/microduck-nugget-sprint", "policy.onnx"),
    ("hofftv/microduck-nugget-hill", "policy.onnx"),
    ("hofftv/microduck-nugget-hill-walk", "policy.onnx"),
    ("dashi/microduck-sprint-remix-r856", "policy.onnx"),
    ("jrubiosainz/microduck-game-sprinter", "policy.onnx"),
    ("jrubiosainz/microduck-game-sprinter-remix4", "policy.onnx"),
    ("jrubiosainz/microduck-game-hillclimber-remix5", "policy.onnx"),
    ("jrubiosainz/microduck-game-roller", "policy.onnx"),
    ("austindixson/microduck-roller-improver-rev4", "policy.onnx"),
    ("austindixson/microduck-sprint-improver-rev5", "policy.onnx"),
    ("RhodWeo/microduck-rhod-roller", "policy.onnx"),
    ("namenu/microduck-arena", "policy.onnx"),
    ("witcheer/microduck-roulade", "policy.onnx"),
    ("witcheer/microduck-kick-right", "policy.onnx"),
    ("guang384/microduck-side-roll-left", "policy.onnx"),
    ("joanfox/microduck-happy-hop", "policy.onnx"),
    ("chelleboyer/microduck-hop-once", "policy.onnx"),
    ("kramp/microduck-groove", "policy.onnx"),
    ("neil-jo/microduck-polite-bow", "policy.onnx"),
    ("drfonz-indigo/microduck-playdead", "policy.onnx"),
    ("microduckapp/microduck-chatterbox", "policy.onnx"),
    ("microduckapp/microduck-humming-along", "policy.onnx"),
]


def local_dir(repo, fname):
    owner, name = repo.split("/", 1)
    d = f"{owner}__{name}"
    if repo == OFFICIAL:
        d += "__" + os.path.splitext(fname)[0]
    return os.path.join(BRAIN_DIR, d)


def _get(url, limit, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "microduck-sim-brains/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = int(r.headers.get("Content-Length") or 0)
        if n and n > limit:
            raise ValueError(f"{url}: {n} bytes > limit {limit}")
        b = r.read(limit + 1)
        if len(b) > limit:
            raise ValueError(f"{url}: larger than {limit} bytes")
        return b


def hf_info(repo):
    try:
        return json.loads(_get(f"https://huggingface.co/api/models/{repo}", 4_000_000))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def onnx_signature(path):
    """Open with onnxruntime only; return a dict describing the I/O, or raise ValueError."""
    import onnxruntime as ort
    import numpy as np
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    ins = {i.name: (list(i.shape), i.type) for i in s.get_inputs()}
    outs = {o.name: (list(o.shape), o.type) for o in s.get_outputs()}
    sig = {"inputs": ins, "outputs": outs, "recurrent": False}
    names = set(ins)
    if names >= {"h_in", "c_in"} and len(ins) == 3:
        obs_name = (names - {"h_in", "c_in"}).pop()
        hs = ins["h_in"][0]
        if not (len(hs) == 3 and all(isinstance(x, int) for x in (hs[0], hs[2]))):
            raise ValueError(f"bad LSTM state shape {hs}")
        sig.update(recurrent=True, layers=hs[0], hidden=hs[2])
    elif len(ins) == 1:
        obs_name = next(iter(ins))
    else:
        raise ValueError(f"unsupported inputs {ins}")
    if ins[obs_name][0][-1] != 61 or "float" not in ins[obs_name][1]:
        raise ValueError(f"obs input {ins[obs_name]} is not [*,61] float")
    sig["obs_name"] = obs_name
    # run once on a nominal standing obs, check finite [1,14]
    obs = np.zeros((1, 61), dtype=np.float32)
    obs[0, 5] = -1.0
    feed = {obs_name: obs}
    if sig["recurrent"]:
        feed["h_in"] = np.zeros((sig["layers"], 1, sig["hidden"]), dtype=np.float32)
        feed["c_in"] = np.zeros((sig["layers"], 1, sig["hidden"]), dtype=np.float32)
    res = s.run(None, feed)
    names_out = [o.name for o in s.get_outputs()]
    act_name = "actions" if "actions" in names_out else names_out[0]
    a = res[names_out.index(act_name)]
    if a.shape[-1] != 14 or not np.all(np.isfinite(a)):
        raise ValueError(f"action output shape {a.shape} / non-finite")
    sig["action_name"] = act_name
    return sig


def fetch_one(repo, fname, force=False):
    d = local_dir(repo, fname)
    os.makedirs(d, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main/"
    onnx_path = os.path.join(d, "policy.onnx")
    if force or not os.path.exists(onnx_path):
        b = _get(base + fname, MAX_ONNX_BYTES)
        with open(onnx_path, "wb") as f:
            f.write(b)
    for extra, limit in (("manifest.json", 200_000), ("README.md", 400_000), ("timeline.json", 400_000)):
        p = os.path.join(d, extra)
        if os.path.exists(p) and not force:
            continue
        try:
            with open(p, "wb") as f:
                f.write(_get(base + extra, limit))
        except Exception:  # noqa: BLE001  (optional files)
            if os.path.exists(p) and os.path.getsize(p) == 0:
                os.remove(p)
    info_p = os.path.join(d, "hf_info.json")
    if force or not os.path.exists(info_p):
        info = hf_info(repo)
        card = info.get("cardData") or {}
        keep = {"id": info.get("id", repo), "author": info.get("author") or repo.split("/")[0],
                "license": card.get("license"), "sha": info.get("sha"),
                "lastModified": info.get("lastModified"), "downloads": info.get("downloads"),
                "likes": info.get("likes"), "file": fname}
        with open(info_p, "w") as f:
            json.dump(keep, f, indent=1)
    return d


def manifest_for(d, fname):
    """Return this policy's manifest entry (single-policy repo or one entry of a set)."""
    p = os.path.join(d, "manifest.json")
    if not os.path.exists(p):
        return {}
    try:
        m = json.load(open(p))
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(m.get("policies"), list):
        top = {k: v for k, v in m.items() if k != "policies"}
        for e in m["policies"]:
            if e.get("file") == fname:
                return {**top, **e}
        return top
    return m


def validate(d, fname):
    m = manifest_for(d, fname)
    problems = []
    if m.get("obs_len", 61) != 61:
        problems.append(f"obs_len {m.get('obs_len')}")
    if m.get("action_len", 14) != 14:
        problems.append(f"action_len {m.get('action_len')}")
    if (m.get("robot") or {}).get("model", "microduck") != "microduck":
        problems.append("robot.model")
    if m.get("model_api", 1) not in (1, 2):
        problems.append(f"model_api {m.get('model_api')}")
    try:
        sig = onnx_signature(os.path.join(d, "policy.onnx"))
    except Exception as e:  # noqa: BLE001
        sig = None
        problems.append(f"onnx: {e}")
    if sig and sig["recurrent"] and m.get("model_api", 1) != 2:
        problems.append("LSTM export without model_api 2")
    return m, sig, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(BRAIN_DIR, exist_ok=True)
    ok = 0
    for repo, fname in CANDIDATES:
        try:
            d = local_dir(repo, fname) if args.check else fetch_one(repo, fname, args.force)
            m, sig, problems = validate(d, fname)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {repo}:{fname}: {e}")
            continue
        size = os.path.getsize(os.path.join(d, "policy.onnx"))
        rec = "lstm" if sig and sig["recurrent"] else "mlp"
        status = "ok  " if not problems else "BAD "
        ok += not problems
        print(f"{status}{repo:52s} {fname:24s} {size/1e6:5.2f}MB {rec} kind={m.get('kind')} "
              f"dur={m.get('duration_s')} {'; '.join(problems)}")
    print(f"{ok}/{len(CANDIDATES)} valid")


if __name__ == "__main__":
    sys.exit(main())
