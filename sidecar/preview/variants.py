"""Try decimation variants for one part and render the face view for each (no AO, quick)."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import duck_sim as ds  # noqa: E402
import model_export as me  # noqa: E402
import render as r  # noqa: E402
import fast_simplification as fs  # noqa: E402

part = sys.argv[1]
sim = ds.DuckSim(log=lambda *a: None)
pose = json.load(open(os.path.join(HERE, "pose_stand.json")))
orig = me._decimate


def variant(cell, target):
    def dec(v, f, t):
        if t != me.PARTS[part][1]:
            return orig(v, f, t)
        v, f = me._weld(v, f)
        if cell:
            v, f = me._cluster(v, f, cell)
        for _ in range(3):
            if len(f) <= target * 1.1:
                break
            v2, f2 = fs.simplify((v * 1000).astype(np.float32), f.astype(np.int64), target_reduction=1 - target / len(f), agg=7)
            if len(f2) >= len(f) * 0.98:
                break
            v, f = v2.astype(np.float64) / 1000, f2.astype(np.int64)
        print(part, cell, len(f), "closed", me._closed(f))
        return me._compact(v, f)
    return dec


for cell in [float(x) for x in sys.argv[2].split(",")]:
    me._decimate = variant(cell, int(sys.argv[3]))
    model, _, n = me.build_model_json(sim.model, ao=False)
    P, CF, CB = r.build(model, pose, "after")
    head = r.C @ np.array(pose["bodies"]["jaw_soft"][:3])
    views = [(head + np.array([0.30, 0.0, 0.0]), head, 18), (head + np.array([0.26, 0.06, 0.15]), head + np.array([0.03, -0.012, 0]), 30)]
    imgs = [r.rasterize(P, CF, CB, e, t, fv, 500, 420, ss=2) for e, t, fv in views]
    r.write_png(os.path.join(HERE, f"var_{part}_{cell}.png"), np.concatenate(imgs, 1))
