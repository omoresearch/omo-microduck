#!/usr/bin/env python3
"""Optional stress run: random command sequence for N simulated minutes; reports falls/resets."""
import os, random, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duck_sim as ds

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
logs = []
s = ds.DuckSim(log=lambda m: logs.append(m))
choices = [("walk", {}), ("walk", {}), ("stop", {}), ("turn", {"dir": "left"}), ("turn", {"dir": "right"}),
           ("sit", {}), ("stand", {}), ("follow", {}), ("quack", {}), ("walk", {"vx": 0.1, "wz": 0.5})]
t_end = minutes * 60
cmds = 0
w = time.perf_counter()
while s.t < t_end:
    c, kw = random.choice(choices)
    if c == "follow":
        s.set_player(random.uniform(-3, 3), random.uniform(-3, 3))
    s.command(c, **kw)
    cmds += 1
    for _ in range(int(random.uniform(1.0, 8.0) / ds.CONTROL_DT)):
        s.step()
print(f"{minutes:.0f} min sim, {cmds} random cmds, falls={s.falls}, auto-resets={s.resets}, wall={time.perf_counter() - w:.0f}s")
for m in logs:
    print("  ", m)
