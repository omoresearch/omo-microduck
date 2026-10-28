"""Duck-duck contact for the multi-duck sidecar (multi.py).

Each duck keeps its own MjModel/MjData (so skates on/off, terrain rebuilds and resets stay per-duck), and every
model gets K = max_ducks - 1 "ghost" mocap bodies: a capsule the size of a standing Microduck, attached to the
trunk frame. Before each control step the fleet moves duck i's ghosts onto the other ducks' trunk poses, so the
other ducks are solid in duck i's world. Both ducks see each other, so a bump pushes both apart through real
MuJoCo contacts (the ghost is infinitely heavy for one physics step, the symmetric ghost in the other world
does the same to it).

Ghost geoms are in geom group 5: terrain rays (group 4), the floor/terrain down-ray and the renderer never see
them. Unused ghosts are parked far away above the floor.
"""

import math

import mujoco
import numpy as np

GROUP = 5
# capsule in the trunk frame: covers the standing robot's collision AABB in the trunk frame
# (x -0.055..0.098, y -0.093..0.093, z -0.107..0.161) slightly inside, rounded so pushes stay smooth
OFFSET = (0.02, 0.0, 0.025)
RADIUS = 0.065
HALF_LEN = 0.07
PARK = (500.0, 500.0, 50.0)


def apply(spec, n):
    """Add n parked mocap ghost bodies at the END of the worldbody (robot body/geom ids stay unchanged)."""
    for k in range(int(n)):
        b = spec.worldbody.add_body()
        b.name = f"ghost_{k}"
        b.mocap = True
        b.pos = [PARK[0] + 2.0 * k, PARK[1], PARK[2]]
        g = b.add_geom()
        g.name = f"ghost_{k}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [RADIUS, HALF_LEN, 0.0]
        g.pos = list(OFFSET)
        g.group = GROUP
        g.contype = 1
        g.conaffinity = 1
        g.condim = 3
        g.friction = [0.4, 0.005, 0.0001]   # ducks slide off each other rather than grip
        g.solref = [0.02, 1.0]
        g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        g.rgba = [0.2, 0.6, 1.0, 0.3]


def mocap_ids(sim):
    """Mocap indices of the ghost bodies in the sim's current model (cached per model object)."""
    m = sim.model
    c = getattr(sim, "_ghost_cache", None)
    if c is not None and c[0] is m:
        return c[1]
    ids = []
    k = 0
    while True:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ghost_{k}")
        if b < 0:
            break
        ids.append(int(m.body_mocapid[b]))
        k += 1
    sim._ghost_cache = (m, ids)
    return ids


def update(sim, others):
    """Place sim's ghosts on `others` = [(pos3, quat4), ...] (trunk frames of the other ducks); park the rest."""
    ids = mocap_ids(sim)
    d = sim.data
    for k, mid in enumerate(ids):
        if mid < 0:
            continue
        if k < len(others):
            p, q = others[k]
            d.mocap_pos[mid] = p
            d.mocap_quat[mid] = q
        else:
            d.mocap_pos[mid] = (PARK[0] + 2.0 * k, PARK[1], PARK[2])
            d.mocap_quat[mid] = (1.0, 0.0, 0.0, 0.0)


def contacts(sim):
    """(count, deepest penetration m) of robot-vs-ghost contacts in the sim's current state."""
    m, d = sim.model, sim.data
    gg = getattr(sim, "_ghost_geoms", None)
    if gg is None or gg[0] is not m:
        gg = (m, set(np.nonzero(m.geom_group == GROUP)[0].tolist()))
        sim._ghost_geoms = gg
    n, worst = 0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in gg[1]) != (c.geom2 in gg[1]):
            n += 1
            worst = min(worst, float(c.dist))
    return n, worst


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
