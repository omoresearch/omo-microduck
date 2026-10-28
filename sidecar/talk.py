"""Speaking animation for the Microduck: nod the head / "open the beak" while voice audio plays.

POST /cmd {"cmd":"talk","on":true|false}            start/stop (on:true auto-expires after MAX_S)
POST /cmd {"cmd":"talk","duration":2.5}             talk for N seconds (0.1..MAX_S)

It only adds to the policy's head command channels (13-D command: [3]=neck_pitch, [4]=head_pitch,
[5]=head_yaw, [6]=head_roll), the same channels quack's nod uses, with small amplitudes so the
velstand/sitstand policies keep their balance. Walking, follow, sit etc. keep working while talking.
"""

import math

MAX_S = 30.0          # safety: a lost "talk off" never leaves it nodding forever
PITCH_AMP = 0.35      # rad, head_pitch syllable nod (same size as quack's -0.35 step)
NECK_AMP = 0.10       # rad, neck_pitch follows a little so the whole head bobs
PITCH_BIAS = -0.06    # slight downward tilt while speaking
YAW_AMP = 0.08        # rad, lazy side-to-side
NOD_HZ = 3.2          # ~syllable rate
YAW_HZ = 0.7
RAMP_S = 0.15         # fade in/out so the head never gets a step input


def _state(sim):
    st = getattr(sim, "_talk", None)
    if st is None:
        st = sim._talk = {"until": 0.0, "t0": 0.0, "gain": 0.0, "count": 0}
    return st


def command(sim, on=None, duration=None, **_):
    """Returns (ok, error)."""
    st = _state(sim)
    if duration is not None:
        try:
            d = float(duration)
        except (TypeError, ValueError):
            return False, "duration must be a number"
        if not math.isfinite(d) or d <= 0:
            return False, "duration must be > 0"
        d = min(d, MAX_S)
        if st["until"] <= sim.t:
            st["t0"] = sim.t
            st["count"] += 1
        st["until"] = sim.t + d
        return True, None
    if on is None:
        on = True
    if isinstance(on, str):
        on = on.strip().lower() not in ("0", "false", "off", "no", "")
    if on:
        if st["until"] <= sim.t:
            st["t0"] = sim.t
            st["count"] += 1
        st["until"] = sim.t + MAX_S
    else:
        st["until"] = min(st["until"], sim.t)
    return True, None


def talking(sim):
    return _state(sim)["until"] > sim.t


def count(sim):
    return _state(sim)["count"]


def apply(sim, command_vec, dt):
    """Add the talking head motion to the 13-D policy command (in place)."""
    st = _state(sim)
    target = 1.0 if (st["until"] > sim.t and not sim.fallen) else 0.0
    step = dt / RAMP_S
    g = st["gain"]
    g = min(target, g + step) if target > g else max(target, g - step)
    st["gain"] = g
    if g <= 0.0:
        return
    tt = sim.t - st["t0"]
    # 0..1 "mouth open" envelope with a little irregularity so it reads as speech
    nod = 0.5 - 0.5 * math.cos(2 * math.pi * NOD_HZ * tt + 0.6 * math.sin(2 * math.pi * 0.9 * tt))
    command_vec[3] += g * (-NECK_AMP * nod)
    command_vec[4] += g * (PITCH_BIAS - PITCH_AMP * nod)
    command_vec[5] += g * YAW_AMP * math.sin(2 * math.pi * YAW_HZ * tt)
