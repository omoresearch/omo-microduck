# Sidecar — MuJoCo Microduck for Minecraft

Runs the open-source Pollen Robotics **Microduck** (MJCF + meshes from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl), Apache-2.0) with the
pretrained ONNX policies from [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
on the CPU, in real time at 50 Hz, and serves its pose over HTTP on `127.0.0.1:8796` for the Fabric mod.
API contract: [`CONTRACT.md`](CONTRACT.md).

## Setup

```bash
./setup.sh              # clone the Pollen robot (pinned), venv + deps, community brains (~28 MB), smoke test
./setup.sh --no-brains   # same, without the Hugging Face brains (no tricks/sprint/skates)
```

## Run

```bash
systemctl --user status microduck-sim        # installed + enabled user service
systemctl --user restart microduck-sim
journalctl --user -u microduck-sim -f        # commands, falls, auto-resets are logged

# manual (stop the service first; port 8796 only):
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python server.py

# tests
.venv/bin/python test_headless.py            # no HTTP, ~11x real time, PASS/FAIL summary
.venv/bin/python test_http.py                # against the running service
.venv/bin/python stress.py 10 1              # optional: 10 sim-minutes of random commands
```

Service unit: `contrib/microduck-sim.service` (copy to `~/.config/systemd/user/`) (`Nice=10`, `OMP_NUM_THREADS=1`,
`Restart=on-failure`). Steady-state CPU ≈ 7 % of one core, single sim thread, RSS ≈ 470 MB.

## Endpoints (summary — see CONTRACT.md)

| | |
|---|---|
| `GET /health` | `{ok, mode, fallen, t, policy, hz}` + extras: `leash, quack, resets, falls, seq, overruns, player, pos` |
| `GET /model.json` | 15 bodies, 25 536 triangles (~2.1 MB), body-local verts (geom pos/quat baked), real-robot rgba; optional v1-compatible per-mesh extras `name`, `normals` (crease-split smooth), `colors` (sRGB bytes, baked AO), `cull` (closed mesh). Previews: `preview/` |
| `GET /pose` | `{seq, t, mode, fallen, quack, bodies:{name:[px,py,pz,qw,qx,qy,qz]}}`, world frame, every model.json body |
| `POST /cmd` | `walk` (opt. `vx,vy,wz`), `stop`, `sit`, `stand`, `follow`, `quack`, `turn` + `dir`, `reset`. `left`/`right` accepted as aliases for turn. `from:"twitch:*"` may not `reset` (403). |
| `POST /player` | `{x,y}` sim-frame metres, used by `follow` |
| `POST /config` | `{leash: r}` (0.3–100 m, default 2.5) |
| `GET /ducks` / `POST /ducks` / `POST /ducks/remove` / `DELETE /ducks/<sel>` | multiple ducks: list, spawn (`persona`: id, name, none = next unused, `all`), remove (`duck`: sel or `all`) |
| `GET /poses` | every duck's `/pose` (+ name, color, quack_pitch, talking) keyed by id, plus `order`, `default` |
| `POST /cmd {"duck": sel}` | optional target: id, name (case-insensitive, prefix ok) or `all`; omitted = default (first) duck |

Modes reported: `stand`, `walk`, `turn`, `follow`, `sit` (sitting down / seated). While rising from a sit the
mode reports the queued target (`stand` or e.g. `walk`).

## Design

- `duck_sim.py` — sim core (no HTTP). Replicates `scripts/infer_policy.py` defaults:
  - 61-D obs = imu gyro (3) + projected gravity from trunk quat (3) + joint pos − default pose (14) + joint vel (14)
    + last action (14) + 13-D command `[vx,vy,wz, head(4), body(6)]` (`--new-cmd-obs` layout; matches the ONNX
    metadata `observation_names`).
  - action → target = `DEFAULT_POSE + action × 1.0`; physics `dt = 0.005`, decimation 4 → 50 Hz.
  - **BAM M6 XL330 actuator model** (the default in infer_policy, the actuator the policies were trained with):
    `better-actuator-models` from the `mjlab_frictionloss` branch, installed with `--ignore-requires-python`
    (it pins Python < 3.13; it's pure numpy and works on 3.14). kp_fw 200, vin 7.4 V, load sag 0.1 V/Nm
    (passed as `vin_drop_resistance = 0.1·kt` — this bam version has no `vin_drop_gain` kwarg), vin_min 6 V,
    stiff friction constraint. `DuckSim(use_bam=False)` gives the `--no-bam` XML PD path.
  - onnxruntime sessions with intra/inter-op threads = 1.
- Policies: `velstand.onnx` for stand/walk/turn/follow and fall recovery; `alpha_sitstand.onnx` for sit/stand
  using the manifest's `posture_flag` (twist.vx = 1 sit, 0 stand). Sit is considered settled after `ramp_s` = 2 s;
  standing up keeps sitstand at flag 0 for 2.5 s (≥ `unwind_s`) until upright, then hands back to velstand.
  A stand request during the 2 s descent is deferred until the seat settles (flipping mid-descent can tip it).
  A walk/turn/follow command while seated stands up first, then runs the queued mode.
- Backlash: velstand was trained on the backlash robot, alpha_sitstand on the no-backlash standup robot, so the
  14 backlash joints are clamped to ±0.02° while sitstand is in control (runtime `jnt_range` change). Without this
  the rise tipped over in ~5–12 % of random trials; with it 0/130.
- High-level control (50 Hz, on top of the policy command):
  - `walk`: heading hold (velstand drifts in yaw), a 0.8 s start kick at 0.3 (velstand has a standing deadband
    below ~0.2 from rest), and a speed gain (see limitations).
  - `turn`: wz up to 1.0 rad/s until yaw is within 0.08 rad of ±90°, then `stand` (8 s timeout).
  - `follow`: steer toward the player's last `/player` position, stop within 0.4 m.
  - leash: walking outward past `leash − min(0.45, 0.3·leash)` switches to “return to origin” until within
    0.35·leash, then continues in the new heading; follow targets are clamped inside the leash.
  - `quack`: increments the counter and nods the head (velstand head-pitch command) for 0.35 s.
- Falls: tilt > 60° or trunk < 6 cm → `fallen`. velstand gets up on its own (≈0.3–1.5 s in tests); if still down
  after 3 s → auto-`reset` (respawn upright at origin). A fall during sitstand hands control to velstand.
- `server.py` — stdlib `ThreadingHTTPServer` on 127.0.0.1; one sim thread steps under a lock with an absolute
  50 Hz schedule (resyncs instead of spiralling if > 250 ms behind, counted in `overruns`) and publishes a
  pre-serialised `/pose` snapshot; HTTP threads only read snapshots or apply commands under the lock.
- `model_export.py` — visual geoms = mesh geoms in group 2 with contype/conaffinity 0. Per-part palette and
  triangle targets in `PARTS` (white shells, grey face plate, amber eye ring + black lens, orange beak/feet, amber
  soles, black servos; enclosed PCBs/speaker/inner bearings dropped; ~25.5k tris). Decimation: weld -> 0.2 mm
  vertex clustering (regularises CAD sliver fans; quadric simplification alone stalls at ~4k tris and leaves
  fold-over gashes on the shells) -> fast-simplification in mm. Then 35° crease-split smooth normals, per-vertex AO
  (24 `mj_multiRay` rays, same-body occluders only so it stays valid in any pose), plus two face decals (black
  depth-sensor window over the face-plate cut-out, lens catch-light). ~20 s; cached in `model_cache.json`
  (signature includes palette/targets, so edits rebuild it). Offline previews (same shading model as the mod):
  `preview/render.py`, `preview/compose.py` -> `preview/compare.png`.

## Multiple ducks

Up to 4 ducks (`MICRODUCK_MAX_DUCKS`, max 8), each with a persona from `personas.json` (name, GPT-Live voice,
personality prompt, favourite brain/trick, quack pitch, accent colour). Full API in CONTRACT.md "Multiple ducks".

| persona | voice (median F0) | personality | brain / trick | colour |
|---|---|---|---|---|
| Quackers | juniper (~230 Hz) | bubbly original | default / groove | #F2C230 yellow |
| Rusty | arbor (97 Hz) | grumpy sarcastic old engineer | default / kick | #C0392B red |
| Zippy | maple (216 Hz) | hyper speedrunner | sprint / dash | #19C3E6 cyan |
| Plato | cove (122 Hz) | calm dreamy philosopher | default / flamingo | #9B6BD6 violet |
| Newton | vale (166 Hz) | nerdy scientist (explains RL/MuJoCo) | default / roulade | #3DBE5A green |

- MC chat: `!duck spawn [persona]`, `!duck spawn all` (every persona not yet spawned, up to the max, each at its own
  free spot), `!duck remove <name>`, `!duck remove all` (keeps nothing; respawn with `!duck spawn ...`), `!duck list`,
  `!duck <name> <cmd>`, `!ducks <cmd>`; plain `!duck <cmd>` = nearest duck. Twitch: `!duck [<name>] <cmd>`, `!ducks <cmd>`.
- Design: one MjModel/MjData per duck (robot variants, terrain rebuilds and resets stay per duck) plus mocap ghost
  capsules (geom group 5) mirroring the other ducks' trunks, so they physically bump; one 50 Hz thread steps all;
  ONNX sessions shared per policy file. Not one merged MJCF: a merged world would have to be recompiled (~0.35 s per
  robot) on every spawn/remove/roller-skate swap/terrain rebuild and would couple resets. Limitation: a duck feels the
  others only as a trunk-sized capsule (legs/heads pass through each other, rarely visible at the duck's scale).
- Roster persists in `ducks.json` (`MICRODUCK_STATE`); SIGTERM saves it. First run: duck 1 = Quackers at the anchor.
- Timing (test_multi.py, test server at nice 10 like the service, busy streaming box): 1 duck 1.7 ms/step 7.7 % of a
  core, 2 ducks 2.9 ms 13.7 %, 3 ducks 4.6 ms 22 %, 4 ducks 4.0 ms 22 %; 50.0 Hz, 0 overruns at every count.
  Headless step cost mean 0.64 / 1.22 / 1.85 / 2.46 ms for 1-4 ducks.
- Tests: `nice -n 10 .venv/bin/python test_multi.py` (headless + own HTTP server on :8806 with a temp state file),
  `test_http.py [--multi]` (multi checks; spawn/remove only against a `MICRODUCK_TEST=1` sidecar).

## Known limitations / deviations

- **Scene**: uses `scene_sim_allcollisions_backlash.xml` (generated copy of `scene.xml` including
  `robot_allcollisions_backlash.xml`, the robot velstand was trained on: full collisions + 2° backlash) instead of
  infer_policy's default `scene.xml` (ground-contact-only, no backlash), where walking at low commands is noticeably
  worse. Override with `MICRODUCK_SCENE=scene.xml`.
- **Speed**: in this CPU sim velstand achieves only ~⅓ of the commanded forward speed (0.15 cmd → ~0.05 m/s,
  0.4 → ~0.15 m/s; alpha_walking behaves the same, so it's a sim/actuator-fidelity gap, not the obs). To make
  `walk` move at the contract's ~0.15 m/s, API velocities are multiplied by `MICRODUCK_SPEED_GAIN` (default 2.667,
  clipped to the trained ±0.4 / ±0.3), so the default walk feeds the policy 0.4. Follow / leash-return use 0.3.
- Velocity overrides `vx,vy,wz` are therefore in “nominal” m/s, not exact.
- No joint-velocity observation lag (training uses 1 step; infer_policy doesn't either). `MICRODUCK_JV_LAG=1` enables it.
- Terrain (`terrain.py`, CONTRACT.md "Terrain"): the mod posts the real blocks around the anchor as boxes; walls
  block it (mj_ray wall-avoid steering turns it away), it falls off ledges, carpets are walkable (ramped edges).
  Physically honest step height: at scale 4 a slab is 12.5 cm and a block 25 cm (its own height) — it cannot climb
  either. Tests: `test_terrain.py` (headless room/wall/ledge/swap), `test_terrain_http.py [url]` (POST/GET + timing).
  Without a terrain POST it's the old flat floor at z=0.
- `/player` is not timed out: follow uses the last position received.
- Stress (`stress.py 10 2`, `stress.py 10 3`: 2 × 10 sim-minutes of random commands incl. sit/stand/follow): 0 falls, 0 auto-resets.
- Rising from a sit with velstand alone (`MICRODUCK_RISE_WITH=velstand`) works but goes through a protective
  fall (~100° tilt) first, so sitstand does the rise by default.

## Brains (community duck brains)

`brains_fetch.py` downloads candidate policies (one `.onnx` + `manifest.json` + `README.md` + HF card info, ~0.8 MB
each, 28 MB total) into `brains/<owner>__<name>/` and validates them: manifest `obs_len 61 / action_len 14 /
robot.model / model_api`, ONNX opened with onnxruntime only, input `[*,61]` -> output `[*,14]`, LSTM `h_in/c_in`
(model_api 2) supported, one finite warm-up inference. Nothing from a repo is executed. `brains.py eval` (run under
`nice -n 19`, ~4 min for 36 candidates, ~30x real time) runs each candidate through the runtime code path
(`BrainSim`) on 3 perturbed seeds and a small config grid (joint-velocity obs lag 0/1 x one-step action delay 0/1,
and command speed for the Arena brains) and writes `brains/catalogue.json` (status `pass` / `wobbly` / `fail`, chosen
config, per-trial metrics). The 12 `featured` ids that pass/wobble are served (`MICRODUCK_BRAINS=all` serves every
passing one).

```bash
.venv/bin/python brains_fetch.py                 # download + validate candidates (skips present files)
nice -n 19 .venv/bin/python brains.py eval       # (re)evaluate all -> brains/catalogue.json  (or: eval <id> ...)
.venv/bin/python brains.py list
nice -n 19 .venv/bin/python test_brains.py       # one check per catalogued brain + hot-swap invariants (~7 s)
```

Eval checks: perpetual = 6 s unconstrained run (distance/speed/fell) + `stop` hand-back to velstand (upright) +
20 s with leash 2.5 (no falls, stays inside); rollers also a steering probe (heading-error slot). Tricks = 1 s
velstand settle, the trick, 5 s velstand: did it do the thing (roll: inverted; kick: swing-foot speed > 0.6 m/s and
> 1.3x the other foot; hop: airborne >= 40 ms; flamingo: one foot up > 1.5 s; playdead: ends > 75° tilted; dance:
joint motion without tilting), upright afterwards, no auto-reset. `wobbly` = works but doesn't land on its own feet
(velstand catches it).

| id | source repo | kind | eval (3 seeds) |
|---|---|---|---|
| `sprint` | pollen-robotics/microduck-sprint-2m-reference | brain | pass: 0.51 m/s at vx 0.55, 0 falls, leash 20 s ok |
| `hill` | pollen-robotics/microduck-hill-climb-1m5-reference | brain | pass: 0.57 m/s, 0 falls, leash ok |
| `nugget` | hofftv/microduck-nugget-sprint | brain | pass: 0.34 m/s, 0 falls, leash ok |
| `roller` | pollen-robotics/microduck-policies roller.onnx | brain (skates) | pass: 0.43 m/s, steers 90°, 0 falls, leash ok |
| `roulade` | pollen-robotics/microduck-policies roulade.onnx | trick 1.8 s | pass: full forward roll (178°), lands upright |
| `kick` | pollen-robotics/microduck-policies ball_kick_right.onnx | trick 0.5 s | pass: right foot 0.82 m/s (left 0.39) |
| `cartwheel` | guang384/microduck-side-roll-left | trick 3.5 s | wobbly: side roll (168°), 1/3 needs velstand to finish |
| `hop` | joanfox/microduck-happy-hop | trick 3 s | pass: 107 ms airborne (needs the 1-step action delay) |
| `groove` | kramp/microduck-groove | trick 10 s | pass: knee-bounce dance, tilt < 6° |
| `flamingo` | RemiFabre/microduck-flamingo-cycle | trick 7 s | pass: one foot up 3.7 s (flag 1 for 5 s, then 0) |
| `playdead` | drfonz-indigo/microduck-playdead | trick 5 s | pass: flops over (90°), velstand gets up |
| `dash` | RemiFabre/microduck-sprint-fixed | trick 1.5 s | wobbly: 1.9 m in 1.5 s (1.7 m/s), ends in a tumble |

Also passing but not featured: `bow`, `chatter`/`hum` (LSTM), `flip` (witcheer roulade), `kick-left`, `leap`,
`pick`, `crouch` (roller trick), `nugget-hill`, `nugget-trot`. Rejected (see catalogue.json): Arena "remix"
graphs with per-seed branch tables (`zoom-remix`, `dashi-remix`, `sprint-rev5`, `climber`, `skate-rev4`) and the
command-blind PPO/ES roller racers (`skate`, `arena-skate`) fall within ~1 s here; `rhod` crawls (0.12 m/s);
`roller-sprint`/`roller-fixed` skate 0.6 m/s but drift and fall within 6 s and aren't trained to steer;
`sprint-fixed`/`hill-fixed` run 1.7/1.4 m/s but ignore the speed command, so nothing can stop them at the leash
(velstand can't catch a 1.7 m/s duck: 6-9 falls per 60 s) -> `sprint-fixed` is offered only as the `dash` trick;
`zoom` (jrubiosainz game-sprinter) falls in all seeds.

Runtime (`BrainSim(DuckSim)` in brains.py; server.py only swaps the class, adds `/brains`, `name`, health fields):
- Perpetual brains drive `walk` mode with command `[vx,0,0]` (manifest idle elsewhere); everything else (stand,
  turn, follow, sit, get-up) stays velstand/sitstand. Leash: when the trunk velocity points outward within
  `margin + 0.45·speed` of the leash, velstand takes over and walks back (existing "returning" logic), then the brain
  resumes. A fall while a brain drives -> default brain.
- Roller brains need `robot_groundcontact_rollers.xml`: `_switch_variant` loads it through `ds.load_model` (so the
  terrain boxes are included and `terrain.build` follows the current robot via `ds.SCENE_XML`), caches it, rebinds all
  joint/sensor indices and respawns standing at the same x/y/yaw (spawn height found by a contact probe). Swap ~0.35 s
  the first time (compile + ONNX load, under the sim lock -> one resync), ~0 ms after (cached per terrain). Leash: heading-error steering back to the origin past half the leash; if it still
  reaches the leash, skates come off and velstand walks back. Fallen on skates -> hold pose -> auto-reset (3 s) ->
  default robot + default brain. `/pose` renders feet only (wheels/blades not in model.json).
- Tricks: `trick_command` = constant (manifest `command.idle`), `phase` encoding (`[cos 2πφ, sin 2πφ, 0]`) or a
  timed sequence (flamingo); fall detection is paused during the trick and resumes at hand-back.
- Per-brain obs/actuation quirks found by the eval: Arena / challenge brains want the 1-step action delay (their
  training servo-command delay) — without it `sprint-fixed` falls in 2 s, with it it runs 10 m in 6 s; robot model
  (allcollisions+backlash vs robot_walk) and integrator (Euler vs implicitfast) made no difference, so walking brains
  run on velstand's robot (no model swap, no pop).
