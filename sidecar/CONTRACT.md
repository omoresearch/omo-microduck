# Microduck-in-Minecraft: sidecar <-> mod contract

Goal: the real open-source Pollen Robotics Microduck (github.com/pollen-robotics/microduck_rl MJCF + meshes,
pretrained ONNX policies from huggingface.co/pollen-robotics/microduck-policies) runs in a local CPU MuJoCo
sidecar, and a Fabric client mod renders it, articulated, in your Minecraft world (MC 26.2).

## Pieces
| piece | path |
|---|---|
| sim sidecar (Python, mujoco + onnxruntime) | `sidecar/` |
| user service | `sidecar/contrib/microduck-sim.service` |
| Fabric client mod `microduck` | `mod/` -> `mods/microduck-1.0.0+mc26.2.jar` |
| Twitch hook (optional) | any chat bot that POSTs `/cmd` with `"from":"twitch:<user>"` |

## Sidecar HTTP API — 127.0.0.1:8796 only (never 0.0.0.0), JSON, no auth (localhost only)
Sim frame: MuJoCo world, z-up, metres, floor plane at z=0. Quaternions are [w,x,y,z].

- `GET /health` -> `{"ok":true,"mode":"stand","fallen":false,"t":12.3,"policy":"velstand.onnx","hz":50}`
- `GET /model.json` -> geometry for rendering, fetched once by the mod:
  `{"version":1,"bodies":[{"name":"trunk","meshes":[{"rgba":[r,g,b,a],"verts":[x,y,z,...],"tris":[i,j,k,...]}]}]}`
  verts are in the BODY-LOCAL frame (geom pos/quat already baked in), metres. Only visual geoms (skip
  collision-only/floor). Decimate so the total is <= ~15k triangles. Colours from the MJCF materials/rgba.
- `GET /pose` -> `{"seq":123,"t":12.34,"mode":"walk","fallen":false,"quack":3,"bodies":{"trunk":[px,py,pz,qw,qx,qy,qz],...}}`
  world-frame transform of every body listed in /model.json. `quack` is a monotonically increasing counter;
  the mod plays one quack sound each time it increases.
- `POST /cmd` body `{"cmd":..., "from":"mc"|"twitch:<user>"}` -> `{"ok":true,"mode":...}` or `{"ok":false,"error":...}`
  cmds: `walk` (forward ~0.15 m/s; optional `vx`,`vy`,`wz` override), `stop` (stand still), `sit`, `stand`,
  `follow`, `quack`, `turn` with `"dir":"left"|"right"` (turn in place ~90 deg then stop), `reset` (respawn upright at origin).
- `POST /player` body `{"x":..,"y":..}` player position in SIM frame metres (mod sends ~5 Hz). Used by `follow`:
  walk toward the player, stop within ~0.4 m.
- Leash: the duck must never go further than `leash` metres (default 2.5) from the sim origin; when it would,
  it steers back. `POST /config {"leash":r}` sets it.
- The sim runs in real time at the policy rate (50 Hz) with physics substeps per the MJCF timestep, single-threaded
  (OMP_NUM_THREADS=1, onnxruntime intra/inter op threads = 1), so it never steals CPU from the game/OBS.
- Falls: if it falls, recover (velstand policy gets up) or auto-`reset` after ~3 s on the ground.

## Minecraft mapping (done in the mod)
Anchor A = (ax, ay, az) in MC block coords (the lab floor spot), scale s (default 4.0, so the 25 cm duck is ~1 block tall).
MC position of sim point p=(x,y,z): (ax + s*x, ay + s*z, az - s*y). Same axis change C: (x,y,z)->(x,z,-y) applies to
rotations: R_mc = C * R_sim * C^T. Player (MC px,pz) -> sim x=(px-ax)/s, y=-(pz-az)/s.

## Chat commands
- MC chat (typed by the player, intercepted client-side, never sent to the server): `!duck walk|stop|sit|stand|follow|quack|left|right|reset`,
  plus MC-only `!duck here` (set anchor to the player's feet, persist) and `!duck scale <n>`.
- Twitch chat (via a chat bot): `!duck walk|stop|sit|stand|follow|quack|left|right` only (no here/reset/scale),
  global cooldown 2 s, POSTed to /cmd with `"from":"twitch:<user>"`. Sidecar down -> silently ignore.

## Terrain (Minecraft blocks in the sim) — sidecar `terrain.py`, mod `DuckTerrain` / `TerrainGrid` / `TerrainScan`
- `POST /terrain` body (<= 4 MB, <= 4000 boxes), all in SIM metres (same mapping as above, so MC block Y=ay is sim z=0):
  `{"id":"<mod hash>","dim":"minecraft:overworld","scale":4,"anchor":[ax,ay,az],"region":[x0,y0,z0,x1,y1,z1],
    "boxes":[[x0,y0,z0,x1,y1,z1],...]}` (min/max corners; a flat list of 6*N numbers is accepted too).
  MC box (X0,Y0,Z0,X1,Y1,Z1) -> sim [(X0-ax)/s, -(Z1-az)/s, (Y0-ay)/s, (X1-ax)/s, -(Z0-az)/s, (Y1-ay)/s].
  `region` = scanned volume; the infinite flat floor moves to its bottom z (fallback). `"boxes":[]` = back to flat.
  -> `{"ok":true,"hash":..,"boxes":N,"changed":bool}`; identical geometry is a no-op. The model is recompiled off the
  sim lock (~70-140 ms, debounced 0.2 s, latest wins) and swapped in keeping qpos/qvel/controller state; if the duck
  would be inside a block or the ground under it moved, it is respawned standing on the nearest safe spot.
- `GET /terrain` -> `{"ok":true,"boxes":N,"hash":..,"id":..,"floor_z":..,"bounds":[..],"region":[..],"pending":bool,
  "builds":n,"last_build_ms":..,"last_respawn":reason|null}`. `/health` adds `"terrain":{boxes,hash,pending,builds}`,
  `"late"` (control steps past their 20 ms deadline) and `"step_ms"` (EMA of one control step).
- Sim: boxes are static geoms in group 4 (contype/conaffinity 1, floor friction/solref). Boxes <= 3.5 cm tall
  (carpets, snow layers) get 1:10 ramps on their sides. Walk/follow steer away from walls via mj_ray (group 4 only),
  ledges are not avoided (it falls). Fall detection uses trunk height above the ground below it; reset spawns on the
  nearest safe spot to the origin (flat footprint, 28 cm headroom).
- Mod: scans the blocks' collision shapes (VoxelShape AABBs) in leash*s+3 blocks around the anchor, 4 below to
  ~ceil(0.35 s)+2 above, <= 8192 cells / 0.7 ms per client tick, every 2 s; posts only when the content hash changes
  (merge + HTTP on a daemon thread, capped at 2000 boxes nearest the anchor); re-checks GET /terrain every 10 s and
  re-posts if the sidecar lost it (restart).

## Brains (community policies; brains.py)
Hot-swappable Microduck ONNX policies from Hugging Face (official `pollen-robotics/microduck-policies` + community
`<user>/microduck-*` repos), validated (obs 61 -> actions 14, LSTM per docs/recurrent-policies.md) and evaluated
headless; only the featured ones that passed (status `pass`, or `wobbly` = works but imperfect) are offered.
- `GET /brains` -> `{"brain":"default"|id,"trick":id|null,"robot":"default"|"rollers","brains":[{"id","kind":
  "perpetual"|"episodic","category","variant":"default"|"rollers","duration_s","description","repo","author",
  "license","status","eval":{...}}]}`. `/health` adds `"brain"`, `"trick"`, `"robot"`; mode `"trick"` while one plays.
- `POST /cmd {"cmd":"brain","name":id}`: perpetual brain replaces the walk slot and starts walking with it
  (`"default"` = back to velstand). `stop/turn/follow/sit` still use velstand; `walk` resumes the brain. Rollers brains
  swap the MjModel to the roller-skate robot (respawn standing at the same x/y/yaw); any non-walk command, a non-roller
  trick or `brain default` takes the skates off the same way. `/pose` keeps the same body names (rollers'
  `ankle_l_v1/ankle_r_v1` are reported as `ankle_left/ankle_right`; the wheels are not in /model.json).
- `POST /cmd {"cmd":"trick","name":id}`: episodic brain plays once (duration_s), then velstand stands. Refused while
  fallen/sitting, if a roller trick is asked without skates, or if a travelling trick would cross the leash. Any motion
  command cancels a trick.
- `name` must match `[a-z0-9-]{1,24}` (else 400); unknown ids -> 400. `brain` and `trick` are allowed from Twitch.
- Safety: leash still holds (walking brains hand over to velstand which walks back inside, then the brain resumes;
  roller brains steer back via the heading-error command slot); a fall while a brain drives -> default brain (velstand
  gets up); auto-reset -> default brain and default robot. A brain whose ONNX errors at runtime is dropped.
- Chat: MC `!duck brains` (list), `!duck brain <id>|default`, `!duck trick <id>`; Twitch `!duck brain <id>`,
  `!duck trick <id>` (same 2 s global cooldown; the sidecar validates ids).

## Talk animation + voice
- `POST /cmd {"cmd":"talk","on":true|false}` or `{"cmd":"talk","duration":s}`: while on, the duck nods its head
  ("beak") at ~3 Hz by adding small offsets to the policy's head command channels (neck_pitch/head_pitch/head_yaw,
  same channels quack's nod uses; logic in talk.py). Works while standing, walking or sitting; `on:true` auto-expires
  after 30 s. Not allowed from Twitch. `/health` adds `"talking": bool, "talks": n`.
- Voice backend: a GPT-Live voice backend (not part of this repo yet; 127.0.0.1:8798) runs the
  GPT-Live duck persona, sends `walk/stop/sit/stand/follow/turn/quack/reset/trick/brain` with `"from":"voice"` and
  toggles `talk` from the reply audio level.
- Mod <-> voice backend: `ws://127.0.0.1:8798/svc` (loopback). Binary frames = PCM16 LE mono 48 kHz (mic from the
  mod's Simple Voice Chat plugin; the duck's reply back), text = JSON control (`hello`, `start`, `ptt`, `hangup` from
  the mod; `status`, `transcript`, `clear` to it). The mod plays the reply on an SVC LocationalAudioChannel at the duck.


## Multiple ducks (multi.py, personas.json)

- **Roster**: up to `MICRODUCK_MAX_DUCKS` (default 4, max 8) ducks, persisted in `ducks.json` (`MICRODUCK_STATE`;
  id, persona, last position/yaw, brain) on every spawn/remove and on SIGTERM, reloaded at start. First start with no
  state file = one duck `d1` = Quackers at the anchor, so the old single-duck setup comes back as duck 1. An empty
  roster (after `remove all`) stays empty across restarts.
- **Personas**: `personas.json` (user-editable, shared with voice-live): `id`, `name` (unique, not a command word),
  `voice` (arbor|breeze|cove|ember|juniper|maple|sol|spruce|vale), `personality` (spoken prompt), `brain`, `trick`,
  `quack.pitch` (0.5-2), `quack.style`, `color` (#RRGGBB accent: eye ring + name tag), `aliases` (speech-to-text
  spellings), `tagline`. Restart microduck-sim and omo-duck-voice after editing.
- **Physics**: each duck has its own MjModel/MjData (own robot variant, terrain rebuild, BAM state, reset) stepped by
  one 50 Hz thread; ducks feel each other through K = max-1 mocap "ghost" capsules (geom group 5, trunk-sized,
  never collide with terrain) that mirror the other ducks' trunks each step, so they really bump and push.
  ONNX sessions are shared per policy file. Spawn picks a free standable spot (terrain + >= 0.4 m from other ducks,
  inside the leash).
- **Selectors** (`"duck"`): id (`d3`), name or persona id (case-insensitive, unambiguous prefix ok, aliases ok), or
  `all` / `*`. Omitted = the default duck = the first in the roster.
- **Endpoints** (all old endpoints unchanged and act on the default duck; with no ducks `/pose` has empty `bodies`,
  `/health` mode `none`, `/cmd` -> 409):
  - `GET /ducks` -> `{ok, default, max, ducks:[{id,name,persona,voice,color,brain,trick,mode,pos,yaw,fallen,robot,
    quack,talking,quack_pitch,falls,resets,follow,home}], player, leash, scale, personas:[{id,name,voice,color,brain,
    trick,tagline,used}]}`
  - `POST /ducks {"persona": id|name|null|"all", "from"}` -> spawn that persona / the next unused one / every unused
    persona up to the max (`{ok, spawned:[duck...], ducks:n, errors?}`; 409 when full or unknown). Not from twitch (403).
  - `POST /ducks/remove {"duck": sel|"all"}` and `DELETE /ducks/<sel>` -> `{ok, removed:[...], ducks:n}`. Not from twitch.
  - `POST /cmd {..., "duck": sel}` -> one duck (`{ok, mode, duck, name}`) or `all` (`{ok, mode, duck:"all",
    name:"all", results:[{id,name,ok,mode,error?}]}`). `follow` takes optional `"target": <other duck>`.
  - `GET /brains?duck=sel` (one duck; 404 "pick one duck" if ambiguous).
  - `GET /poses` -> `{seq, t, default, order:[ids], ducks:{<id>: {name, color, quack_pitch, seq, t, mode, fallen,
    quack, talking, bodies:{name:[x,y,z,qw,qx,qy,qz]}}}}` (each duck's entry has the `/pose` body format).
- **Mod**: polls `/poses` (404 -> old sidecar: `/pose`, one duck, no name tag), renders each duck with its accent
  eye ring and a name tag (16 blocks), per-duck quacks. Chat: `!duck <cmd>` (nearest duck), `!duck <name> <cmd>`,
  `!ducks <cmd>`, `!duck spawn [persona|all]`, `!duck remove <name|all>`, `!duck list` (spawn/remove MC only).
- **Twitch**: `!duck [<name>] <cmd>` and `!ducks <cmd>`, same allowed cmds and global cooldown, no spawn/remove.
- **Voice (proto 2)**: the mod's `hello` has `"proto":2`; `ptt on` carries `look` (duck id under the crosshair) and
  `near:[{id,dist}]` (blocks). The backend routes: a duck's name in the utterance > the current duck (sticky 20 s) >
  looked-at duck > nearest within 16 blocks > default. One GPT-Live call per duck (its own voice + persona, lazy,
  max 2 live, closed after 2 min idle). Replies: `{"type":"speaker","duck":id,"name"}` precedes that duck's PCM
  (played on its own LocationalAudioChannel); `transcript`/`clear`/`status` carry `duck` + `name`. A proto-1 mod gets
  plain PCM (plays at the default duck) and `Name: ` caption prefixes. Each duck's actions command only itself unless
  the player asks for all ducks.
