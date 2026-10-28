# Microduck client mod (Minecraft 26.2, Fabric)

Renders the Pollen Robotics **Microduck** robot, articulated, from the local
`microduck-sim` sidecar (`../sidecar`, HTTP on `127.0.0.1:8796`) at a fixed spot in
the world, and turns `!duck ...` chat lines into sidecar commands. Client-only; nothing is sent
to the server. Contract: [`../sidecar/CONTRACT.md`](../sidecar/CONTRACT.md).

## Build / install / test

```
./build.sh                     # javac against the unobfuscated 26.2 client jar + fabric-api nested jars
./build.sh install             # + atomic copy to runtime/game/mods/microduck-1.0.0+mc26.2.jar, prints sha256
./build.sh test [URL]          # + standalone harness (test/, not in the jar) against URL (default :8796)
python3 test/fake_sidecar.py 8797   # stand-in sidecar for the harness when the real one is down
python3 test/check_refs.py     # every MC/Fabric member the mod calls exists in the target jars (javap)
```

The jar only loads on the next game launch. Older `microduck-*.jar` files are moved to
`runtime/backups/` by `install`, so mods/ never holds two copies.

## Chat commands (typed in MC chat; intercepted, never sent to the server)

| command | effect |
|---|---|
| `!duck` / `!duck help` | help |
| `!duck walk\|stop\|sit\|stand\|follow\|quack\|reset` | `POST /cmd {"cmd":..., "from":"mc"}` |
| `!duck left\|right` | `POST /cmd {"cmd":"turn","dir":...}` |
| `!duck here` | anchor = your feet block (x/z centre, floor y) + dimension; saved |
| `!duck scale <0.5-10>` | MC blocks per sim metre (default 4: the 25 cm duck is ~1 block); saved |
| `!duck <name> <cmd>` | one duck, e.g. `!duck zippy trick roulade`, `!duck rusty follow zippy` (`POST /cmd {"duck":"zippy",...}`) |
| `!ducks <cmd>` | every duck (`"duck":"all"`) |
| `!duck spawn [persona]` | spawn that persona, or the next unused one, at a free spot near the anchor (`POST /ducks`) |
| `!duck spawn all` | spawn every persona not already spawned, up to the max (default 4), each at its own free spot |
| `!duck remove <name>` / `!duck remove all` | remove one duck / every duck (nothing is kept; `!duck spawn ...` brings them back) |
| `!duck list` | every duck: name, id, mode, distance |
| `!duck brains`, `!duck brain <id>`, `!duck trick <id>` | community brains (see the sidecar README) |

Plain `!duck <cmd>` targets the duck nearest to you (the only one if there is one). Names are case-insensitive and
an unambiguous prefix works (`!duck zip walk`); the sidecar resolves them. Spawn/remove/here/scale are MC-chat only
(never Twitch). Replies show as `[Duck] ...` / `[Zippy] ...` client system messages.

## Multiple ducks

The mod polls `GET /poses` (all ducks, roster order) at 25 Hz; on an older single-duck sidecar (404) it falls back
to `GET /pose` and renders the one duck exactly as before (no name tag). Each duck gets its persona accent colour on
the eye ring (mesh `noenoeil`, recoloured keeping its baked shading), a vanilla-style floating name tag within 16
blocks, and its own quack (per-duck counter, persona `quack_pitch`). Terrain scanning is shared (one scan, all ducks).
Voice: push-to-talk sends `look` (the duck under your crosshair) and `near` (every duck's distance) to the voice
backend, and each duck's reply plays from its own SVC LocationalAudioChannel at its head.

## Config: `runtime/game/config/microduck.properties`

`anchor.x/y/z` (MC coords of the sim origin), `dim` (e.g. `minecraft:overworld`), `scale`,
`port` (8796), `enabled` (false = no rendering, no network). Edit while the game is closed or use
the chat commands. Nothing renders until an anchor is set (`!duck here`).

## How it works

* **Network** (`SidecarClient`, one low-priority daemon thread, `java.net.http`): fetch `/model.json`
  (retry 1 s -> 15 s backoff), poll `/pose` at ~25 Hz, `POST /player` at ~5 Hz while you are in the
  anchor's dimension. The render/tick threads only read a volatile snapshot; HTTP never blocks
  them. Sidecar down -> one log line, the duck disappears, it reconnects (and refetches the model)
  quietly.
* **Mapping** (`DuckMath`): MC = A + s·C·p with C(x,y,z) = (x, z, -y); rotations R_mc = C R C^T, i.e.
  quaternion [w, x, z, -y]. Player MC (px, pz) -> sim ((px-ax)/s, -(pz-az)/s).
* **Rendering**: `LevelRenderEvents.COLLECT_SUBMITS` (fabric-rendering-v1) -> per body
  `SubmitNodeCollector.submitCustomGeometry(poseStack, RenderTypes.entityCutout(microduck:textures/misc/white.png), ...)`.
  When model.json carries the optional per-mesh `normals` / `colors` / `cull` fields (sidecar model_export v2),
  meshes use crease-split smooth per-vertex normals, per-vertex colours (real-robot palette + baked AO), and
  closed meshes go through `RenderTypes.entityCutoutCull` (back faces culled); adjacent triangle pairs are emitted
  as one real quad (exact same triangles under the (0,1,2)(2,3,0) quad split), ~16k quads for ~25.5k triangles.
  Without those fields (old sidecar): per-triangle flat normals, MJCF rgba, double-sided degenerate quads.
  A vanilla-style blob shadow (`entityShadow(textures/misc/shadow.png)`) is drawn on the anchor floor when entity
  shadows are on and no Iris shader pack is active. Note: with an Iris shader pack, Iris recomputes a face normal
  per entity quad, so shading is per-quad there (colours/AO/details still apply). Block/sky light sampled at the duck's
  feet, poses interpolated (lerp + shortest-arc nlerp) between the last two sidecar poses. Skipped
  when > 64 blocks from the camera, outside the frustum, in another dimension, or without anchor.
  `entity_cutout` is vanilla's no-cull mob pipeline, so Sodium (which leaves entity rendering to
  vanilla) and Iris (which maps `ENTITY_CUTOUT` to `gbuffers_entities` and hooks custom-geometry
  submits) both handle it like any mob.
* **Terrain** (`DuckTerrain` MC glue; pure-Java `TerrainScan` + `TerrainGrid`, tested in the harness): every 2 s
  the client thread scans block collision shapes (`BlockState.getCollisionShape` -> `VoxelShape.toAabbs()`) in
  leash·s+3 blocks around the anchor (4 below, a few above), <= 8192 cells / 0.7 ms per tick; an unloaded chunk
  aborts the pass. If the content hash changed, a daemon thread merges cells greedily (full blocks in 3-D,
  slabs/carpets in x/z, multi-box shapes per box; cap 2000 nearest), converts to sim metres and `POST /terrain`s;
  every 10 s it checks `GET /terrain` still has its id (sidecar restarted -> resend). Chat says
  `[Duck] terrain: 27x8x27 blocks -> N boxes` after `!duck here` / `!duck scale`.
* **Quack**: when `pose.quack` increases, plays `microduck:quack` (3 variants) at the trunk.
  The oggs were rendered with Pollen Robotics' own voice synth (`microduck/sounds/src`,
  "greet" recipe, seed 1234, Apache-2.0, licence in `assets/microduck/sounds/`) and converted
  with ffmpeg.

## What to check in-game after the next restart

1. `logs/latest.log`: `microduck: sidecar http://127.0.0.1:8796 ...` and `sidecar ... connected (model: N bodies, M triangles)`.
2. Stand on the lab floor, `!duck here` -> `[Duck] anchor set to ...`. The duck appears ~1 block tall.
3. `!duck walk`, `!duck left`, `!duck stop`, `!duck follow` (it walks toward you inside the leash), `!duck quack` (sound).
4. With an Iris shader pack on and off: duck is visible, coloured, lit like a mob, no z-fighting/holes.
5. `!duck` in chat must not show up for other players / the server log (it is cancelled client-side).
6. Terrain: after `!duck here`, chat shows `[Duck] terrain: ... -> N boxes` (log: `microduck: terrain ...`);
   `curl -s 127.0.0.1:8796/terrain` shows the same box count. The duck's feet sit on the real floor (also on a slab
   floor), `!duck walk` toward a wall: it stops/turns instead of walking through; walk it off a 1-block ledge: it
   falls, then gets up or resets to the anchor. Place/break a block near it: within ~2-3 s the sim updates
   (`builds` in `/terrain` increments) without the duck jumping; no frame hitches while that happens.
7. Multiple ducks: the existing duck is still there as Quackers (yellow eye ring, name tag "Quackers"). `!duck spawn all`
   -> `[Duck] spawned Rusty (arbor), Zippy (maple), ...`; each appears at its own free spot with its own eye-ring colour
   and name tag. `!duck zippy walk` moves only Zippy; `!ducks sit` sits all; `!duck list`; `!duck remove all` -> no
   ducks; `!duck spawn` brings Quackers back. Push-to-talk "hey Zippy, ..." -> Zippy answers from Zippy's body with
   a `[Zippy] ...` caption.

## Uninstall

Close the game, delete `runtime/game/mods/microduck-1.0.0+mc26.2.jar` (and optionally
`runtime/game/config/microduck.properties`).

## Voice: talk to the duck with Simple Voice Chat (optional)

With Simple Voice Chat installed (`deps/voicechat-fabric-2.6.26+26.2.jar`, fetched + sha512-checked against Modrinth
by `deps/fetch.py`; not installed automatically), the mod's SVC plugin (`DuckVoicePlugin`, entrypoint `voicechat`,
listed under `suggests` so the mod loads fine without SVC) runs on the integrated server:

- Hold your SVC push-to-talk within `hear_range` (16) blocks of the duck. Only your own player is heard.
- Shared mic with the GLaDOS mod: while `System.getProperty("omo.voice.focus")` is `glados:<ms>` and < 1 s old
  (glados-mod claims it when you are near her and looking at her / closer to her), the duck ignores your mic.
- Mic Opus -> PCM (API decoder) -> `ws://127.0.0.1:8798/svc` (a GPT-Live voice backend; not part of this repo yet, protocol in `../sidecar/CONTRACT.md`)
  -> GPT-Live duck persona (moves the duck through the sidecar) -> reply PCM -> API encoder -> 20 ms frames on a
  LocationalAudioChannel that follows the duck. It is game audio (SVC volume category "Microduck"), so OBS's game /
  desktop capture records it. Captions of what the duck says go to chat (`captions=false` to hide).
- Config `runtime/game/config/microduck-voice.properties`: `enabled`, `url` (loopback only), `hear_range` (blocks or
  `always`), `player` (blank = you), `voice_distance` (24), `captions`.
- Tests without the game: `./build.sh voicetest` (PCM framing, config, Opus round trip through SVC's bundled Concentus
  and native opus4j) and `./build.sh voicetest ws://127.0.0.1:8798/svc a.pcm b.pcm` (live: the plugin's exact
  encode/decode/link pipeline against the running backend). `python3 test-voice/check_svc_refs.py` javaps every SVC API
  member the plugin calls.
- In-game check after installing SVC + this jar and restarting: log shows `microduck voice: Simple Voice Chat plugin
  loaded` and `voice chat server started`; `[Duck] ...` captions after you speak; the voice comes from the duck and
  gets quieter as you walk away; the duck nods while talking.
