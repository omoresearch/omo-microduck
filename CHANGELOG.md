# Changelog

## v1.0.0 (2026-10-28): First flight

The first public release of Omo Microduck.

- **Sidecar**: Pollen Robotics' Microduck in MuJoCo with the BAM XL330 servo
  model and backlash. `velstand` handles walk, turn, follow and getting up;
  `alpha_sitstand` handles sit and stand. Also a heading hold, a leash and
  auto-reset. Loopback HTTP API on 127.0.0.1:8796.
- **Mod** (Fabric, MC 26.2, client-only): renders the duck articulated at 25.5k
  triangles with smooth normals and baked AO, interpolates poses, shows a blob
  shadow, and works with Sodium and Iris. `!duck` chat commands, quack sounds.
- **Terrain**: real blocks become MuJoCo collision boxes, live-updated as you
  build.
- **Multiple ducks**: up to 4 (8 max), with personas Quackers, Rusty, Zippy,
  Plato and Newton, each with a name tag, eye-ring colour and quack pitch.
- **Community brains**: 12 featured Hugging Face policies (sprint, hill,
  roller skates, roulade, cartwheel, kick, hop, groove, flamingo, play dead,
  dash, nugget), each graded in the sim. Download and validation happen in
  `setup.sh`.
- **Voice** (optional): a Simple Voice Chat plugin with positional replies from
  each duck's head and captions. The backend is not included yet.
- **Twitch-safe**: chat commands from Twitch are limited to moves and tricks.
