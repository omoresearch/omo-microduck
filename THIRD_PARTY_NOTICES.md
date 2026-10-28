# Third-party notices

Omo Microduck's own code (`mod/`, `sidecar/`) is MIT licensed (see [LICENSE](LICENSE)).
It builds on the following work. Thank you.

| Component | By | Licence | How it is used |
|---|---|---|---|
| [Microduck robot: MJCF + meshes](https://github.com/pollen-robotics/microduck_rl) | Pollen Robotics | Apache-2.0 | Fetched by `sidecar/setup.sh` at a pinned commit; not redistributed here. `sidecar/robot-overlay/` holds two small scene wrappers that include the upstream robot files. |
| [Microduck policies](https://huggingface.co/pollen-robotics/microduck-policies) (`velstand`, `alpha_sitstand`, `alpha_stand`, `alpha_walking`) | Pollen Robotics | Apache-2.0 | Redistributed unmodified in `sidecar/policies/` with their manifest and model card. |
| Quack sounds (`mod/.../sounds/quack*.ogg`) | Rendered with Pollen Robotics' Microduck voice synth | Apache-2.0 | Licence text in `mod/src/main/resources/assets/microduck/sounds/`. |
| Community duck brains (Hugging Face `*/microduck-*`) | Their authors (see `sidecar/brains/catalogue.json`) | As stated on each model card; many are unspecified | **Not redistributed.** Downloaded on your machine by `brains_fetch.py`. Check a card before reusing a brain elsewhere. |
| [better-actuator-models (BAM)](https://github.com/Rhoban/bam) | Rhoban | Apache-2.0 | Installed by `setup.sh`; M6 XL330 servo model. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Google DeepMind | Apache-2.0 | Physics (pip dependency). |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Microsoft | MIT | Policy inference (pip dependency). |
| [fast-simplification](https://github.com/pyvista/fast-simplification) | PyVista | MIT | Mesh decimation (pip dependency). |
| [Fabric Loader / Fabric API](https://fabricmc.net) | FabricMC | Apache-2.0 | Compile and runtime dependency; not bundled. |
| [Simple Voice Chat API](https://modrinth.com/plugin/simple-voice-chat) | Max Henkel | see its project page | Optional, compile-only; fetched and hash-checked by `mod/deps/fetch.py`; not bundled. |

The Microduck is a robot designed by [Pollen Robotics](https://www.pollen-robotics.com).
This project is an independent community project, not affiliated with or endorsed by
Pollen Robotics.

NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR
MICROSOFT. Minecraft is a trademark of Mojang/Microsoft.
