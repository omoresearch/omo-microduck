#!/usr/bin/env bash
# One-shot setup for the Microduck MuJoCo sidecar.
#   ./setup.sh            fetch the Pollen Robotics robot (pinned), create .venv, install deps
#   ./setup.sh --no-brains   skip downloading the community duck brains (tricks, sprint, skates) from Hugging Face
# Re-running is safe (skips what is already there).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

UPSTREAM_REPO="https://github.com/pollen-robotics/microduck_rl"
UPSTREAM_REV="b02fdea4902c2b93b7d34a83a6f05da4ececb7d3"   # develop, 2026-10-12 (velstand + standup)
ROBOT_DIR="upstream/src/mjlab_microduck/robot/microduck"

say() { printf '\033[1;33m>>\033[0m %s\n' "$*"; }

# 1. Robot model (MJCF + meshes, Apache-2.0) -----------------------------------------------------
if [ ! -d upstream/.git ]; then
  say "cloning pollen-robotics/microduck_rl @ ${UPSTREAM_REV:0:7}"
  git clone --filter=blob:none --no-checkout "$UPSTREAM_REPO" upstream
fi
git -C upstream fetch -q --filter=blob:none origin "$UPSTREAM_REV" 2>/dev/null || true
git -C upstream -c advice.detachedHead=false checkout -q "$UPSTREAM_REV"
cp robot-overlay/scene_sim_*.xml "$ROBOT_DIR/"     # our scene wrappers (backlash robot velstand was trained on)

# 2. Python env ------------------------------------------------------------------------------------
PY="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv -q --python "${UV_PYTHON:-3.12}" .venv
  say "installing requirements (uv)"
  uv pip install -q --python .venv/bin/python -r requirements.txt
  # BAM pins python<3.13 but is pure numpy
  uv pip install -q --python .venv/bin/python --no-deps \
    "better-actuator-models @ https://github.com/Rhoban/bam/archive/refs/heads/mjlab_frictionloss.tar.gz"
else
  [ -d .venv ] || "$PY" -m venv .venv
  say "installing requirements (pip)"
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/pip install -q --no-deps --ignore-requires-python \
    "https://github.com/Rhoban/bam/archive/refs/heads/mjlab_frictionloss.tar.gz"
fi

# 3. Community brains (~28 MB) --------------------------------------------------------------------
if [ "${1:-}" != "--no-brains" ]; then
  say "fetching community brains (ONNX + manifest only; nothing from a repo is executed)"
  .venv/bin/python brains_fetch.py
fi

say "smoke test"
OMP_NUM_THREADS=1 .venv/bin/python - <<'EOF'
import duck_sim
s = duck_sim.DuckSim()
for _ in range(100):
    s.step()
print("ok: sim stepped 2 s, mode", s.mode, "fallen", s.fallen)
EOF
say "done. Run:  OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python server.py   (127.0.0.1:8796)"
