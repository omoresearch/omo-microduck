# Contributing

Issues and PRs are welcome, especially new duck brains, renderer fixes and
macOS test reports.

- **Sidecar changes**: run `sidecar/test_headless.py` and `sidecar/test_brains.py`
  (under `nice`, because they are CPU-bound). For anything touching multiple
  ducks, also run `sidecar/test_multi.py`.
- **Mod changes**: run `mod/build.sh test http://127.0.0.1:8806` against a
  test sidecar (`MICRODUCK_TEST=1 .venv/bin/python server.py --port 8806`).
  Don't point the harness at a sidecar you are playing with: it posts terrain
  and spawns and removes ducks.
- **New brains**: add an entry in `sidecar/brains.py`, then run
  `brains.py eval <id>` and include the catalogue diff in your PR. A brain is
  featured only if it passes or wobbles over 3 seeds.
- **API changes**: update `sidecar/CONTRACT.md` in the same PR.

Keep the sidecar loopback-only and the mod client-side.
