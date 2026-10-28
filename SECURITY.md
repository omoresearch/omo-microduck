# Security

Omo Microduck is local-first. Nothing in it talks to the internet at runtime.

## The model

- The sidecar listens on **127.0.0.1 only** (port 8796). It has no auth, so do
  not port-forward or reverse-proxy it.
- The mod is **client-side**. `!duck` chat lines are intercepted and cancelled
  on your client; they never reach the server.
- The optional voice link only accepts a **loopback** WebSocket URL.
- Commands tagged `from: "twitch:*"` cannot `reset`, `spawn`, `remove`, move the
  anchor or change the scale. Chat can make the duck walk; it cannot touch your
  setup.
- Community brains are treated as **untrusted data**. `brains_fetch.py`
  downloads only `policy.onnx`, `manifest.json` and `README.md`, never runs code
  from a model repo, and validates each graph's shapes before it can be used.
  A brain that throws inside onnxruntime is dropped, and the sim keeps running.

## Reporting

Please report vulnerabilities privately to **harry@omo.space** with the subject
"Omo security", not in a public issue. Include the version, what an attacker
needs, and how to reproduce it. We aim to reply within a few days.
