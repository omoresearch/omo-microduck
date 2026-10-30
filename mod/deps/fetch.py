#!/usr/bin/env python3
"""Fetch Simple Voice Chat (Fabric, MC 26.2) + its compile-only API jar into this deps/ dir and verify them.

  SVC jar : Modrinth API /v2/project/simple-voice-chat/version -> sha512 from Modrinth
  API jar : maven.maxhenkel.de de.maxhenkel.voicechat:voicechat-api -> .sha1 from the Maven repo

Never installs anything into the game's mods dir.
"""
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SVC_VERSION = os.environ.get("SVC_VERSION", "fabric-2.6.26+26.2")
API_VERSION = os.environ.get("SVC_API_VERSION", "2.6.25")
# Expected hashes for the default versions, checked in here so a changed file on either server is refused
# even if that server's own hash listing changed with it. Overriding the versions skips the pin.
PINNED = {
    "voicechat-fabric-2.6.26+26.2.jar": ("sha512", "028ecc1be6c910d688f884e9e94401b9ce43ca69eb0b120947390be15a4871863190e664aeaa5f5e3517dededa15fecfb880b3dd782094b5c3d2db2288726b6e"),
    "voicechat-api-2.6.25.jar": ("sha1", "3a3cb5bf0ed7f15abaa0f8854f96c193baa65b48"),
}
MAVEN = "https://maven.maxhenkel.de/repository/public/de/maxhenkel/voicechat/voicechat-api"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "microduck-mod-deps/1.0 (local build)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch(url, path, algo, want):
    if not os.path.exists(path):
        data = get(url)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    got = hashlib.new(algo, open(path, "rb").read()).hexdigest()
    ok = got == want
    print(f"{'OK ' if ok else 'BAD'} {os.path.basename(path)} {algo}={got[:32]}... (expected {want[:32]}...)")
    return ok


def main():
    versions = json.loads(get("https://api.modrinth.com/v2/project/simple-voice-chat/version"
                              "?loaders=%5B%22fabric%22%5D&game_versions=%5B%2226.2%22%5D"))
    v = next((x for x in versions if x["version_number"] == SVC_VERSION), None)
    if v is None:
        sys.exit(f"{SVC_VERSION} not on Modrinth; have {[x['version_number'] for x in versions[:5]]}")
    f = next(x for x in v["files"] if x["primary"]) if any(x.get("primary") for x in v["files"]) else v["files"][0]
    want1 = f["hashes"]["sha512"]
    if f["filename"] in PINNED and PINNED[f["filename"]][1] != want1:
        sys.exit(f"Modrinth lists a different hash for {f['filename']} than the pinned one: refusing")
    ok1 = fetch(f["url"], os.path.join(HERE, f["filename"]), "sha512", want1)
    print(f"    published {v['date_published']} game_versions={v['game_versions']} loaders={v['loaders']}")
    name = f"voicechat-api-{API_VERSION}.jar"
    sha1 = PINNED[name][1] if name in PINNED else get(f"{MAVEN}/{API_VERSION}/{name}.sha1").decode().split()[0]
    ok2 = fetch(f"{MAVEN}/{API_VERSION}/{name}", os.path.join(HERE, name), "sha1", sha1)
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
