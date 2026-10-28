#!/usr/bin/env python3
"""Check every Simple Voice Chat API member DuckVoicePlugin uses exists in deps/voicechat-api-*.jar (javap)."""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(os.environ.get("JDK_BIN", os.path.expanduser("~/.local/share/omo/toolchain/jdk25/bin")), "javap")
api = glob.glob(os.path.join(HERE, "..", "deps", "voicechat-api-*.jar"))[0]
cls = os.path.join(HERE, "..", "build", "classes")
out = subprocess.run([J, "-v", "-cp", cls, "dev.yifan.microduck.DuckVoicePlugin"], capture_output=True, text=True).stdout
refs = set(re.findall(r"(?:Interface)?Methodref\s+#\d+\.#\d+\s+// (de/maxhenkel/[\w/$]+)\.([\w<>]+):(\S+)", out))
cache, missing = {}, 0
for owner, name, desc in sorted(refs):
    c = owner.replace("/", ".")
    if c not in cache:
        cache[c] = subprocess.run([J, "-s", "-cp", api, c], capture_output=True, text=True).stdout
    body = cache[c]
    # javap -s prints "descriptor: (..)..." after each member line containing the name
    ok = re.search(rf"\b{re.escape(name)}\(.*\n\s+descriptor: {re.escape(desc)}", body) is not None
    todo = [] if ok else [body]
    while todo and not ok:  # inherited (ServerPlayer -> Player -> Entity, VoicechatServerApi -> VoicechatApi)
        b = todo.pop()
        for sup in re.findall(r"extends ([\w.,$]+)", b.split("{")[0]):
            for s in sup.split(","):
                b2 = subprocess.run([J, "-s", "-cp", api, s], capture_output=True, text=True).stdout
                if re.search(rf"\b{re.escape(name)}\(.*\n\s+descriptor: {re.escape(desc)}", b2):
                    ok = True
                todo.append(b2)
    missing += not ok
    print(f"{'ok ' if ok else 'MISSING'} {c}.{name} {desc}")
print(f"\n{len(refs)} SVC API references checked against {os.path.basename(api)}, {missing} missing")
sys.exit(1 if missing else 0)
