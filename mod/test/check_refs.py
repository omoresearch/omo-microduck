#!/usr/bin/env python3
"""Check every Minecraft/Fabric/JOML member referenced by the mod's classes exists (name + descriptor) in the
real target jars (MC 26.2 client jar, fabric-api nested jars, loader, joml, slf4j). Uses javap only."""
import glob
import os
import re
import subprocess
import sys

H = os.path.expanduser("~")
J = os.path.join(os.environ.get("JDK_BIN", H + "/.local/share/omo/toolchain/jdk25/bin"), "javap")
L = os.environ.get("MC_LIBRARIES", H + "/.local/share/omo/runtime/prism/libraries")
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..", "build")
cp = [L + "/com/mojang/minecraft/26.2/minecraft-26.2-client.jar"]
cp += glob.glob(B + "/fapi/META-INF/jars/*.jar")
for pat in ["net/fabricmc/fabric-loader/*/*.jar", "org/joml/joml/*/*.jar", "org/slf4j/slf4j-api/*/*.jar",
            "com/google/code/gson/gson/*/*.jar"]:
    cp += glob.glob(L + "/" + pat)
CP = ":".join(cp)

refs = set()
for cls in sorted(glob.glob(B + "/classes/dev/yifan/microduck/*.class")):
    out = subprocess.run([J, "-v", "-p", cls], capture_output=True, text=True).stdout
    for m in re.finditer(r"= (Methodref|InterfaceMethodref|Fieldref)\s+#\d+\.#\d+\s+// (\S+)", out):
        kind, ref = m.groups()
        owner, rest = ref.split(".", 1)
        name, desc = rest.split(":", 1)
        name = name.strip('"')
        if owner.startswith(("net/minecraft", "com/mojang", "net/fabricmc", "org/joml", "org/slf4j", "com/google/gson")):
            refs.add((owner, name, desc, kind))

cache = {}
bad = 0
for owner, name, desc, kind in sorted(refs):
    if owner not in cache:
        r = subprocess.run([J, "-s", "-p", "-cp", CP, owner.replace("/", ".")], capture_output=True, text=True)
        # include supertypes' members by resolving through javap of the hierarchy
        text = r.stdout
        sup = re.search(r"(?:extends|implements) ([^{]+)\{", text)
        seen = {owner}
        queue = []
        if sup:
            queue += [s.strip().split("<")[0] for s in re.split(r",|extends|implements", sup.group(1)) if s.strip()]
        while queue:
            s = queue.pop()
            if s in seen or s.startswith("java."):
                continue
            seen.add(s)
            r2 = subprocess.run([J, "-s", "-p", "-cp", CP, s], capture_output=True, text=True)
            text += r2.stdout
            m2 = re.search(r"(?:extends|implements) ([^{]+)\{", r2.stdout)
            if m2:
                queue += [x.strip().split("<")[0] for x in re.split(r",|extends|implements", m2.group(1)) if x.strip()]
        cache[owner] = text
    text = cache[owner]
    found = False
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if i + 1 < len(lines) and ("descriptor: " + desc) == lines[i + 1].strip():
            if re.search(r"[ .]" + re.escape(name) + r"(\(|;| =)", line) or (name == "<init>" and "(" in line):
                found = True
                break
    status = "ok " if found else "MISSING"
    if not found:
        bad += 1
    print(f"{status} {kind:18} {owner.replace('/', '.')}.{name} {desc}")
print(f"\n{len(refs)} references checked, {bad} missing")
sys.exit(1 if bad else 0)
