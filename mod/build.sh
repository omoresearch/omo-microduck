#!/usr/bin/env bash
# Build the Microduck client mod with plain javac (MC 26.2 ships unobfuscated).
#   ./build.sh            build only
#   ./build.sh install    build + install into $MC_MODS (takes effect next launch)
#
# Paths (override with env vars; defaults match an Omo Space install):
#   JDK_BIN        dir with javac/jar/java (JDK 21+)            default: ~/.local/share/omo/toolchain/jdk25/bin, else PATH
#   MC_LIBRARIES   launcher libraries dir holding minecraft-26.2-client.jar etc. (Prism/MultiMC "libraries")
#   MC_MODS        the instance's mods/ dir (fabric-api-*.jar is read from here; install target)
#   FABRIC_API_JAR explicit fabric-api jar (default: first $MC_MODS/fabric-api-*.jar)
#   ./build.sh test       build + compile and run the standalone test harness (test/)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
J="${JDK_BIN:-$HOME/.local/share/omo/toolchain/jdk25/bin}"
[ -x "$J/javac" ] || J="$(dirname "$(command -v javac)")"
L="${MC_LIBRARIES:-$HOME/.local/share/omo/runtime/prism/libraries}"
MODS="${MC_MODS:-$HOME/.local/share/omo/runtime/game/mods}"
FAPI="${FABRIC_API_JAR:-$(ls "$MODS"/fabric-api-*.jar | head -1)}"
BACKUPS="${MC_BACKUPS:-$(dirname "$MODS")/../backups}"
NAME=microduck-1.0.0+mc26.2.jar

rm -rf build && mkdir -p build/fapi build/classes
"$J/jar" --extract --file "$FAPI" --dir build/fapi 'META-INF/jars/' 2>/dev/null || true
CP="$L/com/mojang/minecraft/26.2/minecraft-26.2-client.jar"
for f in build/fapi/META-INF/jars/*.jar \
         "$L"/net/fabricmc/fabric-loader/*/*.jar \
         "$L"/com/mojang/datafixerupper/*/*.jar \
         "$L"/com/mojang/brigadier/*/*.jar "$L"/com/mojang/logging/*/*.jar \
         "$L"/com/google/guava/guava/*/*.jar "$L"/com/google/code/gson/gson/*/*.jar \
         "$L"/org/joml/joml/*/*.jar "$L"/it/unimi/dsi/fastutil/*/*.jar \
         "$L"/org/slf4j/slf4j-api/*/*.jar "$L"/org/jspecify/jspecify/*/*.jar; do
  [ -f "$f" ] && CP="$CP:$f"
done
# Simple Voice Chat API: compile-only (SVC is optional at runtime; deps/fetch.py downloads + verifies it)
SVC_API="$(ls "$HERE"/deps/voicechat-api-*.jar 2>/dev/null | head -1)"
SVC_JAR="$(ls "$HERE"/deps/voicechat-fabric-*.jar 2>/dev/null | head -1)"
[ -n "$SVC_API" ] && CP="$CP:$SVC_API"
SRCS="$(find src/main/java -name '*.java')"
[ -n "$SVC_API" ] || { echo "deps/voicechat-api-*.jar missing (run deps/fetch.py): building without the voice chat plugin"; SRCS="$(echo "$SRCS" | grep -v DuckVoicePlugin.java)"; }
"$J/javac" --release 21 -proc:none -Xlint:-options -cp "$CP" -d build/classes $SRCS
cp -r src/main/resources/. build/classes/
OUT="build/$NAME"
"$J/jar" --create --file "$OUT" -C build/classes .
echo "built $OUT ($(stat -c %s "$OUT") bytes)"

if [ "${1:-}" = test ]; then
  GSON="$(ls "$L"/com/google/code/gson/gson/*/*.jar | head -1)"
  mkdir -p build/test
  # the harness sees only the pure-Java classes (no Minecraft/Fabric on its classpath)
  "$J/javac" --release 21 -proc:none -cp "build/classes:$GSON" -d build/test $(find test -name '*.java')
  "$J/java" -cp "build/test:build/classes:$GSON" dev.yifan.microduck.Harness "${2:-http://127.0.0.1:8796}"
fi

if [ "${1:-}" = voicetest ]; then
  # ./build.sh voicetest [ws://127.0.0.1:8798/svc mic.pcm ...]: PCM framing + real SVC Opus round trip (Concentus and
  # native opus4j from the SVC jar in deps/), and optionally a live run against the duck voice backend.
  GSON="$(ls "$L"/com/google/code/gson/gson/*/*.jar | head -1)"
  LOG4J="$(ls "$L"/org/apache/logging/log4j/log4j-api/*/*.jar | head -1):$(ls "$L"/org/apache/logging/log4j/log4j-core/*/*.jar | head -1)"
  mkdir -p build/test-voice
  "$J/javac" --release 21 -proc:none -cp "build/classes:$GSON:$SVC_JAR" -d build/test-voice $(find test-voice -name '*.java')
  "$J/java" -cp "build/test-voice:build/classes:$GSON:$SVC_JAR:$LOG4J" dev.yifan.microduck.VoiceTests "${@:2}"
fi

if [ "${1:-}" = install ]; then
  for old in "$MODS"/microduck-*.jar; do
    if [ -e "$old" ] && [ "$(basename "$old")" != "$NAME" ]; then
      mkdir -p "$BACKUPS"   # backups live outside mods/ (no duplicate loading)
      mv "$old" "$BACKUPS/$(basename "$old").replaced"
    fi
  done
  cp "$OUT" "$MODS/.microduck.tmp" && mv "$MODS/.microduck.tmp" "$MODS/$NAME"
  sha256sum "$MODS/$NAME"
fi
