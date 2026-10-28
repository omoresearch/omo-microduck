package dev.yifan.microduck;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/** config/microduck-voice.properties: the Simple Voice Chat bridge to the duck's GPT-Live voice. Pure Java. */
public final class DuckVoiceConfig {
    public volatile boolean enabled = true;
    /** Voice backend (voice-live/duck.mjs). Loopback only. */
    public volatile String url = "ws://127.0.0.1:8798/svc";
    /** Only the local player's mic within this many blocks of the duck is sent; 0 = always (any distance/dimension). */
    public volatile double hearRange = 16.0;
    /** Optional player name to listen to instead of the local (host) player. */
    public volatile String player = "";
    /** How far the duck's voice carries (SVC locational channel distance, blocks). */
    public volatile float voiceDistance = 24f;
    /** Show what the duck says (and what it heard) in chat. */
    public volatile boolean captions = true;

    public static DuckVoiceConfig load(Path file) {
        DuckVoiceConfig c = new DuckVoiceConfig();
        if (!Files.exists(file)) {
            return c;
        }
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        } catch (IOException e) {
            return c;
        }
        c.enabled = !"false".equalsIgnoreCase(p.getProperty("enabled", "true").trim());
        String u = p.getProperty("url", c.url).trim();
        if (u.startsWith("ws://127.0.0.1:") || u.startsWith("ws://localhost:")) {
            c.url = u; // loopback only: this carries the player's microphone
        }
        String range = p.getProperty("hear_range", "16").trim();
        c.hearRange = "always".equalsIgnoreCase(range) ? 0 : clamp(parse(range, 16), 0, 256);
        c.player = p.getProperty("player", "").trim();
        c.voiceDistance = (float) clamp(parse(p.getProperty("voice_distance", "24"), 24), 4, 128);
        c.captions = !"false".equalsIgnoreCase(p.getProperty("captions", "true").trim());
        return c;
    }

    public void save(Path file) throws IOException {
        Properties p = new Properties();
        p.setProperty("enabled", Boolean.toString(enabled));
        p.setProperty("url", url);
        p.setProperty("hear_range", hearRange <= 0 ? "always" : Double.toString(hearRange));
        p.setProperty("player", player);
        p.setProperty("voice_distance", Float.toString(voiceDistance));
        p.setProperty("captions", Boolean.toString(captions));
        Files.createDirectories(file.getParent());
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            p.store(w, "Microduck voice (Simple Voice Chat push-to-talk near the duck -> GPT-Live duck persona; reply plays from the duck)."
                    + " hear_range: blocks or 'always'. player: blank = you (the local player).");
        }
    }

    /** Whether a mic packet from a player at distance {@code dist} (blocks, NaN = other dimension) should reach the duck. */
    public boolean inRange(double dist) {
        if (hearRange <= 0) {
            return true;
        }
        return !Double.isNaN(dist) && dist <= hearRange;
    }

    private static double parse(String s, double def) {
        try {
            return Double.parseDouble(s.trim());
        } catch (RuntimeException e) {
            return def;
        }
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
