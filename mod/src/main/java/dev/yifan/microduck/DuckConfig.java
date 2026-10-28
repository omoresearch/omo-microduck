package dev.yifan.microduck;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Properties;

/** config/microduck.properties: anchor (block-centre MC coords), dimension, scale, sidecar port, enabled. Pure Java. */
public final class DuckConfig {
    public volatile boolean enabled = true;
    public volatile int port = 8796;
    public volatile double scale = 4.0;
    /** Name tags over the ducks (off unless turned on: !duck names on). */
    public volatile boolean names = false;
    /** Anchor A in MC coordinates (block centre x/z, block floor y); null dim means "not set". */
    public volatile double ax, ay, az;
    public volatile String dim;

    public boolean hasAnchor() {
        return dim != null && !dim.isEmpty();
    }

    /** Set the anchor from the player's feet block: centre of the block in x/z, its floor in y. */
    public void setAnchorBlock(int bx, int by, int bz, String dimension) {
        ax = bx + 0.5;
        ay = by;
        az = bz + 0.5;
        dim = dimension;
    }

    public static DuckConfig load(Path file) {
        DuckConfig c = new DuckConfig();
        if (!Files.exists(file)) {
            return c;
        }
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        } catch (IOException e) {
            return c;
        }
        c.enabled = Boolean.parseBoolean(p.getProperty("enabled", "true").trim());
        c.port = parseInt(p.getProperty("port"), 8796);
        c.names = Boolean.parseBoolean(p.getProperty("names", "false").trim());
        double s = parseDouble(p.getProperty("scale"), 4.0);
        c.scale = Math.max(DuckCommand.MIN_SCALE, Math.min(DuckCommand.MAX_SCALE, s));
        String d = p.getProperty("dim", "").trim();
        if (!d.isEmpty() && p.getProperty("anchor.x") != null) {
            c.ax = parseDouble(p.getProperty("anchor.x"), 0);
            c.ay = parseDouble(p.getProperty("anchor.y"), 0);
            c.az = parseDouble(p.getProperty("anchor.z"), 0);
            c.dim = d;
        }
        return c;
    }

    public void save(Path file) throws IOException {
        Properties p = new Properties();
        p.setProperty("enabled", Boolean.toString(enabled));
        p.setProperty("port", Integer.toString(port));
        p.setProperty("scale", fmt(scale));
        p.setProperty("names", Boolean.toString(names));
        if (hasAnchor()) {
            p.setProperty("anchor.x", fmt(ax));
            p.setProperty("anchor.y", fmt(ay));
            p.setProperty("anchor.z", fmt(az));
            p.setProperty("dim", dim);
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try (Writer w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
            p.store(w, "Microduck renderer (anchor = sim origin in MC coords; set in-game with !duck here)");
        }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private static String fmt(double d) {
        return String.format(Locale.ROOT, "%.4f", d).replaceAll("0+$", "").replaceAll("\\.$", ".0");
    }

    private static int parseInt(String s, int def) {
        try {
            return s == null ? def : Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static double parseDouble(String s, double def) {
        try {
            double v = s == null ? def : Double.parseDouble(s.trim());
            return Double.isFinite(v) ? v : def;
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
