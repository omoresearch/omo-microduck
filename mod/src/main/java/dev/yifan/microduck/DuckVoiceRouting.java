package dev.yifan.microduck;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Multi-duck voice routing, pure Java (unit-tested in the harness; no Minecraft / Simple Voice Chat types).
 * <ul>
 *   <li>{@link #lookTarget}: the duck the player's crosshair is on (sent with push-to-talk so the backend knows who
 *       is being addressed).</li>
 *   <li>{@link #pttJson}: the {@code ptt on} control message with {@code look} and {@code near} (distances in blocks).</li>
 *   <li>{@link Speakers}: reply audio arrives tagged by {@code {"type":"speaker","duck":id}} control messages; the
 *       binary PCM that follows goes to that duck's queue, so each duck talks from its own body.</li>
 * </ul>
 */
public final class DuckVoiceRouting {
    private DuckVoiceRouting() {
    }

    /** A duck's head position in MC coords. */
    public record Head(String id, double x, double y, double z) {
    }

    /**
     * Duck under the crosshair: the smallest angle between the view vector and the eye->head direction, within
     * {@code maxDist} blocks and an angular radius of ~0.45 duck heights (at least 4 degrees). Null if none.
     */
    public static String lookTarget(double ex, double ey, double ez, double vx, double vy, double vz, List<Head> heads,
                                    double scale, double maxDist) {
        double vn = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vn < 1e-9) {
            return null;
        }
        String best = null;
        double bestAng = Double.MAX_VALUE;
        for (Head h : heads) {
            double dx = h.x - ex, dy = h.y - ey, dz = h.z - ez;
            double d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d < 1e-6 || d > maxDist) {
                continue;
            }
            double cos = (dx * vx + dy * vy + dz * vz) / (d * vn);
            double ang = Math.acos(Math.max(-1, Math.min(1, cos)));
            double lim = Math.max(Math.toRadians(4), Math.atan2(0.45 * 0.25 * scale + 0.3, d));
            if (ang < lim && ang < bestAng) {
                bestAng = ang;
                best = h.id;
            }
        }
        return best;
    }

    /** {"type":"ptt","on":true,"look":id|null,"near":[{"id":..,"dist":..}]} (dist in blocks from the player). */
    public static String pttJson(String look, double px, double py, double pz, List<Head> heads) {
        StringBuilder b = new StringBuilder("{\"type\":\"ptt\",\"on\":true,\"look\":");
        b.append(look == null ? "null" : "\"" + esc(look) + "\"").append(",\"near\":[");
        int i = 0;
        for (Head h : heads) {
            double d = Math.sqrt((h.x - px) * (h.x - px) + (h.y - py) * (h.y - py) + (h.z - pz) * (h.z - pz));
            b.append(i++ > 0 ? "," : "").append("{\"id\":\"").append(esc(h.id)).append("\",\"dist\":")
                    .append(String.format(Locale.ROOT, "%.2f", d)).append('}');
        }
        return b.append("]}").toString();
    }

    /** Nearest head distance (blocks), NaN if none. */
    public static double nearest(double px, double py, double pz, List<Head> heads) {
        double best = Double.NaN;
        for (Head h : heads) {
            double d = Math.sqrt((h.x - px) * (h.x - px) + (h.y - py) * (h.y - py) + (h.z - pz) * (h.z - pz));
            if (Double.isNaN(best) || d < best) {
                best = d;
            }
        }
        return best;
    }

    static String esc(String s) {
        StringBuilder b = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') {
                b.append('\\');
            }
            if (c >= 0x20) {
                b.append(c);
            }
        }
        return b.toString();
    }

    /** Per-duck reply queues. Thread-safe (the WebSocket thread pushes, the 20 ms sender polls). */
    public static final class Speakers {
        private final Map<String, VoicePcm.Frames> queues = new LinkedHashMap<>();
        private final int prebuffer, maxFrames;
        private String current = "";

        public Speakers(int prebuffer, int maxFrames) {
            this.prebuffer = prebuffer;
            this.maxFrames = maxFrames;
        }

        /** Control message {"type":"speaker","duck":id}: following PCM belongs to that duck ("" = default duck). */
        public synchronized void speaker(String id) {
            current = id == null ? "" : id;
        }

        public synchronized String current() {
            return current;
        }

        public synchronized void push(short[] pcm) {
            queues.computeIfAbsent(current, k -> new VoicePcm.Frames(prebuffer, maxFrames)).push(pcm);
        }

        /** Barge-in: clear one duck's queue (null = all). */
        public synchronized void clear(String id) {
            if (id == null) {
                queues.values().forEach(VoicePcm.Frames::clear);
            } else {
                VoicePcm.Frames f = queues.get(id);
                if (f != null) {
                    f.clear();
                }
            }
        }

        public synchronized List<String> ids() {
            return new ArrayList<>(queues.keySet());
        }

        public synchronized short[] poll(String id) {
            VoicePcm.Frames f = queues.get(id);
            return f == null ? null : f.poll();
        }

        public synchronized int size(String id) {
            VoicePcm.Frames f = queues.get(id);
            return f == null ? 0 : f.size();
        }

        public synchronized void retain(java.util.Collection<String> keep) {
            queues.keySet().removeIf(k -> !k.isEmpty() && !keep.contains(k) && queues.get(k).size() == 0);
        }
    }
}
