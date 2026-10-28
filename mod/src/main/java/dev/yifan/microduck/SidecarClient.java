package dev.yifan.microduck;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.IntSupplier;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Talks to the microduck-sim sidecar on 127.0.0.1 from one daemon thread: fetches /model.json (retrying with
 * backoff), polls /pose at ~25 Hz, POSTs /player at ~5 Hz. The render/tick threads only read the volatile
 * {@link #frames()} snapshot and never block on HTTP. Pure Java (java.net.http + Gson).
 */
public final class SidecarClient implements Runnable {

    /** One received /pose, with body transforms aligned to the model's body order. */
    public static final class Pose {
        public final long seq;
        public final double t;
        public final String mode;
        public final boolean fallen;
        public final long quack;
        /** [bodyIndex] -> {px,py,pz,qw,qx,qy,qz} in the sim frame, or null if absent. */
        public final float[][] bodies;
        public final long recvNanos;

        Pose(long seq, double t, String mode, boolean fallen, long quack, float[][] bodies, long recvNanos) {
            this.seq = seq;
            this.t = t;
            this.mode = mode;
            this.fallen = fallen;
            this.quack = quack;
            this.bodies = bodies;
            this.recvNanos = recvNanos;
        }
    }

    /** Last two poses (prev may equal cur), for interpolation. */
    public static final class Frames {
        public final DuckModel model;
        public final Pose prev;
        public final Pose cur;

        Frames(DuckModel model, Pose prev, Pose cur) {
            this.model = model;
            this.prev = prev;
            this.cur = cur;
        }
    }

    /** One duck from GET /poses (multi-duck sidecar), with its last two poses for interpolation. */
    public static final class Duck {
        public final String id;
        public final String name;
        /** Accent colour 0xRRGGBB (persona). */
        public final int color;
        public final float quackPitch;
        public final boolean talking;
        public final Pose prev;
        public final Pose cur;

        Duck(String id, String name, int color, float quackPitch, boolean talking, Pose prev, Pose cur) {
            this.id = id;
            this.name = name;
            this.color = color;
            this.quackPitch = quackPitch;
            this.talking = talking;
            this.prev = prev;
            this.cur = cur;
        }
    }

    public static final long POSE_PERIOD_MS = 40;      // 25 Hz
    public static final long PLAYER_PERIOD_MS = 200;   // 5 Hz

    private final IntSupplier port;
    private final BooleanSupplier enabled;
    private final Consumer<String> log;
    private final HttpClient http;
    private volatile boolean running = true;
    private volatile DuckModel model;
    private volatile Frames frames;
    private volatile double[] playerSim;             // latest {x,y} in sim metres, or null
    private volatile java.util.List<Duck> ducks = java.util.List.of();
    /** null = not probed yet; false = old single-duck sidecar (no /poses): poll /pose. */
    private volatile Boolean multi;
    private boolean down;
    private boolean everUp;

    public SidecarClient(IntSupplier port, BooleanSupplier enabled, Consumer<String> log) {
        this.port = port;
        this.enabled = enabled;
        this.log = log;
        this.http = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofMillis(500))
                .build();
    }

    public Thread start() {
        Thread t = new Thread(this, "microduck-sidecar");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        t.start();
        return t;
    }

    public void stop() {
        running = false;
    }

    /** Null while the sidecar is down or before the first pose. */
    public Frames frames() {
        return frames;
    }

    public DuckModel model() {
        return model;
    }

    /** Every duck (roster order, the first is the default duck); empty while down or before the first poll. */
    public java.util.List<Duck> ducks() {
        return ducks;
    }

    /** Whether the sidecar serves GET /poses (null until probed). */
    public Boolean multi() {
        return multi;
    }

    /** Duck by id (or null). */
    public Duck duck(String id) {
        for (Duck d : ducks) {
            if (d.id.equals(id)) {
                return d;
            }
        }
        return null;
    }

    /** Player position in sim metres (null = not in a world / anchor not set). Sent at ~5 Hz. */
    public void setPlayer(double[] simXY) {
        playerSim = simXY;
    }

    public String base() {
        return "http://127.0.0.1:" + port.getAsInt();
    }

    @Override
    public void run() {
        long backoff = 1000;
        long nextPlayer = 0;
        double[] lastSentPlayer = null;
        while (running) {
            try {
                if (!enabled.getAsBoolean()) {
                    frames = null;
                    Thread.sleep(500);
                    continue;
                }
                if (model == null) {
                    model = DuckModel.parse(get("/model.json", 5000));
                    markUp("model: " + model.bodies.length + " bodies, " + model.triangles + " triangles");
                    backoff = 1000;
                }
                long t0 = System.nanoTime();
                if (!Boolean.FALSE.equals(multi)) {
                    String body = getOrNull("/poses", 1000);
                    if (body == null) {
                        multi = false;                    // old sidecar: one duck from /pose
                        log.accept("sidecar has no /poses (single-duck sidecar): rendering the one duck from /pose");
                    } else {
                        multi = true;
                        java.util.List<Duck> ds = parsePoses(body, model, System.nanoTime(), ducks);
                        ducks = ds;
                        Duck d0 = ds.isEmpty() ? null : ds.get(0);
                        frames = d0 == null ? null : new Frames(model, d0.prev, d0.cur);
                    }
                }
                if (Boolean.FALSE.equals(multi)) {
                    Pose p = parsePose(get("/pose", 1000), model, System.nanoTime());
                    Frames f = frames;
                    if (f == null || f.model != model) {
                        frames = new Frames(model, p, p);
                    } else if (p.seq != f.cur.seq) {
                        frames = new Frames(model, f.cur, p);
                    }
                    Frames g = frames;
                    ducks = java.util.List.of(new Duck("", "Duck", DEFAULT_COLOR, 1f, false, g.prev, g.cur));
                }
                markUp(null);
                backoff = 1000;
                long now = System.currentTimeMillis();
                double[] pl = playerSim;
                if (pl != null && now >= nextPlayer && pl != lastSentPlayer) {
                    nextPlayer = now + PLAYER_PERIOD_MS;
                    lastSentPlayer = pl;
                    post("/player", "{\"x\":" + num(pl[0]) + ",\"y\":" + num(pl[1]) + "}", 1000);
                }
                long sleep = POSE_PERIOD_MS - (System.nanoTime() - t0) / 1_000_000L;
                if (sleep > 0) {
                    Thread.sleep(sleep);
                }
            } catch (InterruptedException ie) {
                return;
            } catch (Exception e) {
                frames = null;                        // nothing renders while the sidecar is down
                ducks = java.util.List.of();
                multi = null;                         // re-probe /poses when it comes back (may be a newer/older one)
                if (!down) {
                    down = true;
                    log.accept("sidecar " + base() + " unavailable (" + e.getClass().getSimpleName()
                            + (e.getMessage() != null ? ": " + e.getMessage() : "") + "); retrying quietly");
                    model = null;                     // it may come back with different geometry
                }
                try {
                    Thread.sleep(backoff);
                } catch (InterruptedException ie) {
                    return;
                }
                backoff = Math.min(backoff * 2, 15000);
            }
        }
    }

    private void markUp(String detail) {
        if (down || !everUp) {
            log.accept("sidecar " + base() + " connected" + (detail != null ? " (" + detail + ")" : ""));
            down = false;
            everUp = true;
        } else if (detail != null) {
            log.accept(detail);
        }
    }

    /** Fire-and-forget POST /cmd; reply gets a one-line human result (on an HttpClient thread). */
    public void sendCommand(String json, Consumer<String> reply) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + "/cmd"))
                .timeout(Duration.ofSeconds(2))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        http.sendAsync(req, HttpResponse.BodyHandlers.ofString()).whenComplete((r, err) -> {
            if (err != null) {
                reply.accept("sidecar not reachable at " + base());
                return;
            }
            try {
                JsonObject o = JsonParser.parseString(r.body()).getAsJsonObject();
                boolean ok = o.has("ok") && o.get("ok").getAsBoolean();
                if (ok) {
                    if (o.has("results") && o.get("results").isJsonArray()) {
                        int n = 0, good = 0;
                        for (JsonElement e : o.getAsJsonArray("results")) {
                            n++;
                            good += e.getAsJsonObject().has("ok") && e.getAsJsonObject().get("ok").getAsBoolean() ? 1 : 0;
                        }
                        reply.accept("ok for " + good + "/" + n + " ducks");
                        return;
                    }
                    reply.accept("ok" + (o.has("mode") ? " (mode: " + str(o.get("mode")) + ")" : ""));
                } else {
                    reply.accept("error: " + (o.has("error") ? str(o.get("error")) : "HTTP " + r.statusCode()));
                }
            } catch (Exception e) {
                reply.accept("bad reply (HTTP " + r.statusCode() + ")");
            }
        });
    }

    private static String str(JsonElement e) {
        return e == null || e.isJsonNull() ? "?" : e.isJsonPrimitive() ? e.getAsString() : e.toString();
    }

    String get(String path, long timeoutMs) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofMillis(timeoutMs)).GET().build();
        HttpResponse<String> r = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) {
            throw new java.io.IOException("HTTP " + r.statusCode() + " for " + path);
        }
        return r.body();
    }

    /** GET that returns null on HTTP 404 (endpoint missing on an older sidecar); other errors throw. */
    String getOrNull(String path, long timeoutMs) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofMillis(timeoutMs)).GET().build();
        HttpResponse<String> r = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() == 404) {
            return null;
        }
        if (r.statusCode() != 200) {
            throw new java.io.IOException("HTTP " + r.statusCode() + " for " + path);
        }
        return r.body();
    }

    /** POST returning the body (for spawn/remove); reply gets the parsed JSON or an error object (HttpClient thread). */
    public void request(String method, String path, String json, Consumer<JsonObject> reply) {
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(base() + path)).timeout(Duration.ofSeconds(8))
                .header("Content-Type", "application/json");
        b = "DELETE".equals(method) ? b.DELETE() : b.POST(HttpRequest.BodyPublishers.ofString(json == null ? "{}" : json));
        http.sendAsync(b.build(), HttpResponse.BodyHandlers.ofString()).whenComplete((r, err) -> {
            JsonObject o;
            if (err != null) {
                o = new JsonObject();
                o.addProperty("ok", false);
                o.addProperty("error", "sidecar not reachable at " + base());
            } else {
                try {
                    o = JsonParser.parseString(r.body()).getAsJsonObject();
                } catch (Exception e) {
                    o = new JsonObject();
                    o.addProperty("ok", false);
                    o.addProperty("error", "bad reply (HTTP " + r.statusCode() + ")");
                }
            }
            reply.accept(o);
        });
    }

    public static final int DEFAULT_COLOR = 0xF2C230;

    static int parseColor(String s, int def) {
        if (s == null || !s.matches("#[0-9A-Fa-f]{6}")) {
            return def;
        }
        return Integer.parseInt(s.substring(1), 16);
    }

    /** GET /poses -> ducks in roster order, interpolation pairs carried over from the previous poll. */
    public static java.util.List<Duck> parsePoses(String json, DuckModel model, long recvNanos, java.util.List<Duck> old) {
        JsonObject o = JsonParser.parseString(json).getAsJsonObject();
        JsonObject ds = o.has("ducks") && o.get("ducks").isJsonObject() ? o.getAsJsonObject("ducks") : new JsonObject();
        java.util.List<String> order = new java.util.ArrayList<>();
        if (o.has("order") && o.get("order").isJsonArray()) {
            for (JsonElement e : o.getAsJsonArray("order")) {
                order.add(e.getAsString());
            }
        } else {
            order.addAll(ds.keySet());
        }
        java.util.Map<String, Duck> prev = new java.util.HashMap<>();
        for (Duck d : old) {
            prev.put(d.id, d);
        }
        java.util.List<Duck> out = new java.util.ArrayList<>();
        for (String id : order) {
            if (!ds.has(id) || !ds.get(id).isJsonObject()) {
                continue;
            }
            JsonObject e = ds.getAsJsonObject(id);
            Pose p = parsePose(e, model, recvNanos);
            Duck q = prev.get(id);
            Pose pp, cc;
            if (q == null) {
                pp = p;
                cc = p;
            } else if (p.seq != q.cur.seq) {
                pp = q.cur;
                cc = p;
            } else {
                pp = q.prev;
                cc = q.cur;
            }
            String name = e.has("name") ? str(e.get("name")) : id;
            int color = parseColor(e.has("color") ? str(e.get("color")) : null, DEFAULT_COLOR);
            float pitch = e.has("quack_pitch") ? e.get("quack_pitch").getAsFloat() : 1f;
            boolean talking = e.has("talking") && e.get("talking").getAsBoolean();
            out.add(new Duck(id, name, color, pitch, talking, pp, cc));
        }
        return out;
    }

    void post(String path, String json, long timeoutMs) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofMillis(timeoutMs))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json)).build();
        http.send(req, HttpResponse.BodyHandlers.discarding());
    }

    static String num(double d) {
        return Double.isFinite(d) ? Double.toString(d) : "0";
    }

    public static Pose parsePose(String json, DuckModel model, long recvNanos) {
        return parsePose(JsonParser.parseString(json).getAsJsonObject(), model, recvNanos);
    }

    public static Pose parsePose(JsonObject o, DuckModel model, long recvNanos) {
        float[][] bodies = new float[model.bodies.length][];
        JsonObject b = o.has("bodies") ? o.getAsJsonObject("bodies") : new JsonObject();
        for (var e : b.entrySet()) {
            Integer i = model.index.get(e.getKey());
            if (i == null || !e.getValue().isJsonArray() || e.getValue().getAsJsonArray().size() < 7) {
                continue;
            }
            float[] v = new float[7];
            for (int k = 0; k < 7; k++) {
                v[k] = e.getValue().getAsJsonArray().get(k).getAsFloat();
            }
            float n = (float) Math.sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6]);
            if (!(n > 1e-6f) || !Float.isFinite(v[0] + v[1] + v[2])) {
                continue;
            }
            for (int k = 3; k < 7; k++) {
                v[k] /= n;
            }
            bodies[i] = v;
        }
        return new Pose(
                o.has("seq") ? o.get("seq").getAsLong() : 0,
                o.has("t") ? o.get("t").getAsDouble() : 0,
                o.has("mode") ? str(o.get("mode")) : "?",
                o.has("fallen") && o.get("fallen").getAsBoolean(),
                o.has("quack") ? o.get("quack").getAsLong() : 0,
                bodies, recvNanos);
    }

    /** The trunk's sim pose (or the first body present): where quacks come from. */
    public static float[] trunk(Pose p, DuckModel m) {
        Integer i = m.index.get("trunk");
        if (i != null && p.bodies[i] != null) {
            return p.bodies[i];
        }
        for (int k = 0; k < m.bodies.length; k++) {            // e.g. the real MJCF's "trunk_base"
            if (m.bodies[k].name.startsWith("trunk") && p.bodies[k] != null) {
                return p.bodies[k];
            }
        }
        for (float[] b : p.bodies) {
            if (b != null) {
                return b;
            }
        }
        return null;
    }

    /**
     * Interpolation factor between prev and cur: we display one poll interval behind the newest pose, so
     * alpha = (now - cur.recv) / (cur.recv - prev.recv), clamped to [0, 1].
     */
    public static float alpha(Frames f, long nowNanos) {
        return alpha(f.prev, f.cur, nowNanos);
    }

    public static float alpha(Pose prev, Pose cur, long nowNanos) {
        long span = cur.recvNanos - prev.recvNanos;
        if (span <= 0) {
            return 1f;
        }
        float a = (float) (nowNanos - cur.recvNanos) / (float) span;
        return a < 0 ? 0 : a > 1 ? 1 : a;
    }
}
