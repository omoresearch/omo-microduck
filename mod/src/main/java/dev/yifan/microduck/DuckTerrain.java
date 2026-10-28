package dev.yifan.microduck;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.shapes.VoxelShape;

/**
 * Mirrors the real blocks around the anchor into the sim: on the client thread it scans block collision shapes
 * (VoxelShape AABBs: slabs, stairs, carpets, fences are exact) in a box around the anchor, spread over ticks with a
 * ~0.7 ms budget; when a pass finds the geometry changed (content hash) it hands the finished grid to a daemon thread
 * that merges it into boxes, converts to sim metres and POSTs /terrain. Re-scans every 2 s; every 10 s it checks
 * GET /terrain still carries our id (sidecar restarted -> resend).
 */
public final class DuckTerrain {
    static final long PERIOD_NS = 2_000_000_000L;
    static final long VERIFY_NS = 10_000_000_000L;
    static final int CELLS_PER_TICK = 8192;
    static final long BUDGET_NS = 700_000L;
    static final int MAX_BOXES = 2000;

    private static TerrainScan scan;
    private static String scanKey;
    private static long nextPass;
    private static final IdentityHashMap<VoxelShape, Integer> IDS = new IdentityHashMap<>();
    private static final BlockPos.MutableBlockPos POS = new BlockPos.MutableBlockPos();

    private static final AtomicBoolean BUSY = new AtomicBoolean();
    private static volatile long lastSentHash;
    private static volatile String lastSentKey;
    private static volatile long nextVerify;
    private static volatile boolean failing;
    /** Last pass stats, for logs: cells, ticks, boxes. */
    static volatile String status = "idle";
    private static int passTicks;

    private static final ExecutorService SENDER = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "microduck-terrain");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        return t;
    });
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1).connectTimeout(Duration.ofMillis(500)).build();

    private DuckTerrain() {
    }

    public static void init() {
        ClientTickEvents.END_CLIENT_TICK.register(DuckTerrain::tick);
    }

    private static void tick(Minecraft mc) {
        try {
            tickInner(mc);
        } catch (RuntimeException e) {
            scan = null;
            nextPass = System.nanoTime() + 5 * PERIOD_NS;
            MicroDuck.LOG.warn("microduck: terrain scan failed (retrying in 10 s)", e);
        }
    }

    private static void tickInner(Minecraft mc) {
        DuckConfig c = MicroDuck.config;
        if (c == null || mc.level == null || mc.player == null || !c.enabled || !c.hasAnchor()
                || !MicroDuck.dimId(mc.level).equals(c.dim)) {
            scan = null;
            return;
        }
        String key = c.dim + "|" + c.ax + "|" + c.ay + "|" + c.az + "|" + c.scale + "|" + c.port;
        long now = System.nanoTime();
        if (scan != null && !key.equals(scanKey)) {
            scan = null;                               // anchor/scale changed mid-pass: start over
        }
        if (scan == null) {
            if (key.equals(scanKey) && now < nextPass) {
                return;
            }
            scan = new TerrainScan(TerrainScan.regionFor(c.ax, c.ay, c.az, c.scale));
            scanKey = key;
            IDS.clear();                               // ids are per grid (each grid interns its own shapes)
            passTicks = 0;
        }
        final ClientLevel level = mc.level;
        passTicks++;
        if (!scan.step((g, x, y, z) -> shapeAt(level, g, x, y, z), CELLS_PER_TICK, now + BUDGET_NS)) {
            return;
        }
        TerrainScan s = scan;
        scan = null;
        nextPass = now + PERIOD_NS;
        if (s.aborted()) {
            return;                                    // a chunk in the area isn't loaded yet: retry next period
        }
        long h = s.grid.hash();
        boolean changed = h != lastSentHash || !key.equals(lastSentKey);
        boolean verify = !changed && now >= nextVerify;
        if ((changed || verify) && BUSY.compareAndSet(false, true)) {
            TerrainGrid g = s.grid;                    // no longer touched by this thread: next pass gets a new grid
            double ax = c.ax, ay = c.ay, az = c.az, sc = c.scale;
            String dim = c.dim;
            int port = c.port, ticks = passTicks;
            boolean announce = !key.equals(lastSentKey);
            SENDER.execute(() -> send(g, h, key, ax, ay, az, sc, dim, port, !changed, announce, ticks));
        }
    }

    private static int shapeAt(ClientLevel level, TerrainGrid g, int x, int y, int z) {
        if (!level.hasChunk(SectionPos.blockToSectionCoord(x), SectionPos.blockToSectionCoord(z))) {
            return -1;
        }
        POS.set(x, y, z);
        BlockState st = level.getBlockState(POS);
        if (st.isAir()) {
            return 0;
        }
        VoxelShape vs = st.getCollisionShape(level, POS);
        if (vs.isEmpty()) {
            return 0;
        }
        Integer id = IDS.get(vs);
        if (id == null) {
            List<AABB> l = vs.toAabbs();
            double[][] b = new double[l.size()][];
            for (int i = 0; i < b.length; i++) {
                AABB a = l.get(i);
                b[i] = new double[] {a.minX, a.minY, a.minZ, a.maxX, a.maxY, a.maxZ};
            }
            id = g.shapeId(b);
            IDS.put(vs, id);
        }
        return id;
    }

    // ---------------------------------------------------------------- sender thread

    private static void send(TerrainGrid g, long h, String key, double ax, double ay, double az, double s, String dim,
                             int port, boolean verifyOnly, boolean announce, int ticks) {
        String base = "http://127.0.0.1:" + port;
        String id = Long.toHexString(h);
        try {
            if (verifyOnly) {
                HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create(base + "/terrain"))
                        .timeout(Duration.ofSeconds(2)).GET().build(), HttpResponse.BodyHandlers.ofString());
                if (r.statusCode() == 200 && r.body().contains("\"id\":\"" + id + "\"")) {
                    return;                            // sidecar still has our terrain
                }
            }
            long t0 = System.nanoTime();
            List<double[]> merged = g.merge();
            List<double[]> boxes = TerrainGrid.cap(merged, MAX_BOXES, ax, az);
            String json = TerrainGrid.toJson(boxes, g.regionMc(), ax, ay, az, s, dim, id);
            long t1 = System.nanoTime();
            HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(URI.create(base + "/terrain"))
                    .timeout(Duration.ofSeconds(5)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json)).build(), HttpResponse.BodyHandlers.ofString());
            if (r.statusCode() != 200) {
                throw new java.io.IOException("HTTP " + r.statusCode() + ": " + r.body());
            }
            lastSentHash = h;
            lastSentKey = key;
            failing = false;
            status = g.nx + "x" + g.ny + "x" + g.nz + " blocks -> " + boxes.size() + " boxes"
                    + (merged.size() > boxes.size() ? " (capped from " + merged.size() + ")" : "");
            MicroDuck.LOG.info("microduck: terrain {} ({} KB, scan {} ticks, merge {} ms)", status, json.length() / 1024,
                    ticks, (t1 - t0) / 1_000_000);
            if (announce) {
                MicroDuck.say("terrain: " + status);
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            if (!failing) {
                failing = true;                        // one line, then quiet retries every pass
                MicroDuck.LOG.info("microduck: terrain upload to {} failed ({}); retrying quietly", base, e.toString());
            }
            lastSentHash = 0;
        } finally {
            nextVerify = System.nanoTime() + VERIFY_NS;
            BUSY.set(false);
        }
    }
}
