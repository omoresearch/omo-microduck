package dev.yifan.microduck;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/** Terrain sampler/merger/sim-conversion tests on fake block grids (no Minecraft classes). Called from Harness. */
final class TerrainTests {
    private TerrainTests() {
    }

    static final double[][] BOTTOM_SLAB = {{0, 0, 0, 1, 0.5, 1}};
    static final double[][] CARPET = {{0, 0, 0, 1, 0.0625, 1}};
    // vanilla bottom stairs facing north: full bottom half + back (north, low z) top quarter
    static final double[][] STAIRS = {{0, 0, 0, 1, 0.5, 1}, {0, 0.5, 0, 1, 1, 0.5}};
    static final double[][] FENCE_POST = {{0.375, 0, 0.375, 0.625, 1.5, 0.625}};

    static void check(String name, boolean ok, String detail) {
        Harness.check("terrain: " + name, ok, detail);
    }

    static double volume(List<double[]> boxes) {
        double v = 0;
        for (double[] b : boxes) {
            v += (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]);
        }
        return v;
    }

    static double shapeVolume(TerrainGrid g) {
        double v = 0;
        for (int c : g.cells) {
            for (double[] b : g.shapes.get(c)) {
                v += (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]);
            }
        }
        return v;
    }

    static boolean overlap(double[] a, double[] b) {
        return a[0] < b[3] - 1e-9 && b[0] < a[3] - 1e-9 && a[1] < b[4] - 1e-9 && b[1] < a[4] - 1e-9
                && a[2] < b[5] - 1e-9 && b[2] < a[5] - 1e-9;
    }

    static int overlaps(List<double[]> boxes) {
        int n = 0;
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = i + 1; j < boxes.size(); j++) {
                if (overlap(boxes.get(i), boxes.get(j))) {
                    n++;
                }
            }
        }
        return n;
    }

    /** Lab-like room: solid ground y<64, 2-high walls on the border, a slab row, a carpet, stairs, a fence. */
    static TerrainGrid room() {
        TerrainGrid g = new TerrainGrid(-6, 60, -6, 13, 7, 13);   // x -6..6, y 60..66, z -6..6
        int slab = g.shapeId(BOTTOM_SLAB), carpet = g.shapeId(CARPET), stairs = g.shapeId(STAIRS);
        int fence = g.shapeId(FENCE_POST);
        for (int y = 0; y < g.ny; y++) {
            for (int z = 0; z < g.nz; z++) {
                for (int x = 0; x < g.nx; x++) {
                    int wy = g.oy + y;
                    boolean border = x == 0 || z == 0 || x == g.nx - 1 || z == g.nz - 1;
                    if (wy < 64 || (border && wy < 66)) {
                        g.set(x, y, z, 1);
                    }
                }
            }
        }
        for (int x = 2; x <= 5; x++) {
            g.set(x, 4, 2, slab);                      // slab row at y=64, z=-4
        }
        g.set(8, 4, 8, carpet);
        g.set(9, 4, 8, carpet);
        g.set(3, 4, 9, stairs);
        g.set(10, 4, 3, fence);
        return g;
    }

    static void run() {
        run(null);
    }

    static void run(String url) {
        // ---- merging
        TerrainGrid floor = new TerrainGrid(0, 0, 0, 5, 2, 5);
        java.util.Arrays.fill(floor.cells, 1);
        List<double[]> fb = floor.merge();
        check("solid 5x2x5 -> 1 box", fb.size() == 1 && near(fb.get(0), new double[] {0, 0, 0, 5, 2, 5}),
                fb.size() + " box(es)");

        TerrainGrid g = room();
        List<double[]> boxes = g.merge();
        double vol = volume(boxes), want = shapeVolume(g);
        int ov = overlaps(boxes);
        long cellsSolid = java.util.Arrays.stream(g.cells).filter(c -> c != 0).count();
        check("room: merged volume == block shape volume, no overlaps",
                Math.abs(vol - want) < 1e-9 && ov == 0, String.format(Locale.ROOT, "%d solid cells -> %d boxes, vol %.4f vs %.4f, overlaps %d",
                        cellsSolid, boxes.size(), vol, want, ov));
        check("room: compact (<= 16 boxes)", boxes.size() <= 16, boxes.size() + " boxes");
        boolean slabRow = false, carpet2 = false, fence = false;
        int stairBoxes = 0;
        for (double[] b : boxes) {
            slabRow |= near(b, new double[] {-4, 64, -4, 0, 64.5, -3});
            carpet2 |= near(b, new double[] {2, 64, 2, 4, 64.0625, 3});
            fence |= near(b, new double[] {4.375, 64, -2.625, 4.625, 65.5, -2.375});
            if (b[0] == -3 && b[3] == -2 && b[2] >= 3 && b[5] <= 4 && b[1] >= 64 && b[4] <= 65) {
                stairBoxes++;
            }
        }
        check("bottom slab row merges into one half-height box", slabRow, null);
        check("two carpets merge into one 1/16 box", carpet2, null);
        check("stairs emitted as their 2 collision boxes", stairBoxes == 2, stairBoxes + " boxes");
        check("fence post keeps its 1.5-block collision height", fence, null);

        // randomized: volume conservation + no overlaps on a mixed grid
        Random rnd = new Random(42);
        boolean allOk = true;
        String worst = "";
        for (int trial = 0; trial < 30; trial++) {
            TerrainGrid r = new TerrainGrid(rnd.nextInt(100) - 50, 40, rnd.nextInt(100) - 50, 9, 5, 7);
            int[] ids = {0, 1, 1, 1, r.shapeId(BOTTOM_SLAB), r.shapeId(CARPET), r.shapeId(STAIRS),
                r.shapeId(new double[][] {{0, 0, 0.8125, 1, 1, 1}})};   // thin wall panel (fences overlap the cell above)
            for (int i = 0; i < r.cells.length; i++) {
                r.cells[i] = ids[rnd.nextInt(ids.length)];
            }
            List<double[]> m = r.merge();
            double dv = Math.abs(volume(m) - shapeVolume(r));
            int o = overlaps(m);
            if (dv > 1e-9 || o > 0) {
                allOk = false;
                worst = "trial " + trial + " dv=" + dv + " overlaps=" + o;
            }
        }
        check("30 random mixed grids: volume conserved, no overlapping boxes", allOk, worst);

        // ---- hash / change detection
        TerrainGrid a = room(), b = room();
        long ha = a.hash();
        check("hash: same blocks -> same hash", ha == b.hash(), null);
        b.set(6, 4, 6, 1);                             // player places a block
        check("hash: one placed block -> different hash", ha != b.hash(), null);
        TerrainGrid c2 = new TerrainGrid(-6, 60, -6, 13, 7, 13);   // same content, shapes interned in another order
        int fence2 = c2.shapeId(FENCE_POST), stairs3 = c2.shapeId(STAIRS), carpet3 = c2.shapeId(CARPET);
        int slab3 = c2.shapeId(BOTTOM_SLAB);
        TerrainGrid ref = room();
        for (int i = 0; i < ref.cells.length; i++) {
            double[][] sh = ref.shapes.get(ref.cells[i]);
            c2.cells[i] = ref.cells[i] <= 1 ? ref.cells[i] : c2.shapeId(sh);
        }
        check("hash: independent of shape interning order", ha == c2.hash() && fence2 + stairs3 + carpet3 + slab3 > 0, null);

        // ---- incremental scan == full scan; unloaded chunk aborts
        TerrainGrid src = room();
        TerrainGrid dst = new TerrainGrid(src.ox, src.oy, src.oz, src.nx, src.ny, src.nz);
        for (int i = 2; i < src.shapes.size(); i++) {
            dst.shapeId(src.shapes.get(i));
        }
        TerrainScan sc = new TerrainScan(dst);
        int ticks = 0;
        TerrainScan.Source fake = (gg, x, y, z) -> src.get(x - src.ox, y - src.oy, z - src.oz);
        while (!sc.step(fake, 97, Long.MAX_VALUE)) {
            ticks++;
        }
        check("incremental scan (97 cells/tick) == source grid", java.util.Arrays.equals(dst.cells, src.cells) && !sc.aborted(),
                ticks + 1 + " ticks for " + src.cells.length + " cells");
        TerrainScan sc2 = new TerrainScan(new TerrainGrid(0, 0, 0, 20, 2, 20));
        boolean done = sc2.step((gg, x, y, z) -> x >= 16 ? -1 : 0, 100000, Long.MAX_VALUE);
        check("unloaded chunk -> pass aborted", done && sc2.aborted(), null);
        TerrainScan sc3 = new TerrainScan(new TerrainGrid(0, 0, 0, 64, 8, 64));
        sc3.step((gg, x, y, z) -> 0, Integer.MAX_VALUE, System.nanoTime() - 1);
        check("deadline stops a step early (256-cell granularity)", sc3.cellsThisPass == 256, sc3.cellsThisPass + " cells");

        // ---- region around the anchor
        TerrainGrid reg = TerrainScan.regionFor(10.5, 64, -3.5, 4);
        check("region at scale 4: leash 2.5 m = 10 blocks + 3 margin, 4 down",
                reg.ox == -3 && reg.nx == 27 && reg.oz == -17 && reg.nz == 27 && reg.oy == 60 && reg.ny >= 7,
                "origin " + reg.ox + "," + reg.oy + "," + reg.oz + " size " + reg.nx + "x" + reg.ny + "x" + reg.nz
                        + " = " + reg.cells.length + " cells");

        // ---- sim frame conversion (anchor = block centre x/z, block floor y, as DuckConfig.setAnchorBlock stores it)
        DuckConfig cfg = new DuckConfig();
        cfg.setAnchorBlock(10, 64, -4, "minecraft:overworld");
        double ax = cfg.ax, ay = cfg.ay, az = cfg.az, s = 4;
        double[] under = TerrainGrid.toSim(new double[] {10, 63, -4, 11, 64, -3}, ax, ay, az, s);
        check("block under the feet -> sim box centred on origin, top at z=0",
                near(under, new double[] {-0.125, -0.125, -0.25, 0.125, 0.125, 0}), Harness.v(under));
        double[] north = TerrainGrid.toSim(new double[] {10, 64, -5, 11, 65, -4}, ax, ay, az, s);
        check("block north (MC -z) -> sim +y", near(north, new double[] {-0.125, 0.125, 0, 0.125, 0.375, 0.25}),
                Harness.v(north));
        double[] east = TerrainGrid.toSim(new double[] {11, 64, -4, 12, 64.5, -3}, ax, ay, az, s);
        check("bottom slab east (MC +x) -> sim +x, 0.125 m tall", near(east, new double[] {0.125, -0.125, 0, 0.375, 0.125, 0.125}),
                Harness.v(east));
        // consistency with the renderer's mapping: the sim box corners map back onto the MC box corners
        double[] p0 = DuckMath.simToMc(ax, ay, az, s, east[0], east[1], east[2]);
        double[] p1 = DuckMath.simToMc(ax, ay, az, s, east[3], east[4], east[5]);
        check("sim box corners -> DuckMath.simToMc == MC box corners (y flip swaps z faces)",
                near(p0, new double[] {11, 64, -3}) && near(p1, new double[] {12, 64.5, -4}), Harness.v(p0) + " " + Harness.v(p1));
        double[] player = DuckMath.mcToSim(ax, az, s, 10.5, -4.5);
        check("player 1 block north of the anchor stands above the north block in sim",
                player[0] > north[0] && player[0] < north[3] && player[1] > north[1] && player[1] < north[4], Harness.v(player));

        // ---- JSON + cap
        TerrainGrid rr = room();
        List<double[]> mb = rr.merge();
        String json = TerrainGrid.toJson(mb, rr.regionMc(), 0.5, 64, 0.5, 4, "minecraft:overworld", Long.toHexString(rr.hash()));
        JsonObject o = JsonParser.parseString(json).getAsJsonObject();
        JsonArray arr = o.getAsJsonArray("boxes");
        boolean minmax = true;
        double topMax = -9;
        for (int i = 0; i < arr.size(); i++) {
            JsonArray bx = arr.get(i).getAsJsonArray();
            for (int k = 0; k < 3; k++) {
                minmax &= bx.get(k).getAsDouble() < bx.get(k + 3).getAsDouble();
            }
            if (bx.get(0).getAsDouble() < 0 && bx.get(3).getAsDouble() > 0 && bx.get(1).getAsDouble() < 0 && bx.get(4).getAsDouble() > 0) {
                topMax = Math.max(topMax, bx.get(5).getAsDouble());
            }
        }
        check("JSON parses; " + arr.size() + " boxes, every box min<max", arr.size() == mb.size() && minmax
                && o.getAsJsonArray("region").size() == 6 && o.get("scale").getAsDouble() == 4, json.length() + " bytes");
        check("JSON: ground under the origin tops out at sim z=0", Math.abs(topMax) < 1e-9, "top=" + topMax);
        List<double[]> many = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            many.add(new double[] {i, 0, 0, i + 1, 1, 1});
        }
        List<double[]> capped = TerrainGrid.cap(many, 10, 20.5, 0.5);
        double maxd = 0;
        for (double[] q : capped) {
            maxd = Math.max(maxd, Math.abs((q[0] + q[3]) / 2 - 20.5));
        }
        check("cap keeps the boxes nearest the anchor", capped.size() == 10 && maxd <= 5, "max |dx|=" + maxd);
        // speed: a realistic 27x10x27 region (merge runs on the sender thread, still keep it cheap)
        TerrainGrid big = TerrainScan.regionFor(0.5, 64, 0.5, 4);
        Random r2 = new Random(1);
        int slab = big.shapeId(BOTTOM_SLAB);
        for (int i = 0; i < big.cells.length; i++) {
            int y = i / (big.nx * big.nz);
            big.cells[i] = y < 4 ? 1 : r2.nextInt(10) == 0 ? 1 : r2.nextInt(30) == 0 ? slab : 0;
        }
        long t0 = System.nanoTime();
        long hh = 0;
        for (int k = 0; k < 20; k++) {
            hh ^= big.hash();
        }
        long t1 = System.nanoTime();
        List<double[]> bm = big.merge();
        long t2 = System.nanoTime();
        check("hash + merge of a " + big.cells.length + "-cell region are cheap",
                (t1 - t0) / 20 < 3_000_000 && (t2 - t1) < 50_000_000,
                String.format(Locale.ROOT, "hash %.2f ms, merge %.2f ms -> %d boxes (%x)", (t1 - t0) / 20e6, (t2 - t1) / 1e6, bm.size(), hh & 0xf));
        if (url != null) {
            http(url);
        }
    }

    /** End-to-end: the Java JSON is accepted by the real sidecar (POST then GET /terrain), then cleared again. */
    static void http(String url) {
        java.net.http.HttpClient hc = java.net.http.HttpClient.newHttpClient();
        try {
            TerrainGrid g = room();
            String id = "harness-" + Long.toHexString(g.hash());
            String json = TerrainGrid.toJson(g.merge(), g.regionMc(), 0.5, 64, 0.5, 4, "minecraft:overworld", id);
            var post = hc.send(java.net.http.HttpRequest.newBuilder(java.net.URI.create(url + "/terrain"))
                    .header("Content-Type", "application/json").POST(java.net.http.HttpRequest.BodyPublishers.ofString(json)).build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            String got = "";
            for (int i = 0; i < 40 && !got.contains("\"id\":\"" + id + "\""); i++) {
                Thread.sleep(100);
                got = hc.send(java.net.http.HttpRequest.newBuilder(java.net.URI.create(url + "/terrain")).GET().build(),
                        java.net.http.HttpResponse.BodyHandlers.ofString()).body();
            }
            check("sidecar accepts the mod's JSON (POST 200, GET /terrain has our id, 10 boxes)",
                    post.statusCode() == 200 && got.contains("\"id\":\"" + id + "\"") && got.contains("\"boxes\":10"), got);
            hc.send(java.net.http.HttpRequest.newBuilder(java.net.URI.create(url + "/terrain"))
                    .header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("{\"id\":\"flat\",\"boxes\":[]}")).build(),
                    java.net.http.HttpResponse.BodyHandlers.discarding());
        } catch (Exception e) {
            check("sidecar accepts the mod's JSON", false, "sidecar not reachable: " + e);
        }
    }

    static boolean near(double[] a, double[] b) {
        return Harness.near(a, b, 1e-9);
    }
}
