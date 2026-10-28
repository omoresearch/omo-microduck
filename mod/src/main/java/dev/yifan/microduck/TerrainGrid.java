package dev.yifan.microduck;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Block-collision grid around the anchor and its conversion to sim-frame boxes for POST /terrain. Pure Java (no
 * Minecraft classes) so the harness can test it on fake grids.
 *
 * A cell holds a shape id: 0 = no collision, otherwise an index into {@link #shapes} (interned by content; id 1 is
 * always the full cube). A shape is a list of block-local AABBs {x0,y0,z0,x1,y1,z1} in [0,1].
 *
 * Merging is greedy over cells with the same single-box shape: a run extends along an axis only if the box spans
 * that axis fully (full cubes merge in x, z and y; bottom slabs/carpets merge in x and z; a wall-hugging thin
 * panel merges along its long axes). Multi-box shapes (stairs, fences, ...) are emitted box by box.
 */
public final class TerrainGrid {
    public static final double[] FULL = {0, 0, 0, 1, 1, 1};

    /** Origin block (min corner) and size in blocks. */
    public final int ox, oy, oz, nx, ny, nz;
    public final int[] cells;
    public final List<double[][]> shapes;
    private final Map<String, Integer> shapeIndex;

    public TerrainGrid(int ox, int oy, int oz, int nx, int ny, int nz) {
        this(ox, oy, oz, nx, ny, nz, new int[nx * ny * nz], new ArrayList<>(), new HashMap<>());
        shapes.add(new double[0][]);           // 0: empty
        shapes.add(new double[][] {FULL.clone()});
        shapeIndex.put(key(shapes.get(1)), 1);
    }

    private TerrainGrid(int ox, int oy, int oz, int nx, int ny, int nz, int[] cells, List<double[][]> shapes,
                        Map<String, Integer> shapeIndex) {
        this.ox = ox;
        this.oy = oy;
        this.oz = oz;
        this.nx = nx;
        this.ny = ny;
        this.nz = nz;
        this.cells = cells;
        this.shapes = shapes;
        this.shapeIndex = shapeIndex;
    }

    /** Independent copy of the cells (shapes are immutable and shared) for handing to another thread. */
    public TerrainGrid snapshot() {
        return new TerrainGrid(ox, oy, oz, nx, ny, nz, cells.clone(), new ArrayList<>(shapes), new HashMap<>(shapeIndex));
    }

    public int index(int x, int y, int z) {
        return (y * nz + z) * nx + x;
    }

    public void set(int x, int y, int z, int shape) {
        cells[index(x, y, z)] = shape;
    }

    public int get(int x, int y, int z) {
        return cells[index(x, y, z)];
    }

    /** Intern a shape (list of block-local boxes; empty -> 0). Not thread-safe: call from the scanning thread. */
    public int shapeId(double[][] boxes) {
        if (boxes == null || boxes.length == 0) {
            return 0;
        }
        String k = key(boxes);
        Integer id = shapeIndex.get(k);
        if (id == null) {
            id = shapes.size();
            shapes.add(boxes);
            shapeIndex.put(k, id);
        }
        return id;
    }

    static String key(double[][] boxes) {
        StringBuilder sb = new StringBuilder();
        for (double[] b : boxes) {
            for (double d : b) {
                sb.append(Math.round(d * 4096)).append(',');
            }
            sb.append(';');
        }
        return sb.toString();
    }

    /** Content hash of the grid (cell shapes + region): changes iff the collision geometry changes. */
    public long hash() {
        long h = 0xcbf29ce484222325L;
        long[] head = {ox, oy, oz, nx, ny, nz};
        for (long v : head) {
            h = (h ^ v) * 0x100000001b3L;
        }
        int[] sh = new int[shapes.size()];           // hash shape CONTENT: ids interned in another order still match
        for (int i = 1; i < sh.length; i++) {
            sh[i] = key(shapes.get(i)).hashCode() | 1;
        }
        for (int c : cells) {
            h = (h ^ sh[c]) * 0x100000001b3L;
        }
        return h;
    }

    // ------------------------------------------------------------------ merging (MC block coordinates)

    /** Merged boxes in MC world coordinates {X0,Y0,Z0,X1,Y1,Z1}. */
    public List<double[]> merge() {
        List<double[]> out = new ArrayList<>();
        boolean[] used = new boolean[cells.length];
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                for (int x = 0; x < nx; x++) {
                    int i = index(x, y, z);
                    int id = cells[i];
                    if (id == 0 || used[i]) {
                        continue;
                    }
                    double[][] sh = shapes.get(id);
                    if (sh.length != 1) {
                        used[i] = true;
                        for (double[] b : sh) {
                            out.add(new double[] {ox + x + b[0], oy + y + b[1], oz + z + b[2],
                                ox + x + b[3], oy + y + b[4], oz + z + b[5]});
                        }
                        continue;
                    }
                    double[] b = sh[0];
                    boolean sx = b[0] <= 0 && b[3] >= 1, sy = b[1] <= 0 && b[4] >= 1, sz = b[2] <= 0 && b[5] >= 1;
                    int ex = 1, ez = 1, ey = 1;
                    while (sx && x + ex < nx && free(used, x + ex, y, z, id)) {
                        ex++;
                    }
                    while (sz && z + ez < nz && rowFree(used, x, ex, y, z + ez, id)) {
                        ez++;
                    }
                    while (sy && y + ey < ny && layerFree(used, x, ex, y + ey, z, ez, id)) {
                        ey++;
                    }
                    for (int yy = y; yy < y + ey; yy++) {
                        for (int zz = z; zz < z + ez; zz++) {
                            for (int xx = x; xx < x + ex; xx++) {
                                used[index(xx, yy, zz)] = true;
                            }
                        }
                    }
                    out.add(new double[] {ox + x + b[0], oy + y + b[1], oz + z + b[2],
                        ox + x + ex - 1 + b[3], oy + y + ey - 1 + b[4], oz + z + ez - 1 + b[5]});
                }
            }
        }
        return out;
    }

    private boolean free(boolean[] used, int x, int y, int z, int id) {
        int i = index(x, y, z);
        return !used[i] && cells[i] == id;
    }

    private boolean rowFree(boolean[] used, int x, int ex, int y, int z, int id) {
        for (int xx = x; xx < x + ex; xx++) {
            if (!free(used, xx, y, z, id)) {
                return false;
            }
        }
        return true;
    }

    private boolean layerFree(boolean[] used, int x, int ex, int y, int z, int ez, int id) {
        for (int zz = z; zz < z + ez; zz++) {
            if (!rowFree(used, x, ex, y, zz, id)) {
                return false;
            }
        }
        return true;
    }

    /** Keep at most `cap` boxes, preferring those horizontally nearest the anchor (MC ax, az). */
    public static List<double[]> cap(List<double[]> boxes, int cap, double ax, double az) {
        if (boxes.size() <= cap) {
            return boxes;
        }
        double[][] arr = boxes.toArray(new double[0][]);
        Arrays.sort(arr, (p, q) -> Double.compare(dist2(p, ax, az), dist2(q, ax, az)));
        return new ArrayList<>(Arrays.asList(arr).subList(0, cap));
    }

    /** Squared horizontal distance from (ax, az) to the nearest point of the box. */
    static double dist2(double[] b, double ax, double az) {
        double dx = Math.max(0, Math.max(b[0] - ax, ax - b[3]));
        double dz = Math.max(0, Math.max(b[2] - az, az - b[5]));
        return dx * dx + dz * dz;
    }

    // ------------------------------------------------------------------ sim frame

    /**
     * MC world box -> sim box {x0,y0,z0,x1,y1,z1} (metres, min/max) for anchor A and scale s:
     * x = (X-ax)/s, y = -(Z-az)/s (so the MC max-Z face becomes the sim min-y face), z = (Y-ay)/s.
     */
    public static double[] toSim(double[] mc, double ax, double ay, double az, double s) {
        return new double[] {
            (mc[0] - ax) / s, -(mc[5] - az) / s, (mc[1] - ay) / s,
            (mc[3] - ax) / s, -(mc[2] - az) / s, (mc[4] - ay) / s};
    }

    /** Region (whole grid) in MC world coordinates. */
    public double[] regionMc() {
        return new double[] {ox, oy, oz, ox + nx, oy + ny, oz + nz};
    }

    /** JSON body for POST /terrain (boxes in sim metres, 4 decimals). */
    public static String toJson(List<double[]> mcBoxes, double[] regionMc, double ax, double ay, double az, double s,
                                String dim, String id) {
        StringBuilder sb = new StringBuilder(64 + mcBoxes.size() * 48);
        sb.append("{\"id\":\"").append(id).append("\",\"dim\":\"").append(dim == null ? "" : dim.replace("\"", ""))
                .append("\",\"scale\":").append(n(s))
                .append(",\"anchor\":[").append(n(ax)).append(',').append(n(ay)).append(',').append(n(az)).append(']');
        if (regionMc != null) {
            sb.append(",\"region\":");
            arr(sb, toSim(regionMc, ax, ay, az, s));
        }
        sb.append(",\"boxes\":[");
        for (int i = 0; i < mcBoxes.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            arr(sb, toSim(mcBoxes.get(i), ax, ay, az, s));
        }
        return sb.append("]}").toString();
    }

    private static void arr(StringBuilder sb, double[] v) {
        sb.append('[');
        for (int k = 0; k < v.length; k++) {
            if (k > 0) {
                sb.append(',');
            }
            sb.append(n(v[k]));
        }
        sb.append(']');
    }

    static String n(double d) {
        if (!Double.isFinite(d)) {
            return "0";
        }
        String s = String.format(Locale.ROOT, "%.4f", d);
        s = s.indexOf('.') >= 0 ? s.replaceAll("0+$", "").replaceAll("\\.$", "") : s;
        return s.equals("-0") ? "0" : s;
    }
}
