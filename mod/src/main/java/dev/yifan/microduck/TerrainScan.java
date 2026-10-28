package dev.yifan.microduck;

/**
 * One incremental scan pass over a {@link TerrainGrid}: the client thread calls {@link #step} every tick with a
 * cell budget and a deadline, so a pass is spread over several ticks and never stalls a frame. Pure Java.
 */
public final class TerrainScan {
    /** Where the cells come from (the MC glue maps blocks to shape ids). */
    public interface Source {
        /** Shape id of the block at MC block coords, or -1 if that chunk isn't loaded (aborts the pass). */
        int shapeAt(TerrainGrid g, int x, int y, int z);
    }

    public static final double LEASH_M = 2.5;   // sidecar default leash (sim metres)
    public static final int MARGIN = 3;          // blocks beyond the leash circle
    public static final int MAX_RADIUS = 40;
    public static final int DOWN = 4;            // blocks below the anchor's feet block (pits / ledges)

    public final TerrainGrid grid;
    private int cursor;
    private boolean aborted;
    public int cellsThisPass;

    public TerrainScan(TerrainGrid grid) {
        this.grid = grid;
    }

    /** Grid around anchor A (block-centre x/z, floor y) at scale s: leash circle + margin, DOWN below, duck height up. */
    public static TerrainGrid regionFor(double ax, double ay, double az, double s) {
        int bx = (int) Math.floor(ax), by = (int) Math.floor(ay), bz = (int) Math.floor(az);
        int r = Math.min(MAX_RADIUS, (int) Math.ceil(LEASH_M * s) + MARGIN);
        int up = Math.max(3, (int) Math.ceil(0.35 * s) + 2);   // duck ~0.25 m tall -> 0.25*s blocks, plus headroom
        return new TerrainGrid(bx - r, by - DOWN, bz - r, 2 * r + 1, DOWN + up, 2 * r + 1);
    }

    public boolean done() {
        return aborted || cursor >= grid.cells.length;
    }

    public boolean aborted() {
        return aborted;
    }

    public double progress() {
        return (double) cursor / grid.cells.length;
    }

    /**
     * Scan up to maxCells (checking System.nanoTime every 256 cells against deadlineNanos). Returns done().
     */
    public boolean step(Source src, int maxCells, long deadlineNanos) {
        TerrainGrid g = grid;
        int end = Math.min(g.cells.length, cursor + maxCells);
        int nx = g.nx, nz = g.nz;
        while (cursor < end) {
            int i = cursor;
            int x = i % nx;
            int z = (i / nx) % nz;
            int y = i / (nx * nz);
            int id = src.shapeAt(g, g.ox + x, g.oy + y, g.oz + z);
            if (id < 0) {
                aborted = true;
                return true;
            }
            g.cells[i] = id;
            cursor++;
            cellsThisPass++;
            if ((cursor & 255) == 0 && System.nanoTime() > deadlineNanos) {
                break;
            }
        }
        return done();
    }
}
