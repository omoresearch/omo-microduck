package dev.yifan.microduck;

import static dev.yifan.microduck.Harness.check;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** model.json v2 extras: per-vertex normals / colours / cull flag, flat fallback, triangle-pair quads. */
final class ModelTests {

    private ModelTests() {
    }

    static void run() {
        System.out.println("== model.json extras (normals, colors, cull) ==");
        // square in the sim x-y plane (two CCW triangles seen from +z) + a lone triangle; normals point sim +z
        String sq = "\"verts\":[0,0,0, 1,0,0, 1,1,0, 0,1,0, 2,0,0],\"tris\":[0,1,2, 0,2,3, 1,4,2]";
        String json = "{\"version\":1,\"bodies\":[{\"name\":\"b\",\"meshes\":["
                + "{\"rgba\":[1,0,0,1]," + sq + ",\"normals\":[0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,2],"
                + "\"colors\":[255,0,0, 0,255,0, 0,0,255, 10,20,30, 300,-5,128],\"cull\":true},"
                + "{\"rgba\":[0,1,0,1]," + sq + "},"
                + "{\"rgba\":[0,0,1,1]," + sq + ",\"normals\":[0,0,1],\"colors\":[1,2],\"cull\":true}]}]}";
        DuckModel m = DuckModel.parse(json);
        DuckModel.Mesh s = m.bodies[0].meshes[0], flat = m.bodies[0].meshes[1], bad = m.bodies[0].meshes[2];

        check("v2 mesh: smooth + cull flags parsed", s.smooth && s.cull, "smooth=" + s.smooth + " cull=" + s.cull);
        check("v1 mesh: flat fallback, double-sided", !flat.smooth && !flat.cull, null);
        check("malformed normals/colors (wrong length) -> ignored, flat, no cull", !bad.smooth && !bad.cull
                && bad.qc[0] == 0xFF0000FF, String.format("0x%08X", bad.qc[0]));
        check("body cull/no-cull passes", m.bodies[0].anyCull && m.bodies[0].anyNoCull, null);

        // per-vertex normal: sim +z -> MC +y, renormalised (0,0,2 -> unit)
        boolean unitUp = true;
        for (int i = 0; i < s.qn.length; i += 3) {
            unitUp &= Math.abs(s.qn[i]) < 1e-6 && Math.abs(s.qn[i + 1] - 1) < 1e-6 && Math.abs(s.qn[i + 2]) < 1e-6;
        }
        check("per-vertex normals in MC axes, unit length", unitUp, null);
        // colours: bytes -> 0xFFRRGGBB, clamped
        Set<Integer> cols = new HashSet<>();
        for (int c : s.qc) {
            cols.add(c);
        }
        check("vertex colours -> ARGB (clamped)", cols.contains(0xFFFF0000) && cols.contains(0xFF00FF00)
                && cols.contains(0xFF0000FF) && cols.contains(0xFF0A141E) && cols.contains(0xFFFF0080), cols.toString());
        check("flat fallback uses mesh rgba", flat.qc[0] == 0xFF00FF00 && flat.qc.length == 12, null);

        // pairing: 3 triangles -> 1 real quad + 1 degenerate; flat mesh: 3 degenerate quads
        check("smooth mesh: 3 tris -> 2 quads (one pair)", s.quads() == 2 && s.triangles() == 3, "quads=" + s.quads());
        check("flat mesh: one degenerate quad per triangle", flat.quads() == 3, "quads=" + flat.quads());
        // geometry check: quad split (0,1,2)(2,3,0) must reproduce exactly the original triangles with the same winding
        Set<String> orig = new HashSet<>(), got = new HashSet<>();
        for (int t = 0; t < s.triangles(); t++) {
            orig.add(canon(s.tri, 9 * t, 9 * t + 3, 9 * t + 6));
        }
        for (int q = 0; q < s.quads(); q++) {
            int o = 12 * q;
            String t1 = canon(s.qv, o, o + 3, o + 6), t2 = canon(s.qv, o + 6, o + 9, o);
            got.add(t1);
            if (!(s.qv[o + 6] == s.qv[o + 9] && s.qv[o + 7] == s.qv[o + 10] && s.qv[o + 8] == s.qv[o + 11])) {
                got.add(t2);                        // degenerate quad's 2nd triangle has zero area
            }
        }
        check("quads == original triangles (same winding)", orig.equals(got), "orig=" + orig + " got=" + got);
        // the old Mesh.tri / nrm (flat) data is still there for v2 meshes
        check("flat tri/nrm kept for v2 meshes", s.tri.length == 27 && Math.abs(s.nrm[1] - 1) < 1e-6, null);
        check("model totals: triangles and quads", m.triangles == 9 && m.quads == 2 + 3 + 3, "tris=" + m.triangles + " quads=" + m.quads);
    }

    /** Live sidecar model: report smooth/cull coverage and the per-frame vertex count. */
    static void live(DuckModel m) {
        int meshes = 0, smooth = 0, cull = 0;
        for (DuckModel.Body b : m.bodies) {
            for (DuckModel.Mesh x : b.meshes) {
                meshes++;
                smooth += x.smooth ? 1 : 0;
                cull += x.cull ? 1 : 0;
            }
        }
        System.out.println(String.format(Locale.ROOT, "  live model: %d meshes (%d smooth, %d culled), %d tris -> %d quads"
                + " (%d vertices/frame)", meshes, smooth, cull, m.triangles, m.quads, 4 * m.quads));
        check("live model: quads <= triangles, every triangle emitted", m.quads <= m.triangles && m.quads * 2 >= m.triangles,
                m.triangles + " tris, " + m.quads + " quads");
    }

    private static String canon(float[] a, int i, int j, int k) {
        String[] p = {pt(a, i), pt(a, j), pt(a, k)};
        int s = 0;                                  // rotate so the smallest vertex comes first (keeps winding)
        for (int r = 1; r < 3; r++) {
            if (p[r].compareTo(p[s]) < 0) {
                s = r;
            }
        }
        return p[s] + "|" + p[(s + 1) % 3] + "|" + p[(s + 2) % 3];
    }

    private static String pt(float[] a, int i) {
        return a[i] + "," + a[i + 1] + "," + a[i + 2];
    }
}
