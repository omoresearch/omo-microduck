package dev.yifan.microduck;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Geometry from GET /model.json, pre-baked for rendering. Pure Java (Gson only).
 *
 * <p>Version-1 fields per mesh: {@code rgba, verts, tris}. Optional extras (sidecar model_export v2):
 * {@code normals} (per-vertex, body-local sim axes, crease-split smooth normals), {@code colors} (per-vertex sRGB
 * bytes, baked ambient occlusion), {@code cull} (closed, outward-wound mesh: safe for back-face culling).
 * Without them a mesh renders exactly as before: flat per-triangle normals, mesh rgba, double-sided.
 *
 * <p>Render data is packed as entity QUADS: with smooth normals, two triangles sharing an edge become one real
 * quad ordered so that the quad index pattern (0,1,2)(2,3,0) reproduces the original two triangles exactly;
 * leftovers (and every triangle of a flat-shaded mesh) go out as degenerate quads (last vertex repeated).
 */
public final class DuckModel {

    public static final class Mesh {
        /** 0xAARRGGBB base colour. */
        public final int argb;
        /** 9 floats per triangle: MC-axis body-local vertices. */
        public final float[] tri;
        /** 3 floats per triangle: flat unit normal in MC-axis body-local frame. */
        public final float[] nrm;
        /** Render quads: 12 floats (4 vertices) per quad, MC-axis body-local. */
        public final float[] qv;
        /** 12 floats per quad: per-vertex unit normals (MC axes). */
        public final float[] qn;
        /** 4 ints per quad: per-vertex colour 0xFFRRGGBB. */
        public final int[] qc;
        /** Model said the mesh is closed: render with back-face culling. */
        public final boolean cull;
        /** Per-vertex normals came from the model (else flat fallback). */
        public final boolean smooth;
        /** Optional model_export mesh name (e.g. "noenoeil" = the eye ring), null if absent. */
        public String name;
        /** Accent mesh: tinted with each duck's persona colour (multi-duck). */
        public boolean accent;
        private int tintKey = 0;
        private int[] tintCache;

        Mesh(int argb, float[] tri, float[] nrm, float[] qv, float[] qn, int[] qc, boolean cull, boolean smooth) {
            this.argb = argb;
            this.tri = tri;
            this.nrm = nrm;
            this.qv = qv;
            this.qn = qn;
            this.qc = qc;
            this.cull = cull;
            this.smooth = smooth;
        }

        public int triangles() {
            return nrm.length / 3;
        }

        public int quads() {
            return qc.length / 4;
        }

        /**
         * Per-vertex colours recoloured to {@code rgb} (0xRRGGBB), keeping the baked AO/shading: each vertex keeps its
         * brightness relative to the mesh's base colour. Cached for the last colour asked for (one per render pass,
         * ducks are drawn one after another so the cache flips at most N times per frame; cheap: ~1k vertices).
         */
        public int[] tinted(int rgb) {
            int key = rgb | 0x01000000;
            int[] c = tintCache;
            if (c != null && tintKey == key) {
                return c;
            }
            float br = ((argb >> 16) & 255) + ((argb >> 8) & 255) + (argb & 255);
            int tr = (rgb >> 16) & 255, tg = (rgb >> 8) & 255, tb = rgb & 255;
            c = new int[qc.length];
            for (int k = 0; k < qc.length; k++) {
                int v = qc[k];
                float f = br <= 0 ? 1f : (((v >> 16) & 255) + ((v >> 8) & 255) + (v & 255)) / br;
                f = Math.max(0f, Math.min(1.25f, f));
                c[k] = 0xFF000000 | (clamp255(Math.round(tr * f)) << 16) | (clamp255(Math.round(tg * f)) << 8)
                        | clamp255(Math.round(tb * f));
            }
            tintCache = c;
            tintKey = key;
            return c;
        }
    }

    /** Meshes that carry the per-duck accent colour (the amber eye ring around the lens). */
    public static final java.util.Set<String> ACCENT_MESHES = java.util.Set.of("noenoeil");

    public static final class Body {
        public final String name;
        public final Mesh[] meshes;
        public final boolean anyCull;
        public final boolean anyNoCull;

        Body(String name, Mesh[] meshes) {
            this.name = name;
            this.meshes = meshes;
            boolean c = false, n = false;
            for (Mesh m : meshes) {
                c |= m.cull;
                n |= !m.cull;
            }
            this.anyCull = c;
            this.anyNoCull = n;
        }
    }

    public final Body[] bodies;
    public final Map<String, Integer> index = new HashMap<>();
    public final int triangles;
    /** Quads actually emitted per frame (after pairing). */
    public final int quads;
    /** Largest distance (metres) of any vertex from its own body origin; pads the culling box. */
    public final float radius;

    private DuckModel(Body[] bodies, int triangles, int quads, float radius) {
        this.bodies = bodies;
        this.triangles = triangles;
        this.quads = quads;
        this.radius = radius;
        for (int i = 0; i < bodies.length; i++) {
            index.put(bodies[i].name, i);
        }
    }

    public static DuckModel parse(String json) {
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        JsonArray bs = root.getAsJsonArray("bodies");
        List<Body> out = new ArrayList<>();
        int total = 0, totalQuads = 0;
        float maxR = 0;
        for (JsonElement be : bs) {
            JsonObject b = be.getAsJsonObject();
            String name = b.get("name").getAsString();
            List<Mesh> meshes = new ArrayList<>();
            JsonArray ms = b.has("meshes") ? b.getAsJsonArray("meshes") : new JsonArray();
            for (JsonElement me : ms) {
                Mesh mesh = parseMesh(me.getAsJsonObject());
                if (mesh == null) {
                    continue;
                }
                JsonObject mo = me.getAsJsonObject();
                mesh.name = mo.has("name") && mo.get("name").isJsonPrimitive() ? mo.get("name").getAsString() : null;
                mesh.accent = mesh.name != null && ACCENT_MESHES.contains(mesh.name);
                float[] t = mesh.tri;
                for (int i = 0; i < t.length; i += 3) {
                    maxR = Math.max(maxR, (float) Math.sqrt(t[i] * t[i] + t[i + 1] * t[i + 1] + t[i + 2] * t[i + 2]));
                }
                total += mesh.triangles();
                totalQuads += mesh.quads();
                meshes.add(mesh);
            }
            out.add(new Body(name, meshes.toArray(new Mesh[0])));
        }
        return new DuckModel(out.toArray(new Body[0]), total, totalQuads, maxR);
    }

    static Mesh parseMesh(JsonObject m) {
        int argb = argb(m.has("rgba") ? m.getAsJsonArray("rgba") : null);
        if ((argb >>> 24) < 26) {
            return null;                           // ~invisible geoms (alpha < 0.1) would be cut out anyway
        }
        float[] v = floats(m.getAsJsonArray("verts"));
        int[] idx = ints(m.getAsJsonArray("tris"));
        int nv = v.length / 3;
        // optional per-vertex extras; ignored unless they cover every vertex exactly
        float[] vn = m.has("normals") && m.get("normals").isJsonArray() ? floats(m.getAsJsonArray("normals")) : null;
        if (vn != null && vn.length != nv * 3) {
            vn = null;
        }
        int[] vc = m.has("colors") && m.get("colors").isJsonArray() ? ints(m.getAsJsonArray("colors")) : null;
        if (vc != null && vc.length != nv * 3) {
            vc = null;
        }
        boolean cull = vn != null && m.has("cull") && m.get("cull").isJsonPrimitive() && m.get("cull").getAsBoolean();

        // valid triangles only
        int nt = idx.length / 3;
        int[] good = new int[nt * 3];
        int kept = 0;
        for (int t = 0; t < nt; t++) {
            int a = idx[3 * t], bb = idx[3 * t + 1], c = idx[3 * t + 2];
            if (a < 0 || bb < 0 || c < 0 || a >= nv || bb >= nv || c >= nv) {
                continue;
            }
            good[3 * kept] = a;
            good[3 * kept + 1] = bb;
            good[3 * kept + 2] = c;
            kept++;
        }
        if (kept == 0) {
            return null;
        }
        // MC-axis copies: C (x,y,z) = (x, z, -y)
        float[] P = new float[nv * 3];
        for (int i = 0; i < nv; i++) {
            P[3 * i] = v[3 * i];
            P[3 * i + 1] = v[3 * i + 2];
            P[3 * i + 2] = -v[3 * i + 1];
        }
        float[] N = null;
        if (vn != null) {
            N = new float[nv * 3];
            for (int i = 0; i < nv; i++) {
                float x = vn[3 * i], y = vn[3 * i + 2], z = -vn[3 * i + 1];
                float len = (float) Math.sqrt(x * x + y * y + z * z);
                if (!(len > 1e-6f)) {               // zero or NaN
                    x = 0;
                    y = 1;
                    z = 0;
                    len = 1;
                }
                N[3 * i] = x / len;
                N[3 * i + 1] = y / len;
                N[3 * i + 2] = z / len;
            }
        }
        int base = argb | 0xFF000000;
        int[] C = new int[nv];
        for (int i = 0; i < nv; i++) {
            C[i] = vc == null ? base
                    : 0xFF000000 | (clamp255(vc[3 * i]) << 16) | (clamp255(vc[3 * i + 1]) << 8) | clamp255(vc[3 * i + 2]);
        }

        float[] tri = new float[kept * 9];
        float[] nrm = new float[kept * 3];
        for (int t = 0; t < kept; t++) {
            for (int c = 0; c < 3; c++) {
                int k = good[3 * t + c];
                tri[9 * t + 3 * c] = P[3 * k];
                tri[9 * t + 3 * c + 1] = P[3 * k + 1];
                tri[9 * t + 3 * c + 2] = P[3 * k + 2];
            }
            int o = 9 * t;
            float e1x = tri[o + 3] - tri[o], e1y = tri[o + 4] - tri[o + 1], e1z = tri[o + 5] - tri[o + 2];
            float e2x = tri[o + 6] - tri[o], e2y = tri[o + 7] - tri[o + 1], e2z = tri[o + 8] - tri[o + 2];
            float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
            float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-12f) {
                nx = 0;
                ny = 1;
                nz = 0;
            } else {
                nx /= len;
                ny /= len;
                nz /= len;
            }
            nrm[3 * t] = nx;
            nrm[3 * t + 1] = ny;
            nrm[3 * t + 2] = nz;
        }

        // 4 vertex ids per quad; the flat fallback keeps one degenerate quad per triangle (quad q == triangle q)
        int[] quadIds = N != null ? pairTriangles(good, kept) : degenerate(good, kept);
        int nq = quadIds.length / 4;
        float[] qv = new float[nq * 12];
        float[] qn = new float[nq * 12];
        int[] qc = new int[nq * 4];
        for (int q = 0; q < nq; q++) {
            for (int c = 0; c < 4; c++) {
                int k = quadIds[4 * q + c];
                qv[12 * q + 3 * c] = P[3 * k];
                qv[12 * q + 3 * c + 1] = P[3 * k + 1];
                qv[12 * q + 3 * c + 2] = P[3 * k + 2];
                qc[4 * q + c] = C[k];
                if (N != null) {
                    qn[12 * q + 3 * c] = N[3 * k];
                    qn[12 * q + 3 * c + 1] = N[3 * k + 1];
                    qn[12 * q + 3 * c + 2] = N[3 * k + 2];
                } else {
                    qn[12 * q + 3 * c] = nrm[3 * q];
                    qn[12 * q + 3 * c + 1] = nrm[3 * q + 1];
                    qn[12 * q + 3 * c + 2] = nrm[3 * q + 2];
                }
            }
        }
        return new Mesh(argb, tri, nrm, qv, qn, qc, cull, N != null);
    }

    /** One degenerate quad per triangle (a, b, c, c). */
    static int[] degenerate(int[] f, int nt) {
        int[] q = new int[nt * 4];
        for (int t = 0; t < nt; t++) {
            q[4 * t] = f[3 * t];
            q[4 * t + 1] = f[3 * t + 1];
            q[4 * t + 2] = f[3 * t + 2];
            q[4 * t + 3] = f[3 * t + 2];
        }
        return q;
    }

    /**
     * Greedy pairing of triangles that share an edge with opposite direction (consistent winding). Triangle
     * (p,q,r) and its neighbour (q,p,s) across edge p-q become quad [p, s, q, r]: the quad split (0,1,2)(2,3,0)
     * gives (p,s,q) and (q,r,p), cyclic rotations of the two originals, so geometry, winding and per-vertex
     * attributes are unchanged. Unpaired triangles become degenerate quads.
     */
    static int[] pairTriangles(int[] f, int nt) {
        Map<Long, Integer> edge = new HashMap<>(nt * 4);
        for (int t = 0; t < nt; t++) {
            for (int e = 0; e < 3; e++) {
                edge.putIfAbsent(key(f[3 * t + e], f[3 * t + (e + 1) % 3]), t);
            }
        }
        boolean[] used = new boolean[nt];
        int[] out = new int[nt * 4];
        int n = 0;
        for (int t = 0; t < nt; t++) {
            if (used[t]) {
                continue;
            }
            used[t] = true;
            boolean paired = false;
            for (int e = 0; e < 3 && !paired; e++) {
                int p = f[3 * t + e], q = f[3 * t + (e + 1) % 3], r = f[3 * t + (e + 2) % 3];
                Integer u = edge.get(key(q, p));
                if (u == null || used[u]) {
                    continue;
                }
                int s = -1;
                for (int k = 0; k < 3; k++) {
                    if (f[3 * u + k] == q && f[3 * u + (k + 1) % 3] == p) {
                        s = f[3 * u + (k + 2) % 3];
                    }
                }
                if (s < 0 || s == r || s == p || s == q) {
                    continue;
                }
                used[u] = true;
                out[n++] = p;
                out[n++] = s;
                out[n++] = q;
                out[n++] = r;
                paired = true;
            }
            if (!paired) {
                out[n++] = f[3 * t];
                out[n++] = f[3 * t + 1];
                out[n++] = f[3 * t + 2];
                out[n++] = f[3 * t + 2];
            }
        }
        return java.util.Arrays.copyOf(out, n);
    }

    private static long key(int a, int b) {
        return ((long) a << 32) | (b & 0xffffffffL);
    }

    private static int clamp255(int x) {
        return Math.max(0, Math.min(255, x));
    }

    static int argb(JsonArray rgba) {
        float r = 0.8f, g = 0.8f, b = 0.8f, a = 1f;
        if (rgba != null && rgba.size() >= 3) {
            r = rgba.get(0).getAsFloat();
            g = rgba.get(1).getAsFloat();
            b = rgba.get(2).getAsFloat();
            a = rgba.size() >= 4 ? rgba.get(3).getAsFloat() : 1f;
        }
        return (c(a) << 24) | (c(r) << 16) | (c(g) << 8) | c(b);
    }

    private static int c(float f) {
        return Math.max(0, Math.min(255, Math.round(f * 255f)));
    }

    private static float[] floats(JsonArray a) {
        float[] f = new float[a == null ? 0 : a.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = a.get(i).getAsFloat();
        }
        return f;
    }

    private static int[] ints(JsonArray a) {
        int[] f = new int[a == null ? 0 : a.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = a.get(i).getAsInt();
        }
        return f;
    }
}
