package dev.yifan.microduck;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Standalone test harness for the mod's pure-Java parts (not loaded by Fabric; not in the jar).
 * Run: ./build.sh test [http://127.0.0.1:PORT]
 */
public final class Harness {
    static int pass, fail;

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            pass++;
        } else {
            fail++;
        }
        System.out.println((ok ? "PASS " : "FAIL ") + name + (detail == null ? "" : "  " + detail));
    }

    static boolean near(double[] a, double[] b, double eps) {
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > eps) {
                return false;
            }
        }
        return true;
    }

    static String v(double... d) {
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < d.length; i++) {
            sb.append(i > 0 ? ", " : "").append(String.format(Locale.ROOT, "%.4f", d[i]));
        }
        return sb.append(")").toString();
    }

    public static void main(String[] args) throws Exception {
        String url = args.length > 0 ? args[0] : "http://127.0.0.1:8796";
        math();
        commands();
        config();
        model();
        ModelTests.run();
        TerrainTests.run(url);
        http(url);
        MultiTests.parser();
        MultiTests.poses();
        MultiTests.voice();
        MultiTests.http(url);
        System.out.println();
        System.out.println("RESULT: " + pass + " passed, " + fail + " failed");
        System.exit(fail == 0 ? 0 : 1);
    }

    // ------------------------------------------------------------------ frame mapping

    static void math() {
        System.out.println("== frame mapping (A=(10.5,64,-3.5), s=4) ==");
        double ax = 10.5, ay = 64, az = -3.5, s = 4;
        double[] o = DuckMath.simToMc(ax, ay, az, s, 0, 0, 0);
        check("sim origin -> anchor", near(o, new double[] {10.5, 64, -3.5}, 1e-9), v(o));
        double[] p = DuckMath.simToMc(ax, ay, az, s, 0.1, 0.2, 0.05);
        check("sim (0.1,0.2,0.05) -> MC (10.9,64.2,-4.3)", near(p, new double[] {10.9, 64.2, -4.3}, 1e-9), v(p));
        double[] up = DuckMath.simToMc(ax, ay, az, s, 0, 0, 0.25);
        check("sim +z (up) 0.25 m -> MC +y 1 block", near(up, new double[] {10.5, 65, -3.5}, 1e-9), v(up));
        double[] sy = DuckMath.simToMc(ax, ay, az, s, 0, 1, 0);
        check("sim +y -> MC -z (north)", near(sy, new double[] {10.5, 64, -7.5}, 1e-9), v(sy));
        double[] back = DuckMath.mcToSim(ax, az, s, 10.9, -4.3);
        check("player MC (10.9,_,-4.3) -> sim (0.1,0.2)", near(back, new double[] {0.1, 0.2}, 1e-9), v(back));

        // 90 deg yaw about sim +z
        double c = Math.cos(Math.PI / 4), sn = Math.sin(Math.PI / 4);
        double[] qmc = DuckMath.quatSimToMc(c, 0, 0, sn);
        check("yaw90 quat sim [c,0,0,s] -> MC [c,0,s,0] (about +y)", near(qmc, new double[] {c, 0, sn, 0}, 1e-12), v(qmc));
        double[] fwd = DuckMath.rotate(qmc, 1, 0, 0);
        check("yaw90: sim forward +x ends up MC -z (sim +y)", near(fwd, new double[] {0, 0, -1}, 1e-12), v(fwd));
        float[] pose = {0.1f, 0.2f, 0.05f, (float) c, 0, 0, (float) sn};
        double[] vtx = DuckMath.bodyVertexToMc(ax, ay, az, s, pose, 0.1, 0, 0);
        // sim world = p + R v = (0.1, 0.3, 0.05) -> MC (10.9, 64.2, -4.7)
        check("yaw90 body vertex (0.1,0,0) @p -> MC (10.9,64.2,-4.7)",
                near(vtx, new double[] {10.9, 64.2, -4.7}, 1e-6), v(vtx));

        // arbitrary rotation: renderer composition == contract formula A + s*C*(p + R_sim v)
        double[] q = norm(new double[] {0.7, 0.2, -0.4, 0.55});
        float[] pq = {0.03f, -0.07f, 0.12f, (float) q[0], (float) q[1], (float) q[2], (float) q[3]};
        double[] lv = {0.02, -0.05, 0.08};
        double[] rv = DuckMath.rotate(q, lv[0], lv[1], lv[2]);
        double[] want = DuckMath.simToMc(ax, ay, az, s, pq[0] + rv[0], pq[1] + rv[1], pq[2] + rv[2]);
        double[] got = DuckMath.bodyVertexToMc(ax, ay, az, s, pq, lv[0], lv[1], lv[2]);
        check("general pose: render path == A + s*C*(p + R v)", near(got, want, 1e-5), v(got) + " vs " + v(want));
        // R_mc == C R C^T as matrices
        double[][] C = {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}};
        double[][] r = DuckMath.matrix(q);
        double[][] expect = mul(mul(C, r), transpose(C));
        double[][] rmc = DuckMath.matrix(DuckMath.quatSimToMc(q[0], q[1], q[2], q[3]));
        double err = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                err = Math.max(err, Math.abs(expect[i][j] - rmc[i][j]));
            }
        }
        check("R_mc == C R_sim C^T (max err " + String.format(Locale.ROOT, "%.1e", err) + ")", err < 1e-12, null);

        // interpolation: halfway between yaw 0 and yaw 90 is yaw 45, positions lerp
        float[] a = {0, 0, 0, 1, 0, 0, 0};
        float[] b = {1, 2, 3, (float) c, 0, 0, (float) sn};
        float[] m = DuckMath.lerpPose(a, b, 0.5f, null);
        double yaw = Math.toDegrees(2 * Math.atan2(m[6], m[3]));
        check("lerpPose t=0.5: pos (0.5,1,1.5), yaw 45", Math.abs(yaw - 45) < 1e-3 && near(
                new double[] {m[0], m[1], m[2]}, new double[] {0.5, 1, 1.5}, 1e-6), "yaw=" + String.format(Locale.ROOT, "%.4f", yaw));
        float[] bneg = {1, 2, 3, (float) -c, 0, 0, (float) -sn};      // same rotation, other hemisphere
        float[] m2 = DuckMath.lerpPose(a, bneg, 0.5f, null);
        double yaw2 = Math.toDegrees(2 * Math.atan2(m2[6] * Math.signum(m2[3]), Math.abs(m2[3])));
        check("lerpPose takes shortest arc (q vs -q)", Math.abs(yaw2 - 45) < 1e-3, "yaw=" + String.format(Locale.ROOT, "%.4f", yaw2));
    }

    static double[] norm(double[] q) {
        double n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        return new double[] {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
    }

    static double[][] mul(double[][] a, double[][] b) {
        double[][] r = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return r;
    }

    static double[][] transpose(double[][] a) {
        double[][] r = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i][j] = a[j][i];
            }
        }
        return r;
    }

    // ------------------------------------------------------------------ command parser

    static void commands() {
        System.out.println("== command parser ==");
        String[][] cases = {
            {"hello world", "NOT_DUCK", null},
            {"!duckling", "NOT_DUCK", null},
            {"!duck", "HELP", null},
            {"  !DUCK  ", "HELP", null},
            {"!duck help", "HELP", null},
            {"!duck walk", "SIDECAR", "{\"cmd\":\"walk\",\"from\":\"mc\"}"},
            {"!duck STOP", "SIDECAR", "{\"cmd\":\"stop\",\"from\":\"mc\"}"},
            {"!duck sit", "SIDECAR", "{\"cmd\":\"sit\",\"from\":\"mc\"}"},
            {"!duck stand", "SIDECAR", "{\"cmd\":\"stand\",\"from\":\"mc\"}"},
            {"!duck follow", "SIDECAR", "{\"cmd\":\"follow\",\"from\":\"mc\"}"},
            {"!duck quack", "SIDECAR", "{\"cmd\":\"quack\",\"from\":\"mc\"}"},
            {"!duck reset", "SIDECAR", "{\"cmd\":\"reset\",\"from\":\"mc\"}"},
            {"!duck left", "SIDECAR", "{\"cmd\":\"turn\",\"dir\":\"left\",\"from\":\"mc\"}"},
            {"!duck right", "SIDECAR", "{\"cmd\":\"turn\",\"dir\":\"right\",\"from\":\"mc\"}"},
            {"!duck here", "HERE", null},
            {"!duck scale 2.5", "SCALE", "2.5"},
            {"!duck scale 0.5", "SCALE", "0.5"},
            {"!duck scale 10", "SCALE", "10.0"},
            {"!duck scale 11", "ERROR", null},
            {"!duck scale 0.1", "ERROR", null},
            {"!duck scale abc", "ERROR", null},
            {"!duck scale", "ERROR", null},
            {"!duck scale NaN", "ERROR", null},
            {"!duck fly", "ERROR", null},
            {"!duck brains", "BRAINS", null},
            {"!duck brain sprint", "SIDECAR", "{\"cmd\":\"brain\",\"name\":\"sprint\",\"from\":\"mc\"}"},
            {"!duck Brain DEFAULT", "SIDECAR", "{\"cmd\":\"brain\",\"name\":\"default\",\"from\":\"mc\"}"},
            {"!duck trick roulade", "SIDECAR", "{\"cmd\":\"trick\",\"name\":\"roulade\",\"from\":\"mc\"}"},
            {"!duck trick kick-left", "SIDECAR", "{\"cmd\":\"trick\",\"name\":\"kick-left\",\"from\":\"mc\"}"},
            {"!duck brain", "ERROR", null},
            {"!duck trick", "ERROR", null},
            {"!duck brain a b", "ERROR", null},
            {"!duck brain ../x", "ERROR", null},
            {"!duck trick a\"b", "ERROR", null},
            {"!duck trick abcdefghijklmnopqrstuvwxy", "ERROR", null},
        };
        for (String[] t : cases) {
            DuckCommand c = DuckCommand.parse(t[0]);
            boolean ok = c.kind.name().equals(t[1]);
            String det = c.kind + (c.json != null ? " " + c.json : "") + (c.kind == DuckCommand.Kind.SCALE ? " " + c.scale : "")
                    + (c.message != null && c.kind == DuckCommand.Kind.ERROR ? " (" + c.message + ")" : "");
            if (t[2] != null) {
                ok &= c.kind == DuckCommand.Kind.SCALE ? Double.toString(c.scale).equals(t[2]) : t[2].equals(c.json);
            }
            check(String.format("%-18s", "'" + t[0] + "'"), ok, det);
        }
        // `!duck brains` formatter (GET /brains body -> chat lines)
        java.util.List<String> lines = BrainsClient.format("{\"brain\":\"sprint\",\"trick\":null,\"robot\":\"default\",\"brains\":["
                + "{\"id\":\"sprint\",\"kind\":\"perpetual\",\"variant\":\"default\",\"status\":\"pass\"},"
                + "{\"id\":\"roller\",\"kind\":\"perpetual\",\"variant\":\"rollers\",\"status\":\"pass\"},"
                + "{\"id\":\"dash\",\"kind\":\"episodic\",\"variant\":\"default\",\"status\":\"wobbly\"},"
                + "{\"id\":\"BAD ID!\",\"kind\":\"episodic\"},"
                + "{\"id\":\"roulade\",\"kind\":\"episodic\",\"variant\":\"default\",\"status\":\"pass\"}]}");
        check("brains list formatted", lines.size() == 3 && lines.get(0).startsWith("brain now: sprint")
                && lines.get(1).equals("brains: default, sprint, roller (skates)")
                && lines.get(2).equals("tricks: dash~, roulade"), lines.toString());
        check("brains bad json -> one friendly line", BrainsClient.format("{nope").size() == 1, null);
    }

    // ------------------------------------------------------------------ properties round trip

    static void config() throws Exception {
        System.out.println("== config round trip ==");
        Path dir = Files.createTempDirectory("microduck-test");
        Path f = dir.resolve("config/microduck.properties");
        DuckConfig d = DuckConfig.load(f);
        check("missing file -> defaults", d.enabled && d.port == 8796 && d.scale == 4.0 && !d.hasAnchor(), null);
        d.setAnchorBlock(-12, 63, 40, "minecraft:overworld");
        d.scale = 2.5;
        d.port = 8799;
        d.enabled = false;
        d.save(f);
        System.out.println("  file: " + Files.readString(f).replace("\n", " | "));
        DuckConfig r = DuckConfig.load(f);
        check("round trip", !r.enabled && r.port == 8799 && r.scale == 2.5 && r.hasAnchor()
                && r.ax == -11.5 && r.ay == 63 && r.az == 40.5 && "minecraft:overworld".equals(r.dim),
                "anchor=" + v(r.ax, r.ay, r.az) + " dim=" + r.dim + " scale=" + r.scale + " port=" + r.port + " enabled=" + r.enabled);
        Files.writeString(f, "scale=99\nport=oops\nanchor.x=1\n");
        DuckConfig bad = DuckConfig.load(f);
        check("bad values clamp/default, partial anchor ignored", bad.scale == 10.0 && bad.port == 8796 && !bad.hasAnchor(),
                "scale=" + bad.scale + " port=" + bad.port + " anchor=" + bad.hasAnchor());
        check("no .tmp left behind", !Files.exists(f.resolveSibling("microduck.properties.tmp")), null);
    }

    // ------------------------------------------------------------------ model parse

    static void model() {
        System.out.println("== model parse ==");
        String json = "{\"version\":1,\"bodies\":[{\"name\":\"trunk\",\"meshes\":[{\"rgba\":[1,0.5,0,1],"
                + "\"verts\":[0,0,0, 1,0,0, 0,1,0],\"tris\":[0,1,2]},{\"rgba\":[1,1,1,0],\"verts\":[0,0,0,1,0,0,0,1,0],\"tris\":[0,1,2]}]},"
                + "{\"name\":\"foot\",\"meshes\":[{\"rgba\":[0,0,1,1],\"verts\":[0,0,0],\"tris\":[0,0,5]}]}]}";
        DuckModel m = DuckModel.parse(json);
        DuckModel.Mesh mesh = m.bodies[0].meshes[0];
        check("bodies/order", m.bodies.length == 2 && m.index.get("foot") == 1, null);
        check("invisible (alpha 0) mesh skipped, bad indices dropped",
                m.bodies[0].meshes.length == 1 && m.bodies[1].meshes.length == 0 && m.triangles == 1, "tris=" + m.triangles);
        check("rgba -> ARGB", mesh.argb == 0xFFFF8000, String.format("0x%08X", mesh.argb));
        // sim verts (0,1,0) -> MC axes (0,0,-1); sim normal +z -> MC +y
        check("verts in MC axes", mesh.tri[6] == 0 && mesh.tri[7] == 0 && mesh.tri[8] == -1, null);
        check("flat normal sim +z -> MC +y", near(new double[] {mesh.nrm[0], mesh.nrm[1], mesh.nrm[2]}, new double[] {0, 1, 0}, 1e-6),
                v(mesh.nrm[0], mesh.nrm[1], mesh.nrm[2]));
    }

    // ------------------------------------------------------------------ HTTP against the live (or fake) sidecar

    static void http(String url) throws Exception {
        System.out.println("== HTTP client against " + url + " ==");
        int port = URI.create(url).getPort();
        List<String> logs = new ArrayList<>();
        SidecarClient cl = new SidecarClient(() -> port, () -> true, s -> {
            logs.add(s);
            System.out.println("  [log] " + s);
        });
        cl.start();
        long deadline = System.currentTimeMillis() + 8000;
        while (cl.frames() == null && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        SidecarClient.Frames f = cl.frames();
        check("model + first pose fetched", f != null, null);
        if (f == null) {
            cl.stop();
            return;
        }
        DuckModel m = f.model;
        StringBuilder names = new StringBuilder();
        for (DuckModel.Body b : m.bodies) {
            names.append(b.name).append(' ');
        }
        System.out.println("  model: " + m.bodies.length + " bodies, " + m.triangles + " tris, max local extent "
                + String.format(Locale.ROOT, "%.3f m", m.radius) + "; bodies: " + names.toString().trim());
        ModelTests.live(m);
        long seq0 = f.cur.seq;
        long t0 = System.nanoTime();
        int changes = 0;
        long last = seq0;
        while (System.nanoTime() - t0 < 2_000_000_000L) {
            SidecarClient.Frames g = cl.frames();
            if (g != null && g.cur.seq != last) {
                changes++;
                last = g.cur.seq;
            }
            Thread.sleep(5);
        }
        SidecarClient.Frames g = cl.frames();
        check("pose stream advances (" + changes + " new poses in 2 s ~ " + changes / 2 + " Hz)", changes >= 20, null);
        int present = 0;
        for (float[] b : g.cur.bodies) {
            present += b != null ? 1 : 0;
        }
        check("pose covers every model body", present == m.bodies.length, present + "/" + m.bodies.length);
        float[] tr = SidecarClient.trunk(g.cur, m);
        double[] mc = DuckMath.simToMc(10.5, 64, -3.5, 4, tr[0], tr[1], tr[2]);
        System.out.println("  latest: seq " + g.cur.seq + " mode " + g.cur.mode + " fallen " + g.cur.fallen + " quack "
                + g.cur.quack + "; trunk sim " + v(tr[0], tr[1], tr[2]) + " -> MC " + v(mc));
        float a = SidecarClient.alpha(g, System.nanoTime());
        check("interp alpha in [0,1] and prev != cur", a >= 0 && a <= 1 && g.prev.seq != g.cur.seq, "alpha=" + a);

        // player position is POSTed at ~5 Hz
        cl.setPlayer(new double[] {0.25, -0.5});
        Thread.sleep(300);
        cl.setPlayer(new double[] {0.3, -0.5});
        Thread.sleep(500);
        String lastJson = rawGet(url + "/_last");
        if (lastJson != null) {
            check("POST /player arrived", lastJson.contains("\"player\": {\"x\": 0.3") || lastJson.contains("\"x\":0.3"), lastJson);
        } else {
            String h = rawGet(url + "/health");
            check("POST /player arrived (real sidecar /health echoes player)", h != null && !h.contains("\"player\":null"), h);
        }

        // command round trip (quack is harmless on the real sim)
        CompletableFuture<String> rep = new CompletableFuture<>();
        cl.sendCommand(DuckCommand.parse("!duck quack").json, rep::complete);
        String r = rep.get(5, TimeUnit.SECONDS);
        check("POST /cmd quack -> reply", r.startsWith("ok"), r);
        long q0 = g.cur.quack;
        Thread.sleep(1500);
        System.out.println("  quack counter " + q0 + " -> " + cl.frames().cur.quack);

        // community brains (real sidecar only: the fake server has no /brains)
        if (rawGet(url + "/brains") != null) {
            List<String> bl = java.util.Collections.synchronizedList(new ArrayList<>());
            BrainsClient.fetch(url, bl::add);
            for (int i = 0; i < 50 && bl.size() < 3; i++) {
                Thread.sleep(100);
            }
            check("!duck brains -> 3 chat lines", bl.size() == 3 && bl.get(0).startsWith("brain now: ")
                    && bl.get(1).startsWith("brains: default"), bl.toString());
            CompletableFuture<String> rep3 = new CompletableFuture<>();
            cl.sendCommand(DuckCommand.parse("!duck trick zz-nope").json, rep3::complete);
            String r3 = rep3.get(5, TimeUnit.SECONDS);
            check("!duck trick <unknown> -> sidecar error reply", r3.startsWith("error") && r3.contains("unknown"), r3);
        }
        cl.stop();

        // sidecar down: no frames, exactly one log line, no exceptions
        System.out.println("== sidecar down (port 1) ==");
        List<String> dl = new ArrayList<>();
        SidecarClient dead = new SidecarClient(() -> 1, () -> true, dl::add);
        dead.start();
        Thread.sleep(4000);
        CompletableFuture<String> rep2 = new CompletableFuture<>();
        dead.sendCommand(DuckCommand.parse("!duck walk").json, rep2::complete);
        String r2 = rep2.get(5, TimeUnit.SECONDS);
        check("down: frames null", dead.frames() == null, null);
        check("down: exactly one log line in 4 s of retries", dl.size() == 1, dl.toString());
        check("down: command reply is friendly", r2.contains("not reachable"), r2);
        dead.stop();
    }

    static String rawGet(String url) {
        try {
            HttpResponse<String> r = HttpClient.newHttpClient().send(HttpRequest.newBuilder(URI.create(url)).build(),
                    HttpResponse.BodyHandlers.ofString());
            return r.statusCode() == 200 ? r.body() : null;
        } catch (Exception e) {
            return null;
        }
    }
}
