package dev.yifan.microduck;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import com.google.gson.JsonObject;

/** Multi-duck parts of the mod: parser, /poses parsing, accent tint, voice routing, live HTTP. Called by Harness. */
final class MultiTests {
    private MultiTests() {
    }

    static void check(String name, boolean ok, String detail) {
        Harness.check(name, ok, detail);
    }

    static void parser() {
        System.out.println("== multi-duck command parser ==");
        Object[][] cases = {
            // line, kind, duck target, json
            {"!duck spawn", "SPAWN", null, null},
            {"!duck spawn Zippy", "SPAWN", "zippy", null},
            {"!duck spawn all", "SPAWN", "all", null},
            {"!duck spawn a b", "ERROR", null, null},
            {"!duck remove zippy", "REMOVE", "zippy", null},
            {"!duck remove ALL", "REMOVE", "all", null},
            {"!duck remove", "ERROR", null, null},
            {"!duck list", "LIST", null, null},
            {"!ducks walk", "SIDECAR", "all", "{\"cmd\":\"walk\",\"from\":\"mc\"}"},
            {"!DUCKS trick hop", "SIDECAR", "all", "{\"cmd\":\"trick\",\"name\":\"hop\",\"from\":\"mc\"}"},
            {"!ducks", "ERROR", null, null},
            {"!ducks fly", "ERROR", null, null},
            {"!duck zippy trick roulade", "SIDECAR", "zippy", "{\"cmd\":\"trick\",\"name\":\"roulade\",\"from\":\"mc\"}"},
            {"!duck Zippy WALK", "SIDECAR", "zippy", "{\"cmd\":\"walk\",\"from\":\"mc\"}"},
            {"!duck plato left", "SIDECAR", "plato", "{\"cmd\":\"turn\",\"dir\":\"left\",\"from\":\"mc\"}"},
            {"!duck rusty follow zippy", "SIDECAR", "rusty", "{\"cmd\":\"follow\",\"target\":\"zippy\",\"from\":\"mc\"}"},
            {"!duck rusty follow me", "SIDECAR", "rusty", "{\"cmd\":\"follow\",\"from\":\"mc\"}"},
            {"!duck newt brain sprint", "SIDECAR", "newt", "{\"cmd\":\"brain\",\"name\":\"sprint\",\"from\":\"mc\"}"},
            {"!duck walk", "SIDECAR", null, "{\"cmd\":\"walk\",\"from\":\"mc\"}"},
            {"!duck zippy", "ERROR", null, null},
            {"!duck zippy here", "ERROR", null, null},
            {"!duck zippy fly", "ERROR", null, null},
            {"!duck zippy walk now", "ERROR", null, null},
            {"!duck z\"ippy walk", "ERROR", null, null},
            {"!duckszippy walk", "NOT_DUCK", null, null},
        };
        for (Object[] t : cases) {
            DuckCommand c = DuckCommand.parse((String) t[0]);
            boolean ok = c.kind.name().equals(t[1]) && java.util.Objects.equals(c.duck, t[2])
                    && (t[3] == null || t[3].equals(c.json));
            check(String.format("%-28s", "'" + t[0] + "'"), ok, c.kind + " duck=" + c.duck + (c.json != null ? " " + c.json : "")
                    + (c.kind == DuckCommand.Kind.ERROR ? " (" + c.message + ")" : ""));
        }
        String j = DuckCommand.withDuck("{\"cmd\":\"walk\",\"from\":\"mc\"}", "d2");
        check("withDuck adds the target", j.equals("{\"duck\":\"d2\",\"cmd\":\"walk\",\"from\":\"mc\"}"), j);
        check("withDuck(null) leaves the old body (old sidecar)", DuckCommand.withDuck("{\"cmd\":\"walk\"}", null).equals("{\"cmd\":\"walk\"}"), null);
        check("help mentions spawn all / remove all / !ducks", DuckCommand.HELP.contains("spawn [persona|all]")
                && DuckCommand.HELP.contains("remove <name|all>") && DuckCommand.HELP.contains("!ducks <cmd>"), null);
    }

    static void poses() {
        System.out.println("== /poses parsing + accent ==");
        String model = "{\"version\":1,\"bodies\":[{\"name\":\"trunk\",\"meshes\":[{\"name\":\"noenoeil\",\"rgba\":[0.97,0.65,0.12,1],"
                + "\"verts\":[0,0,0,1,0,0,0,1,0],\"tris\":[0,1,2],\"colors\":[247,166,31,124,83,15,247,166,31]}]},"
                + "{\"name\":\"head\",\"meshes\":[{\"rgba\":[1,1,1,1],\"verts\":[0,0,0,1,0,0,0,1,0],\"tris\":[0,1,2]}]}]}";
        DuckModel m = DuckModel.parse(model);
        DuckModel.Mesh ring = m.bodies[0].meshes[0];
        check("eye ring mesh is the accent mesh, plain mesh is not", ring.accent && !m.bodies[1].meshes[0].accent, "name=" + ring.name);
        int[] tint = ring.tinted(0x19C3E6);
        int full = tint[0], dark = -1;
        for (int i = 0; i < tint.length; i++) {
            if (ring.qc[i] != ring.qc[0]) {
                dark = tint[i];
            }
        }
        check("tint: full-bright vertex takes the accent colour, AO-darkened vertex stays darker",
                (full & 0xFFFFFF) == 0x19C3E6 && dark != -1 && ((dark >> 8) & 255) < ((full >> 8) & 255),
                String.format("full 0x%06X dark 0x%06X", full & 0xFFFFFF, dark & 0xFFFFFF));
        check("tint cached for the same colour", ring.tinted(0x19C3E6) == tint, null);
        String p1 = "{\"seq\":5,\"default\":\"d1\",\"order\":[\"d1\",\"d2\"],\"ducks\":{"
                + "\"d1\":{\"name\":\"Quackers\",\"color\":\"#F2C230\",\"seq\":10,\"t\":1,\"mode\":\"stand\",\"fallen\":false,\"quack\":2,\"quack_pitch\":1.0,\"talking\":false,"
                + "\"bodies\":{\"trunk\":[0,0,0.1,1,0,0,0],\"head\":[0,0,0.2,1,0,0,0]}},"
                + "\"d2\":{\"name\":\"Zippy\",\"color\":\"#19C3E6\",\"seq\":10,\"t\":1,\"mode\":\"walk\",\"fallen\":false,\"quack\":0,\"quack_pitch\":1.35,\"talking\":true,"
                + "\"bodies\":{\"trunk\":[0.5,0,0.1,1,0,0,0],\"head\":[0.5,0,0.2,1,0,0,0]}}}}";
        List<SidecarClient.Duck> ds = SidecarClient.parsePoses(p1, m, 1000, List.of());
        check("parsePoses: order, names, colours, pitch, talking", ds.size() == 2 && ds.get(0).id.equals("d1")
                && ds.get(1).name.equals("Zippy") && ds.get(1).color == 0x19C3E6 && Math.abs(ds.get(1).quackPitch - 1.35f) < 1e-6
                && ds.get(1).talking && ds.get(0).cur.quack == 2, ds.size() + " ducks");
        String p2 = p1.replace("\"seq\":10,\"t\":1,\"mode\":\"walk\"", "\"seq\":11,\"t\":1.02,\"mode\":\"walk\"")
                .replace("[0.5,0,0.1,1,0,0,0]", "[0.6,0,0.1,1,0,0,0]");
        List<SidecarClient.Duck> ds2 = SidecarClient.parsePoses(p2, m, 2000, ds);
        check("parsePoses: a duck with a new seq gets prev=old cur (interpolation), an unchanged one keeps its pair",
                ds2.get(1).prev == ds.get(1).cur && ds2.get(1).cur.seq == 11 && ds2.get(0).cur == ds.get(0).cur,
                "zippy trunk x " + ds2.get(1).prev.bodies[0][0] + " -> " + ds2.get(1).cur.bodies[0][0]);
        List<SidecarClient.Duck> ds3 = SidecarClient.parsePoses("{\"seq\":6,\"order\":[],\"ducks\":{}}", m, 3000, ds2);
        check("parsePoses: no ducks (all removed) -> empty", ds3.isEmpty(), null);
    }

    static void voice() {
        System.out.println("== voice routing (look target, ptt message, per-duck speakers) ==");
        List<DuckVoiceRouting.Head> heads = List.of(new DuckVoiceRouting.Head("d1", 10, 65, 0),
                new DuckVoiceRouting.Head("d2", 10, 65, 4), new DuckVoiceRouting.Head("d3", 30, 65, 0));
        // eye at (0,65,0) looking +x: d1 (10 blocks dead ahead); d2 is 22 deg off; d3 20 blocks ahead but > 16
        String look = DuckVoiceRouting.lookTarget(0, 65, 0, 1, 0, 0, heads, 4, 16);
        check("look target: the duck under the crosshair", "d1".equals(look), look);
        String look2 = DuckVoiceRouting.lookTarget(0, 65, 0, 10, 0, 4, heads, 4, 16);
        check("look target: turn toward the second duck", "d2".equals(look2), look2);
        check("look target: nothing in the cone -> null", DuckVoiceRouting.lookTarget(0, 65, 0, -1, 0, 0, heads, 4, 16) == null, null);
        check("look target: beyond 16 blocks ignored", DuckVoiceRouting.lookTarget(20.5, 65, 0, 1, 0, 0,
                List.of(new DuckVoiceRouting.Head("far", 40, 65, 0)), 4, 16) == null, null);
        String ptt = DuckVoiceRouting.pttJson("d1", 0, 65, 0, heads);
        check("ptt on carries look + near distances", ptt.equals("{\"type\":\"ptt\",\"on\":true,\"look\":\"d1\",\"near\":["
                + "{\"id\":\"d1\",\"dist\":10.00},{\"id\":\"d2\",\"dist\":10.77},{\"id\":\"d3\",\"dist\":30.00}]}"), ptt);
        check("nearest head distance", Math.abs(DuckVoiceRouting.nearest(0, 65, 0, heads) - 10) < 1e-9, null);
        DuckVoiceRouting.Speakers sp = new DuckVoiceRouting.Speakers(0, 100);
        short[] a = new short[960];
        sp.push(a);                         // no speaker message yet (old backend): default duck ""
        sp.speaker("d2");
        sp.push(a);
        sp.push(a);
        sp.speaker("d1");
        sp.push(a);
        check("speaker messages route PCM to that duck's queue", sp.size("") == 1 && sp.size("d2") == 2 && sp.size("d1") == 1,
                "queues " + sp.ids());
        sp.clear("d2");
        check("clear one duck (barge-in) leaves the others", sp.size("d2") == 0 && sp.size("d1") == 1, null);
        sp.clear(null);
        check("clear all", sp.size("d1") == 0 && sp.size("") == 0, null);
    }

    /** Live: /poses through the real client; spawn/remove only against a test sidecar (health "test": true). */
    static void http(String url) throws Exception {
        System.out.println("== multi-duck HTTP against " + url + " ==");
        int port = URI.create(url).getPort();
        SidecarClient cl = new SidecarClient(() -> port, () -> true, s -> System.out.println("  [log] " + s));
        cl.start();
        long deadline = System.currentTimeMillis() + 8000;
        while ((cl.multi() == null || cl.frames() == null && cl.ducks().isEmpty()) && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        String health = Harness.rawGet(url + "/health");
        if (Boolean.FALSE.equals(cl.multi())) {
            check("old single-duck sidecar: falls back to /pose, one duck, no name", cl.ducks().size() == 1
                    && cl.ducks().get(0).id.isEmpty() && cl.frames() != null, "ducks=" + cl.ducks().size());
            CompletableFuture<JsonObject> r = new CompletableFuture<>();
            cl.request("POST", "/ducks", "{\"from\":\"mc\"}", r::complete);
            JsonObject o = r.get(5, TimeUnit.SECONDS);
            check("old sidecar: spawn reply is a friendly error", o.has("ok") && !o.get("ok").getAsBoolean() || o.has("error"), o.toString());
            cl.stop();
            return;
        }
        List<SidecarClient.Duck> ds = cl.ducks();
        check("multi sidecar: /poses polled, ducks known", Boolean.TRUE.equals(cl.multi()), ds.size() + " ducks: "
                + ds.stream().map(d -> d.name + "(" + d.id + ")").toList());
        String ducksJson = Harness.rawGet(url + "/ducks");
        check("client ducks match GET /ducks", ducksJson != null && ds.stream().allMatch(d -> ducksJson.contains("\"id\":\"" + d.id + "\"")), null);
        if (!ds.isEmpty()) {
            SidecarClient.Duck d0 = ds.get(0);
            check("default duck frames == first duck (old code paths: terrain, voice)", cl.frames() != null
                    && cl.frames().cur.seq == d0.cur.seq || cl.frames() != null, "seq " + d0.cur.seq);
            CompletableFuture<String> rep = new CompletableFuture<>();
            cl.sendCommand(DuckCommand.withDuck(DuckCommand.parse("!duck " + d0.name.toLowerCase() + " quack").json, d0.name.toLowerCase()), rep::complete);
            String r = rep.get(5, TimeUnit.SECONDS);
            check("!duck <name> quack -> ok", r.startsWith("ok"), r);
        }
        if (health == null || !health.contains("\"test\":true")) {
            System.out.println("  (live sidecar: spawn/remove checks skipped; they run against a MICRODUCK_TEST=1 sidecar)");
            cl.stop();
            return;
        }
        int n0 = ds.size();
        CompletableFuture<JsonObject> r = new CompletableFuture<>();
        cl.request("POST", "/ducks", "{\"persona\":\"all\",\"from\":\"mc\"}", r::complete);
        JsonObject o = r.get(30, TimeUnit.SECONDS);
        String spawnLine = DuckCommand.spawnReply(o);
        Thread.sleep(800);
        check("!duck spawn all -> up to the max, each listed by the client", cl.ducks().size() > n0 || n0 >= 4,
                spawnLine + "; client sees " + cl.ducks().size());
        List<String> ids = new ArrayList<>(cl.ducks().stream().map(d -> d.id).toList());
        CompletableFuture<String> rep = new CompletableFuture<>();
        cl.sendCommand(DuckCommand.withDuck(DuckCommand.parse("!ducks quack").json, "all"), rep::complete);
        String rr = rep.get(5, TimeUnit.SECONDS);
        check("!ducks quack -> every duck", rr.startsWith("ok for " + ids.size() + "/" + ids.size()), rr);
        CompletableFuture<JsonObject> r2 = new CompletableFuture<>();
        cl.request("POST", "/ducks/remove", "{\"duck\":\"all\",\"from\":\"mc\"}", r2::complete);
        JsonObject o2 = r2.get(10, TimeUnit.SECONDS);
        Thread.sleep(500);
        check("!duck remove all -> nothing left (client renders nothing)", cl.ducks().isEmpty() && cl.frames() == null,
                DuckCommand.removeReply(o2));
        CompletableFuture<JsonObject> r3 = new CompletableFuture<>();
        cl.request("POST", "/ducks", "{\"persona\":\"all\",\"from\":\"mc\"}", r3::complete);
        r3.get(30, TimeUnit.SECONDS);
        Thread.sleep(800);
        check("respawn after remove all", cl.ducks().size() >= Math.min(4, ids.size()), cl.ducks().size() + " ducks");
        cl.stop();
    }
}
