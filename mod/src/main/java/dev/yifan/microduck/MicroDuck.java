package dev.yifan.microduck;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.joml.Quaternionf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.client.renderer.rendertype.RenderType;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.texture.OverlayTexture;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.MutableComponent;
import net.minecraft.resources.Identifier;
import net.minecraft.sounds.SoundEvent;
import net.minecraft.sounds.SoundSource;
import net.minecraft.util.LightCoordsUtil;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

/**
 * Renders the Microducks from the local microduck-sim sidecar (every duck in GET /poses, or the single duck of an
 * older sidecar's /pose), articulated, around a fixed anchor in the world, each with its accent colour (eye ring)
 * and a name tag, and turns `!duck ...` / `!ducks ...` chat lines into sidecar commands (client-side only; never sent
 * to the server).
 */
public final class MicroDuck implements ClientModInitializer {
    public static final String ID = "microduck";
    static final Logger LOG = LoggerFactory.getLogger(ID);

    static final Identifier WHITE = Identifier.fromNamespaceAndPath(ID, "textures/misc/white.png");
    static final SoundEvent QUACK = SoundEvent.createVariableRangeEvent(Identifier.fromNamespaceAndPath(ID, "quack"));
    static final double MAX_DIST = 64.0;
    /** Name tags show within this many blocks (like a vanilla name tag on a nearby mob). */
    static final double TAG_DIST = 16.0;

    static DuckConfig config;
    static Path configFile;
    static SidecarClient client;
    static RenderType renderType;
    static RenderType cullRenderType;
    static final ConcurrentLinkedQueue<Component> REPLIES = new ConcurrentLinkedQueue<>();
    static final Map<String, Long> lastQuack = new HashMap<>();

    @Override
    public void onInitializeClient() {
        configFile = FabricLoader.getInstance().getConfigDir().resolve("microduck.properties");
        config = DuckConfig.load(configFile);
        saveConfig();                                  // write defaults on first run
        client = new SidecarClient(() -> config.port, () -> config.enabled, LOG::info);
        client.start();
        DuckTerrain.init();                            // mirror the blocks around the anchor into the sim (shared)
        ClientSendMessageEvents.ALLOW_CHAT.register(MicroDuck::allowChat);
        ClientTickEvents.END_CLIENT_TICK.register(MicroDuck::tick);
        LevelRenderEvents.COLLECT_SUBMITS.register(MicroDuck::render);
        LOG.info("microduck: sidecar {}, anchor {}", client.base(),
                config.hasAnchor() ? config.ax + "," + config.ay + "," + config.az + " in " + config.dim : "unset");
    }

    // ---------------------------------------------------------------- chat

    private static boolean allowChat(String message) {
        if (!DuckCommand.isDuck(message)) {
            return true;
        }
        try {
            handle(DuckCommand.parse(message));
        } catch (Exception e) {
            LOG.warn("microduck: command failed", e);
        }
        return false;                                  // swallow: never reaches the server
    }

    private static void handle(DuckCommand c) {
        Minecraft mc = Minecraft.getInstance();
        switch (c.kind) {
            case HELP -> say(c.message);
            case ERROR -> say(c.message);
            case SIDECAR -> {
                String sel = c.duck;
                String label;
                if (sel == null) {                                 // plain !duck <cmd>: the nearest (or only) duck
                    SidecarClient.Duck d = nearestDuck(mc);
                    sel = d == null || d.id.isEmpty() ? null : d.id;
                    label = d == null || d.id.isEmpty() ? null : d.name;
                } else {
                    label = sel.equals("all") ? "all ducks" : sel;
                }
                String what = c.cmd;
                String who = label;
                client.sendCommand(DuckCommand.withDuck(c.json, sel), r -> REPLIES.add(line(who, what + ": " + r)));
            }
            case HERE -> {
                if (mc.player == null || mc.level == null) {
                    return;
                }
                BlockPos p = mc.player.blockPosition();
                config.setAnchorBlock(p.getX(), p.getY(), p.getZ(), dimId(mc.level));
                saveConfig();
                say("anchor set to " + p.getX() + " " + p.getY() + " " + p.getZ() + " (" + config.dim + ")");
            }
            case SCALE -> {
                config.scale = c.scale;
                saveConfig();
                say("scale " + c.scale + " (duck ~" + Math.round(c.scale * 25) / 100.0 + " blocks tall)");
            }
            case NAMES -> {
                config.names = c.scale > 0;
                saveConfig();
                say("name tags " + (config.names ? "on" : "off"));
            }
            case BRAINS -> BrainsClient.fetch(client.base(), s -> REPLIES.add(line(s)));
            case SPAWN -> {
                if (Boolean.FALSE.equals(client.multi())) {
                    say("this sidecar runs one duck only (restart microduck-sim for several)");
                    return;
                }
                String body = c.duck == null ? "{\"from\":\"mc\"}" : "{\"persona\":\"" + c.duck + "\",\"from\":\"mc\"}";
                client.request("POST", "/ducks", body, o -> REPLIES.add(line(DuckCommand.spawnReply(o))));
            }
            case REMOVE -> client.request("POST", "/ducks/remove", "{\"duck\":\"" + c.duck + "\",\"from\":\"mc\"}",
                    o -> REPLIES.add(line(DuckCommand.removeReply(o))));
            case LIST -> listDucks(mc);
            default -> {
            }
        }
    }

    static void listDucks(Minecraft mc) {
        List<SidecarClient.Duck> ds = client.ducks();
        if (ds.isEmpty()) {
            say(Boolean.TRUE.equals(client.multi()) ? "no ducks here: !duck spawn [persona|all]" : "sidecar not connected");
            return;
        }
        for (SidecarClient.Duck d : ds) {
            float[] t = SidecarClient.trunk(d.cur, client.model());
            String dist = "";
            if (t != null && mc.player != null && config.hasAnchor()) {
                double[] p = DuckMath.simToMc(config.ax, config.ay, config.az, config.scale, t[0], t[1], t[2]);
                dist = String.format(Locale.ROOT, ", %.1f blocks away", Math.sqrt(mc.player.distanceToSqr(p[0], p[1], p[2])));
            }
            REPLIES.add(Component.literal("[Duck] ").withColor(0xF2C230).append(Component.literal(d.name).withColor(d.color))
                    .append(Component.literal((d.id.isEmpty() ? "" : " (" + d.id + ")") + ": " + d.cur.mode
                            + (d.cur.fallen ? ", fallen" : "") + dist).withColor(0xE0E0E0)));
        }
    }

    /** The duck closest to the local player (MC distance), or the only/first one; null if none. */
    static SidecarClient.Duck nearestDuck(Minecraft mc) {
        List<SidecarClient.Duck> ds = client.ducks();
        if (ds.isEmpty()) {
            return null;
        }
        if (ds.size() == 1 || mc.player == null || !config.hasAnchor()) {
            return ds.get(0);
        }
        SidecarClient.Duck best = ds.get(0);
        double bd = Double.MAX_VALUE;
        for (SidecarClient.Duck d : ds) {
            float[] t = SidecarClient.trunk(d.cur, client.model());
            if (t == null) {
                continue;
            }
            double[] p = DuckMath.simToMc(config.ax, config.ay, config.az, config.scale, t[0], t[1], t[2]);
            double dd = mc.player.distanceToSqr(p[0], p[1], p[2]);
            if (dd < bd) {
                bd = dd;
                best = d;
            }
        }
        return best;
    }

    static void say(String s) {
        REPLIES.add(line(s));
    }

    /** "[Zippy] text" caption in the duck's colour (voice replies). */
    static void sayAs(String name, int color, String s) {
        REPLIES.add(Component.literal("[" + name + "] ").withColor(color).append(Component.literal(s).withColor(0xE0E0E0)));
    }

    static MutableComponent line(String s) {
        return Component.literal("[Duck] ").withColor(0xF2C230).append(Component.literal(s).withColor(0xE0E0E0));
    }

    static MutableComponent line(String who, String s) {
        return who == null ? line(s) : Component.literal("[" + who + "] ").withColor(0xF2C230).append(Component.literal(s).withColor(0xE0E0E0));
    }

    static void saveConfig() {
        try {
            config.save(configFile);
        } catch (IOException e) {
            LOG.warn("microduck: cannot save {}: {}", configFile, e.toString());
        }
    }

    static String dimId(ClientLevel level) {
        return level.dimension().identifier().toString();
    }

    // ---------------------------------------------------------------- tick: chat replies, player position, quacks

    private static void tick(Minecraft mc) {
        Component r;
        int n = 0;
        while (n++ < 10 && (r = REPLIES.poll()) != null) {
            if (mc.level != null) {
                mc.gui.hud.getChat().addClientSystemMessage(r);
            }
        }
        boolean here = mc.level != null && mc.player != null && config.enabled && config.hasAnchor()
                && dimId(mc.level).equals(config.dim);
        if (!here) {
            client.setPlayer(null);
        } else if (mc.level.getGameTime() % 4 == 0) {
            client.setPlayer(DuckMath.mcToSim(config.ax, config.az, config.scale, mc.player.getX(), mc.player.getZ()));
        }
        DuckModel model = client.model();
        List<SidecarClient.Duck> ds = client.ducks();
        if (model == null) {
            return;
        }
        for (SidecarClient.Duck d : ds) {
            long q = d.cur.quack;
            Long last = lastQuack.get(d.id);
            if (last == null || q < last) {
                lastQuack.put(d.id, q);                // new duck / sidecar restarted: new baseline, no sound
            } else if (q > last) {
                lastQuack.put(d.id, q);
                float[] t = SidecarClient.trunk(d.cur, model);
                if (here && t != null) {
                    double[] p = DuckMath.simToMc(config.ax, config.ay, config.az, config.scale, t[0], t[1], t[2]);
                    if (mc.player.distanceToSqr(p[0], p[1], p[2]) < MAX_DIST * MAX_DIST) {
                        float pitch = d.quackPitch * (0.95f + 0.15f * (float) Math.random());
                        mc.level.playLocalSound(p[0], p[1], p[2], QUACK, SoundSource.NEUTRAL, 1.0f,
                                Math.max(0.5f, Math.min(2.0f, pitch)), false);
                    }
                }
            }
        }
        if (lastQuack.size() > ds.size() + 8) {
            lastQuack.keySet().retainAll(ds.stream().map(x -> x.id).toList());
        }
    }

    // ---------------------------------------------------------------- rendering

    static boolean renderBroken;

    private static void render(LevelRenderContext ctx) {
        if (renderBroken) {
            return;
        }
        try {
            renderDucks(ctx);
        } catch (RuntimeException e) {
            renderBroken = true;                       // one log line, never spam or crash the frame loop
            LOG.warn("microduck: render failed, duck rendering disabled until restart", e);
        }
    }

    private static void renderDucks(LevelRenderContext ctx) {
        Minecraft mc = Minecraft.getInstance();
        if (!config.enabled || !config.hasAnchor() || mc.level == null || !dimId(mc.level).equals(config.dim)) {
            return;
        }
        DuckModel model = client.model();
        List<SidecarClient.Duck> ds = client.ducks();
        if (model == null || ds.isEmpty()) {
            return;
        }
        boolean tags = config.names && (ds.size() > 1 || !ds.get(0).id.isEmpty());   // an old single-duck sidecar: no name tag
        long now = System.nanoTime();
        for (SidecarClient.Duck d : ds) {
            renderDuck(ctx, mc, model, d, SidecarClient.alpha(d.prev, d.cur, now), tags);
        }
    }

    private static void renderDuck(LevelRenderContext ctx, Minecraft mc, DuckModel model, SidecarClient.Duck duck,
                                   float alpha, boolean tag) {
        CameraRenderState cam = ctx.levelState().cameraRenderState;
        Vec3 cp = cam.pos;
        double s = config.scale, ax = config.ax, ay = config.ay, az = config.az;
        int nb = model.bodies.length;
        float[][] poses = new float[nb][];
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE, minZ = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE, maxZ = -Double.MAX_VALUE;
        for (int i = 0; i < nb; i++) {
            float[] a = duck.prev.bodies.length == nb ? duck.prev.bodies[i] : null;
            float[] b = duck.cur.bodies.length == nb ? duck.cur.bodies[i] : null;
            if (b == null) {
                continue;
            }
            float[] p = a == null ? b.clone() : DuckMath.lerpPose(a, b, alpha, null);
            poses[i] = p;
            double[] o = DuckMath.simToMc(ax, ay, az, s, p[0], p[1], p[2]);
            minX = Math.min(minX, o[0]);
            minY = Math.min(minY, o[1]);
            minZ = Math.min(minZ, o[2]);
            maxX = Math.max(maxX, o[0]);
            maxY = Math.max(maxY, o[1]);
            maxZ = Math.max(maxZ, o[2]);
        }
        if (minX == Double.MAX_VALUE) {
            return;
        }
        double cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
        double d2 = cp.distanceToSqr(cx, cy, cz);
        if (d2 > MAX_DIST * MAX_DIST) {
            return;
        }
        double pad = model.radius * s + 0.25;
        AABB box = new AABB(minX - pad, minY - pad, minZ - pad, maxX + pad, maxY + pad + (tag ? 0.6 : 0), maxZ + pad);
        Frustum fr = cam.cullFrustum;
        if (fr != null && !fr.isVisible(box)) {
            return;
        }
        // light at the duck: the air block just above its lowest body (feet), and the one above that
        BlockPos lp = BlockPos.containing(cx, minY + 0.1, cz);
        int light = LightCoordsUtil.max(LightCoordsUtil.getLightCoords(mc.level, lp),
                LightCoordsUtil.getLightCoords(mc.level, lp.above()));

        if (renderType == null) {
            renderType = RenderTypes.entityCutout(WHITE);          // double-sided (PER_FACE_LIGHTING)
            cullRenderType = RenderTypes.entityCutoutCull(WHITE);  // back faces culled, for closed meshes
        }
        PoseStack ps = ctx.poseStack();
        float fs = (float) s;
        submitBlobShadow(mc, ctx, ps, cp, cx, cz, ay, s);
        final int accent = duck.color;
        for (int i = 0; i < nb; i++) {
            float[] p = poses[i];
            DuckModel.Body body = model.bodies[i];
            if (p == null || body.meshes.length == 0) {
                continue;
            }
            double[] o = DuckMath.simToMc(ax, ay, az, s, p[0], p[1], p[2]);
            double[] q = DuckMath.quatSimToMc(p[3], p[4], p[5], p[6]);
            ps.pushPose();
            ps.translate(o[0] - cp.x, o[1] - cp.y, o[2] - cp.z);
            ps.mulPose(new Quaternionf((float) q[1], (float) q[2], (float) q[3], (float) q[0]));
            ps.scale(fs, fs, fs);
            final int lightF = light;
            if (body.anyNoCull) {
                ctx.submitNodeCollector().submitCustomGeometry(ps, renderType, (pose, vc) -> emit(body, pose, vc, lightF, false, accent));
            }
            if (body.anyCull) {
                ctx.submitNodeCollector().submitCustomGeometry(ps, cullRenderType, (pose, vc) -> emit(body, pose, vc, lightF, true, accent));
            }
            ps.popPose();
        }
        if (tag && d2 < TAG_DIST * TAG_DIST && !mc.gui.hud.isHidden()) {
            float[] t = SidecarClient.trunk(duck.cur, model);
            if (t != null) {
                double[] o = DuckMath.simToMc(ax, ay, az, s, t[0], t[1], t[2]);
                ps.pushPose();
                ps.translate(o[0] - cp.x, maxY - cp.y, o[2] - cp.z);
                ctx.submitNodeCollector().submitNameTag(ps, new Vec3(0, 0.12 * s + 0.15, 0), 0,
                        Component.literal(duck.name).withColor(nameColor(duck.color)), true, light, cam);
                ps.popPose();
            }
        }
    }

    /** Name tag text: the accent colour lifted toward white so it stays readable on the tag's dark plate. */
    static int nameColor(int rgb) {
        int r = (rgb >> 16) & 255, g = (rgb >> 8) & 255, b = rgb & 255;
        r = r + (255 - r) * 2 / 5;
        g = g + (255 - g) * 2 / 5;
        b = b + (255 - b) * 2 / 5;
        return (r << 16) | (g << 8) | b;
    }

    // ---------------------------------------------------------------- soft blob shadow on the floor

    static final Identifier SHADOW_TEX = Identifier.withDefaultNamespace("textures/misc/shadow.png");
    static RenderType shadowRenderType;
    static Boolean irisPresent;

    /** True when an Iris shader pack is active (it draws real shadows; skip the blob then). */
    static boolean shaderPackInUse() {
        try {
            if (irisPresent == null) {
                irisPresent = FabricLoader.getInstance().isModLoaded("iris");
            }
            if (!irisPresent) {
                return false;
            }
            Class<?> api = Class.forName("net.irisshaders.iris.api.v0.IrisApi");
            Object inst = api.getMethod("getInstance").invoke(null);
            return (Boolean) api.getMethod("isShaderPackInUse").invoke(inst);
        } catch (ReflectiveOperationException | RuntimeException | LinkageError e) {
            irisPresent = false;
            return false;
        }
    }

    /** Vanilla-style shadow quad (shadow.png, entity_shadow pipeline) on the floor under the duck. */
    static void submitBlobShadow(Minecraft mc, LevelRenderContext ctx, PoseStack ps, Vec3 cp, double cx, double cz,
                                 double floorY, double s) {
        if (!mc.options.entityShadows().get() || shaderPackInUse()) {
            return;
        }
        if (shadowRenderType == null) {
            shadowRenderType = RenderTypes.entityShadow(SHADOW_TEX);
        }
        final float r = (float) (0.11 * s);                  // ~ the duck's footprint (sim 11 cm)
        final float y = (float) (floorY + 0.015625 - cp.y);   // just above the floor, like vanilla shadows
        final float x0 = (float) (cx - cp.x), z0 = (float) (cz - cp.z);
        final int col = (Math.round(0.55f * 255) << 24) | 0xFFFFFF;
        ps.pushPose();
        ctx.submitNodeCollector().submitCustomGeometry(ps, shadowRenderType, (pose, vc) -> {
            float[][] c = {{-r, -r, 0, 0}, {-r, r, 0, 1}, {r, r, 1, 1}, {r, -r, 1, 0}};  // CCW seen from above
            for (float[] v : c) {
                vc.addVertex(pose, x0 + v[0], y, z0 + v[1]).setColor(col).setUv(v[2], v[3])
                        .setOverlay(OverlayTexture.NO_OVERLAY).setLight(LightCoordsUtil.FULL_BRIGHT)
                        .setNormal(pose, 0, 1, 0);
            }
        });
        ps.popPose();
    }

    /**
     * Entity render types draw QUADS. DuckModel packs each mesh as quads: paired triangles (smooth meshes) or
     * degenerate quads (one per triangle), with per-vertex normals and (AO-baked) colours. `cullPass` selects
     * the closed meshes (entity_cutout_cull) or the rest (double-sided entity_cutout). Accent meshes (the eye ring)
     * take the duck's persona colour.
     */
    static void emit(DuckModel.Body body, PoseStack.Pose pose, VertexConsumer vc, int light, boolean cullPass, int accent) {
        int overlay = OverlayTexture.NO_OVERLAY;
        for (DuckModel.Mesh m : body.meshes) {
            if (m.cull != cullPass) {
                continue;
            }
            float[] v = m.qv;
            float[] n = m.qn;
            int[] col = m.accent && accent != SidecarClient.DEFAULT_COLOR ? m.tinted(accent) : m.qc;
            int nv = col.length;
            for (int k = 0; k < nv; k++) {
                int j = 3 * k;
                vc.addVertex(pose, v[j], v[j + 1], v[j + 2])
                        .setColor(col[k])
                        .setUv(0.5f, 0.5f)
                        .setOverlay(overlay)
                        .setLight(light)
                        .setNormal(pose, n[j], n[j + 1], n[j + 2]);
            }
        }
    }
}
