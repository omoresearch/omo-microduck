package dev.yifan.microduck;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import de.maxhenkel.voicechat.api.Position;
import de.maxhenkel.voicechat.api.ServerLevel;
import de.maxhenkel.voicechat.api.ServerPlayer;
import de.maxhenkel.voicechat.api.VoicechatApi;
import de.maxhenkel.voicechat.api.VoicechatConnection;
import de.maxhenkel.voicechat.api.VoicechatPlugin;
import de.maxhenkel.voicechat.api.VoicechatServerApi;
import de.maxhenkel.voicechat.api.VolumeCategory;
import de.maxhenkel.voicechat.api.audiochannel.LocationalAudioChannel;
import de.maxhenkel.voicechat.api.events.EventRegistration;
import de.maxhenkel.voicechat.api.events.MicrophonePacketEvent;
import de.maxhenkel.voicechat.api.events.VoicechatServerStartedEvent;
import de.maxhenkel.voicechat.api.events.VoicechatServerStoppedEvent;
import de.maxhenkel.voicechat.api.opus.OpusDecoder;
import de.maxhenkel.voicechat.api.opus.OpusEncoder;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;

/**
 * Simple Voice Chat plugin (entrypoint "voicechat"; only loaded when SVC is installed - the mod lists it
 * under "suggests" and nothing else references this class).
 * <p>Runs on SVC's server side, i.e. the integrated server in singleplayer/LAN:
 * <ol>
 *   <li>MicrophonePacketEvent: keep only the local player's push-to-talk (or {@code player=} from the config) within
 *       {@code hear_range} blocks of the nearest duck, decode Opus with the API decoder and stream PCM to the voice
 *       backend. Each push-to-talk starts with {@code {"type":"ptt","on":true,"look":<duck under the crosshair>,
 *       "near":[{id,dist}]}} so the backend can pick which duck is being talked to (DuckVoiceRouting).</li>
 *   <li>The backend streams each duck's reply PCM back, announced by {@code {"type":"speaker","duck":id}}; a 20 ms
 *       sender encodes each duck's queue with its own encoder and sends it on that duck's LocationalAudioChannel
 *       following its body, so every duck talks from where it stands (game audio, so OBS captures it).</li>
 * </ol>
 */
public final class DuckVoicePlugin implements VoicechatPlugin {
    static final String CATEGORY = "microduck";
    static final UUID CHANNEL_ID = UUID.nameUUIDFromBytes("microduck-voice".getBytes());

    private volatile DuckVoiceConfig config = new DuckVoiceConfig();
    private volatile VoicechatServerApi api;
    private volatile DuckVoiceLink link;
    private final DuckVoiceRouting.Speakers out = new DuckVoiceRouting.Speakers(3, 50 * 30);
    private OpusDecoder decoder;
    private final Map<String, Voice> voices = new HashMap<>();   // per duck id ("" = default / old single duck)
    private volatile ServerLevel duckLevel;
    private ScheduledExecutorService sender;
    private volatile boolean talking; // player's PTT currently streaming
    private long lastMicMs;

    /** One duck's outgoing voice: its own Opus encoder state and locational channel. */
    private static final class Voice {
        OpusEncoder encoder;
        LocationalAudioChannel channel;
        boolean sending;
        int quietTicks;
    }

    @Override
    public String getPluginId() {
        return MicroDuck.ID;
    }

    @Override
    public void initialize(VoicechatApi voicechat) {
        Path file = FabricLoader.getInstance().getConfigDir().resolve("microduck-voice.properties");
        config = DuckVoiceConfig.load(file);
        try {
            config.save(file);
        } catch (Exception e) {
            MicroDuck.LOG.warn("microduck voice: cannot write {}: {}", file, e.toString());
        }
        MicroDuck.LOG.info("microduck voice: Simple Voice Chat plugin loaded (enabled={}, backend {}, hear_range {})",
                config.enabled, config.url, config.hearRange <= 0 ? "always" : config.hearRange);
    }

    @Override
    public void registerEvents(EventRegistration events) {
        events.registerEvent(VoicechatServerStartedEvent.class, this::onStarted);
        events.registerEvent(VoicechatServerStoppedEvent.class, e -> stop());
        events.registerEvent(MicrophonePacketEvent.class, this::onMic);
    }

    private synchronized void onStarted(VoicechatServerStartedEvent e) {
        if (!config.enabled) {
            return;
        }
        api = e.getVoicechat();
        try {
            VolumeCategory cat = api.volumeCategoryBuilder().setId(CATEGORY).setName("Microduck")
                    .setDescription("The Microducks' voices").build();
            api.registerVolumeCategory(cat);
        } catch (RuntimeException ex) {
            MicroDuck.LOG.warn("microduck voice: volume category: {}", ex.toString());
        }
        decoder = api.createDecoder();
        link = new DuckVoiceLink(config.url, this::onVoice, this::onControl, s -> MicroDuck.LOG.info("microduck voice: {}", s));
        link.ensureConnected();
        sender = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "microduck-voice");
            t.setDaemon(true);
            return t;
        });
        sender.scheduleAtFixedRate(this::tick, 20, 20, TimeUnit.MILLISECONDS);
        MicroDuck.LOG.info("microduck voice: voice chat server started; listening for push-to-talk near the ducks");
    }

    private synchronized void stop() {
        if (sender != null) {
            sender.shutdownNow();
            sender = null;
        }
        if (link != null) {
            link.close();
            link = null;
        }
        if (decoder != null) {
            decoder.close();
            decoder = null;
        }
        for (Voice v : voices.values()) {
            if (v.encoder != null) {
                v.encoder.close();
            }
        }
        voices.clear();
        api = null;
        out.clear(null);
    }

    // ---------------------------------------------------------------- mic in (SVC network thread)

    private void onMic(MicrophonePacketEvent e) {
        DuckVoiceLink l = link;
        VoicechatConnection from = e.getSenderConnection();
        if (l == null || from == null || decoder == null) {
            return;
        }
        ServerPlayer sp = from.getPlayer();
        if (sp == null || !isOurPlayer(sp)) {
            return;
        }
        if (companionHasFocus()) {                        // GLaDOS / the RP1 own the mic right now (omo.voice.focus)
            endUtterance(l);
            return;
        }
        List<DuckVoiceRouting.Head> heads = heads();
        String dim = levelDim(sp.getServerLevel());
        Position p = sp.getPosition();
        double dist = Double.NaN;
        if (!heads.isEmpty() && dim != null && dim.equals(duckDim())) {
            dist = DuckVoiceRouting.nearest(p.getX(), p.getY(), p.getZ(), heads);
        }
        if (!config.inRange(dist)) {
            return;
        }
        duckLevel = sp.getServerLevel();
        byte[] opus = e.getPacket().getOpusEncodedData();
        if (opus == null || opus.length == 0) {          // SVC sends an empty packet when push-to-talk ends
            endUtterance(l);
            return;
        }
        short[] pcm;
        synchronized (this) {
            if (decoder == null) {
                return;
            }
            pcm = decoder.decode(opus);
        }
        if (!talking) {
            talking = true;
            l.sendText(pttOn(p, heads));
        }
        lastMicMs = System.currentTimeMillis();
        l.sendPcm(pcm);
    }

    /** ptt on + the duck under the crosshair + every duck's distance (blocks), for the backend's routing. */
    private String pttOn(Position p, List<DuckVoiceRouting.Head> heads) {
        String look = null;
        try {
            Minecraft mc = Minecraft.getInstance();
            var pl = mc == null ? null : mc.player;
            if (pl != null && heads.size() > 1) {
                var eye = pl.getEyePosition();
                var view = pl.getViewVector(1f);
                DuckConfig c = MicroDuck.config;
                look = DuckVoiceRouting.lookTarget(eye.x, eye.y, eye.z, view.x, view.y, view.z, heads,
                        c == null ? 4 : c.scale, 16);
            }
        } catch (RuntimeException ignored) {
            // client state not readable right now: no look target
        }
        return DuckVoiceRouting.pttJson(look, p.getX(), p.getY(), p.getZ(), heads);
    }

    private void endUtterance(DuckVoiceLink l) {
        if (talking) {
            talking = false;
            l.sendText("{\"type\":\"ptt\",\"on\":false}");
            synchronized (this) {
                if (decoder != null) {
                    decoder.resetState();
                }
            }
        }
    }

    private boolean isOurPlayer(ServerPlayer sp) {
        String want = config.player;
        Object mc = sp.getPlayer();
        if (want != null && !want.isEmpty()) {
            return mc instanceof net.minecraft.world.entity.Entity ent && want.equalsIgnoreCase(ent.getScoreboardName());
        }
        Minecraft client = Minecraft.getInstance();
        var local = client == null ? null : client.player;
        return local != null && local.getUUID().equals(sp.getUuid());
    }

    // ---------------------------------------------------------------- voice out (backend -> ducks)

    private void onVoice(short[] pcm) {
        out.push(pcm);
    }

    private void onControl(String json) {
        try {
            JsonObject m = JsonParser.parseString(json).getAsJsonObject();
            String type = m.has("type") ? m.get("type").getAsString() : "";
            String duck = m.has("duck") && !m.get("duck").isJsonNull() ? m.get("duck").getAsString() : null;
            switch (type) {
                case "speaker" -> out.speaker(duck);
                case "clear" -> out.clear(duck);
                case "transcript" -> {
                    if (config.captions && m.has("text")) {
                        boolean assistant = "assistant".equals(m.has("role") ? m.get("role").getAsString() : "");
                        String text = m.get("text").getAsString().trim();
                        String name = m.has("name") && !m.get("name").isJsonNull() ? m.get("name").getAsString() : null;
                        if (assistant && name != null) {
                            SidecarClient sc = MicroDuck.client;
                            SidecarClient.Duck d = sc == null || duck == null ? null : sc.duck(duck);
                            MicroDuck.sayAs(name, d != null ? d.color : 0xF2C230, text);
                        } else {
                            MicroDuck.say((assistant ? "" : "(heard" + (name != null ? " by " + name : "") + ") ") + text);
                        }
                    }
                }
                case "status" -> {
                    if (m.has("level") && "error".equals(m.get("level").getAsString())) {
                        MicroDuck.say("voice: " + m.get("text").getAsString());
                    }
                }
                default -> {
                }
            }
        } catch (RuntimeException ignored) {
            // not JSON / unexpected shape: ignore
        }
    }

    /** 20 ms sender: one Opus frame per duck per tick from its reply queue onto its locational channel. */
    private void tick() {
        try {
            DuckVoiceLink l = link;
            if (l != null && !l.isOpen()) {
                l.ensureConnected();
            }
            if (talking && System.currentTimeMillis() - lastMicMs > 400 && l != null) {
                endUtterance(l);                          // no end packet arrived (e.g. walked out of range)
            }
            for (String id : out.ids()) {
                Voice v = voices.computeIfAbsent(id, k -> new Voice());
                short[] frame = out.poll(id);
                if (frame == null) {
                    if (v.sending && ++v.quietTicks > 5) {
                        LocationalAudioChannel ch = v.channel;
                        if (ch != null) {
                            ch.flush();
                        }
                        synchronized (this) {
                            if (v.encoder != null) {
                                v.encoder.resetState();
                            }
                        }
                        v.sending = false;
                    }
                    continue;
                }
                v.quietTicks = 0;
                LocationalAudioChannel ch = ensureChannel(id, v);
                if (ch == null) {
                    continue;
                }
                byte[] opus;
                synchronized (this) {
                    VoicechatServerApi a = api;
                    if (a == null) {
                        return;
                    }
                    if (v.encoder == null) {
                        v.encoder = a.createEncoder();
                    }
                    opus = v.encoder.encode(frame);
                }
                ch.send(opus);
                v.sending = true;
            }
        } catch (RuntimeException e) {
            MicroDuck.LOG.warn("microduck voice: sender: {}", e.toString());
        }
    }

    private LocationalAudioChannel ensureChannel(String id, Voice v) {
        VoicechatServerApi a = api;
        double[] duck = duckPos(id);
        if (a == null || duck == null) {
            return null;
        }
        ServerLevel level = duckLevel;
        if (level == null) {
            return null;
        }
        Position pos = a.createPosition(duck[0], duck[1], duck[2]);
        LocationalAudioChannel ch = v.channel;
        if (ch == null || ch.isClosed()) {
            UUID cid = id.isEmpty() ? CHANNEL_ID : UUID.nameUUIDFromBytes(("microduck-voice-" + id).getBytes());
            ch = a.createLocationalAudioChannel(cid, level, pos);
            if (ch == null) {
                return null;
            }
            ch.setCategory(CATEGORY);
            ch.setDistance(config.voiceDistance);
            v.channel = ch;
        } else {
            ch.updateLocation(pos);                       // the duck walks: its voice follows
        }
        return ch;
    }

    // ---------------------------------------------------------------- where are the ducks (client-side state, same JVM)

    /** Every duck's head position (MC block coords) from the renderer's config + latest sidecar poses. */
    static List<DuckVoiceRouting.Head> heads() {
        List<DuckVoiceRouting.Head> hs = new ArrayList<>();
        DuckConfig c = MicroDuck.config;
        SidecarClient sc = MicroDuck.client;
        if (c == null || !c.hasAnchor() || sc == null || sc.model() == null) {
            return hs;
        }
        for (SidecarClient.Duck d : sc.ducks()) {
            float[] t = SidecarClient.trunk(d.cur, sc.model());
            if (t == null) {
                continue;
            }
            double[] p = DuckMath.simToMc(c.ax, c.ay, c.az, c.scale, t[0], t[1], t[2]);
            hs.add(new DuckVoiceRouting.Head(d.id, p[0], p[1] + 0.12 * c.scale, p[2]));
        }
        return hs;
    }

    /** One duck's head position in MC block coords ("" / unknown id: the default duck); null if no anchor. */
    static double[] duckPos(String id) {
        DuckConfig c = MicroDuck.config;
        if (c == null || !c.hasAnchor()) {
            return null;
        }
        SidecarClient sc = MicroDuck.client;
        SidecarClient.Duck d = sc == null ? null : sc.duck(id == null ? "" : id);
        if (d == null && sc != null && !sc.ducks().isEmpty()) {
            d = sc.ducks().get(0);
        }
        float[] t = d == null || sc.model() == null ? null : SidecarClient.trunk(d.cur, sc.model());
        if (t == null) {
            return new double[]{c.ax, c.ay + 0.6, c.az};
        }
        double[] p = DuckMath.simToMc(c.ax, c.ay, c.az, c.scale, t[0], t[1], t[2]);
        p[1] += 0.12 * c.scale;                           // trunk -> roughly head height
        return p;
    }

    /** The default duck's head (kept for other mods/tests that used the single-duck API). */
    static double[] duckPos() {
        return duckPos("");
    }

    static String duckDim() {
        DuckConfig c = MicroDuck.config;
        return c == null ? null : c.dim;
    }

    static String levelDim(ServerLevel level) {
        Object o = level == null ? null : level.getServerLevel();
        if (o instanceof net.minecraft.world.level.Level l) {
            return l.dimension().identifier().toString();
        }
        return null;
    }

    /**
     * glados-mod and rp1-mod publish {@code "<owner>:<epoch ms>"} every client tick they own the mic (their scored
     * claims decide which one); yield while any non-duck owner's stamp is < 1 s old.
     */
    static boolean companionHasFocus() {
        String f = System.getProperty("omo.voice.focus", "");
        int c = f.lastIndexOf(':');
        if (c <= 0 || f.startsWith("duck")) {
            return false;
        }
        try {
            return System.currentTimeMillis() - Long.parseLong(f.substring(c + 1)) < 1000;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
