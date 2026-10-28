package dev.yifan.microduck;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;

/**
 * Loopback WebSocket link to the duck voice backend (voice-live/duck.mjs, default ws://127.0.0.1:8798/svc).
 * Pure Java (java.net.http), no Minecraft or Simple Voice Chat types, so it is unit-testable.
 * <p>Wire: binary = PCM16 LE mono 48 kHz (mic out, duck voice in); text = small JSON control messages
 * (out: hello/start/hangup; in: status/transcript/clear).
 */
public final class DuckVoiceLink {
    private final URI uri;
    private final Consumer<short[]> onVoice;
    private final Consumer<String> onText;
    private final Consumer<String> log;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    private volatile WebSocket ws;
    private volatile CompletableFuture<WebSocket> connecting;
    private volatile long nextTry;
    private volatile boolean closed;
    private boolean warned;
    private final Object sendLock = new Object();
    private CompletableFuture<?> sendChain = CompletableFuture.completedFuture(null);
    private volatile long sentBytes, recvBytes;

    public DuckVoiceLink(String url, Consumer<short[]> onVoice, Consumer<String> onText, Consumer<String> log) {
        this.uri = URI.create(url);
        this.onVoice = onVoice;
        this.onText = onText;
        this.log = log;
    }

    public boolean isOpen() {
        WebSocket w = ws;
        return w != null && !w.isOutputClosed() && !w.isInputClosed();
    }

    public long sentBytes() {
        return sentBytes;
    }

    public long recvBytes() {
        return recvBytes;
    }

    /** Non-blocking: starts a connection attempt unless one is running or we are backing off. */
    public void ensureConnected() {
        if (closed || isOpen() || connecting != null || System.currentTimeMillis() < nextTry) {
            return;
        }
        nextTry = System.currentTimeMillis() + 3000;
        connecting = http.newWebSocketBuilder().connectTimeout(Duration.ofSeconds(3)).buildAsync(uri, new Listener());
        connecting.whenComplete((w, err) -> {
            connecting = null;
            if (err != null) {
                if (!warned) {
                    log.accept("duck voice backend not reachable at " + uri + " (" + err.getClass().getSimpleName() + "); start voice-live/duck-start.sh");
                    warned = true;
                }
                return;
            }
            ws = w;
            warned = false;
            log.accept("duck voice link connected to " + uri);
            sendText("{\"type\":\"hello\",\"mod\":\"microduck\",\"player\":\"local\",\"proto\":2}");  // proto 2: multi-duck
        });
    }

    /** Waits up to timeoutMs for the link (used by tests and the first push-to-talk). */
    public boolean awaitOpen(long timeoutMs) {
        long until = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < until) {
            ensureConnected();
            if (isOpen()) {
                return true;
            }
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return isOpen();
    }

    public void sendPcm(short[] pcm) {
        WebSocket w = ws;
        if (w == null || closed) {
            ensureConnected();
            return;
        }
        byte[] b = VoicePcm.toBytes(pcm);
        sentBytes += b.length;
        enqueue(() -> w.sendBinary(ByteBuffer.wrap(b), true));
    }

    public void sendText(String json) {
        WebSocket w = ws;
        if (w == null || closed) {
            return;
        }
        enqueue(() -> w.sendText(json, true));
    }

    // java.net.http.WebSocket allows one outstanding send at a time: chain them.
    private void enqueue(java.util.function.Supplier<CompletableFuture<WebSocket>> op) {
        synchronized (sendLock) {
            sendChain = sendChain.handle((r, e) -> null).thenCompose(x -> op.get()).exceptionally(e -> {
                drop();
                return null;
            });
        }
    }

    private void drop() {
        WebSocket w = ws;
        ws = null;
        if (w != null) {
            w.abort();
        }
    }

    public void close() {
        closed = true;
        WebSocket w = ws;
        ws = null;
        if (w != null) {
            try {
                w.sendClose(WebSocket.NORMAL_CLOSURE, "bye").orTimeout(1, java.util.concurrent.TimeUnit.SECONDS);
            } catch (Exception ignored) {
                w.abort();
            }
        }
    }

    private final class Listener implements WebSocket.Listener {
        private final ByteArrayOutputStream bin = new ByteArrayOutputStream();
        private final StringBuilder txt = new StringBuilder();

        @Override
        public CompletionStage<?> onBinary(WebSocket webSocket, ByteBuffer data, boolean last) {
            byte[] chunk = new byte[data.remaining()];
            data.get(chunk);
            bin.write(chunk, 0, chunk.length);
            if (last) {
                byte[] all = bin.toByteArray();
                bin.reset();
                recvBytes += all.length;
                try {
                    onVoice.accept(VoicePcm.toShorts(all, 0, all.length));
                } catch (RuntimeException e) {
                    log.accept("duck voice: playback error " + e);
                }
            }
            webSocket.request(1);
            return null;
        }

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            txt.append(data);
            if (last) {
                String s = txt.toString();
                txt.setLength(0);
                try {
                    onText.accept(s);
                } catch (RuntimeException e) {
                    log.accept("duck voice: control error " + e);
                }
            }
            webSocket.request(1);
            return null;
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            if (ws == webSocket) {
                ws = null;
            }
            log.accept("duck voice link closed (" + statusCode + ")");
            return null;
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            if (ws == webSocket) {
                ws = null;
            }
            log.accept("duck voice link error: " + error);
        }
    }
}
