package dev.yifan.microduck;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;

/**
 * Pure-Java PCM helpers for the duck's voice (Simple Voice Chat format: 48 kHz, mono, 16-bit, 20 ms = 960 samples).
 * <ul>
 *   <li>{@link #toBytes}/{@link #toShorts}: PCM16 little-endian, the wire format of the voice backend link.</li>
 *   <li>{@link Frames}: re-frames arbitrary-length chunks into exact 960-sample frames, with a small prebuffer
 *       (jitter cushion) before playback starts and a cap so a stall can't build up seconds of lag.</li>
 *   <li>{@link #resample}: linear resampler for other input rates (the backend already sends 48 kHz).</li>
 * </ul>
 */
public final class VoicePcm {
    public static final int RATE = 48000;
    public static final int FRAME = 960;

    private VoicePcm() {
    }

    public static byte[] toBytes(short[] pcm) {
        ByteBuffer b = ByteBuffer.allocate(pcm.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short s : pcm) {
            b.putShort(s);
        }
        return b.array();
    }

    public static short[] toShorts(byte[] le, int off, int len) {
        short[] out = new short[len / 2];
        ByteBuffer b = ByteBuffer.wrap(le, off, len - (len & 1)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getShort();
        }
        return out;
    }

    public static double rms(short[] pcm) {
        if (pcm.length == 0) {
            return 0;
        }
        double s = 0;
        for (short v : pcm) {
            s += (double) v * v;
        }
        return Math.sqrt(s / pcm.length);
    }

    /** Linear resample mono PCM16 from {@code from} Hz to {@code to} Hz. */
    public static short[] resample(short[] in, int from, int to) {
        if (from == to || in.length == 0) {
            return in.clone();
        }
        int n = (int) Math.max(1, Math.round((double) in.length * to / from));
        short[] out = new short[n];
        double step = (double) from / to;
        for (int i = 0; i < n; i++) {
            double x = i * step;
            int a = (int) x;
            int b = Math.min(a + 1, in.length - 1);
            double t = x - a;
            out[i] = (short) Math.round(in[Math.min(a, in.length - 1)] * (1 - t) + in[b] * t);
        }
        return out;
    }

    /** Thread-safe FIFO handing out exact {@link #FRAME}-sample frames. */
    public static final class Frames {
        private final ArrayDeque<short[]> frames = new ArrayDeque<>();
        private final short[] partial = new short[FRAME];
        private int partialLen;
        private final int prebuffer;
        private final int maxFrames;
        private boolean playing;
        private long dropped;

        /** @param prebuffer frames to collect before {@link #poll} starts handing them out; @param maxFrames cap (oldest dropped). */
        public Frames(int prebuffer, int maxFrames) {
            this.prebuffer = Math.max(1, prebuffer);
            this.maxFrames = Math.max(this.prebuffer, maxFrames);
        }

        public synchronized void push(short[] pcm) {
            int i = 0;
            while (i < pcm.length) {
                int n = Math.min(FRAME - partialLen, pcm.length - i);
                System.arraycopy(pcm, i, partial, partialLen, n);
                partialLen += n;
                i += n;
                if (partialLen == FRAME) {
                    frames.add(partial.clone());
                    partialLen = 0;
                    while (frames.size() > maxFrames) {
                        frames.poll();
                        dropped++;
                    }
                }
            }
        }

        /** Pads the partial tail frame with silence so the end of an utterance is not lost. */
        public synchronized void flushTail() {
            if (partialLen > 0) {
                short[] f = new short[FRAME];
                System.arraycopy(partial, 0, f, 0, partialLen);
                frames.add(f);
                partialLen = 0;
                playing = true; // a short utterance below the prebuffer still plays
            }
        }

        /** Next frame, or null (nothing buffered / still prebuffering). */
        public synchronized short[] poll() {
            if (!playing) {
                if (frames.size() < prebuffer) {
                    return null;
                }
                playing = true;
            }
            short[] f = frames.poll();
            if (f == null) {
                playing = false; // underrun: prebuffer again
            }
            return f;
        }

        public synchronized void clear() {
            frames.clear();
            partialLen = 0;
            playing = false;
        }

        public synchronized int size() {
            return frames.size();
        }

        public synchronized boolean isPlaying() {
            return playing;
        }

        public synchronized long dropped() {
            return dropped;
        }
    }
}
