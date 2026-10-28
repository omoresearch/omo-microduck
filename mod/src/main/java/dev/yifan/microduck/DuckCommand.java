package dev.yifan.microduck;

import java.util.Locale;
import java.util.Set;

/**
 * Parser for `!duck ...` / `!ducks ...` chat lines. Pure Java.
 * <ul>
 *   <li>{@code !duck <cmd>}: the duck nearest to the player (or the only one)</li>
 *   <li>{@code !duck <name> <cmd>}: that duck (case-insensitive; an unambiguous prefix works; the sidecar resolves it)</li>
 *   <li>{@code !ducks <cmd>}: every duck</li>
 *   <li>{@code !duck spawn [persona|all]}, {@code !duck remove <name|all>}, {@code !duck list}</li>
 * </ul>
 * {@link #json} never contains the target: the caller adds it with {@link #withDuck} once it knows the nearest duck.
 */
public final class DuckCommand {

    public enum Kind {
        /** Not a !duck message: let it through to the server. */
        NOT_DUCK,
        HELP,
        /** Forward to the sidecar: {@link #json} is the POST /cmd body, {@link #duck} the target selector. */
        SIDECAR,
        HERE,
        SCALE,
        /** Name tags on/off: {@link #scale} 1 = on, 0 = off. */
        NAMES,
        /** List the community brains (GET /brains on the sidecar). */
        BRAINS,
        /** Spawn a duck: {@link #duck} = persona (null = next unused, "all" = every unused persona). */
        SPAWN,
        /** Remove a duck: {@link #duck} = name/id or "all". */
        REMOVE,
        /** List the ducks (GET /ducks). */
        LIST,
        /** Bad usage; {@link #message} explains. */
        ERROR
    }

    public static final double MIN_SCALE = 0.5;
    public static final double MAX_SCALE = 10.0;
    /** Community brain / trick ids; the sidecar validates that the id exists. */
    public static final java.util.regex.Pattern BRAIN_ID = java.util.regex.Pattern.compile("[a-z0-9-]{1,24}");
    /** Duck names / persona ids as typed in chat. */
    public static final java.util.regex.Pattern NAME = java.util.regex.Pattern.compile("[a-z][a-z0-9_-]{0,19}");
    /** Words that are commands, so they can't be duck names. */
    public static final Set<String> WORDS = Set.of("walk", "stop", "sit", "stand", "follow", "quack", "reset", "left",
            "right", "here", "brains", "brain", "trick", "scale", "names", "spawn", "remove", "list", "help", "?", "all");

    public static final String HELP = "!duck walk|stop|sit|stand|follow|quack|left|right|reset (nearest duck), "
            + "!duck <name> <cmd> (one duck, e.g. !duck zippy trick roulade), !ducks <cmd> (every duck), "
            + "!duck spawn [persona|all], !duck remove <name|all>, !duck list, "
            + "!duck here (anchor at your feet), !duck scale <0.5-10>, !duck names on|off, "
            + "!duck brains (list), !duck brain <id>|default, !duck trick <id>";

    public final Kind kind;
    public final String cmd;
    public final String json;
    public final double scale;
    public final String message;
    /** Target: null = nearest/only duck, "all" = every duck, else a name/id as typed (lower case). */
    public final String duck;

    private DuckCommand(Kind kind, String cmd, String json, double scale, String message, String duck) {
        this.kind = kind;
        this.cmd = cmd;
        this.json = json;
        this.scale = scale;
        this.message = message;
        this.duck = duck;
    }

    /** True if the chat line is addressed to the duck(s) (so it must not be sent to the server). */
    public static boolean isDuck(String line) {
        if (line == null) {
            return false;
        }
        String t = line.trim();
        for (String p : new String[] {"!ducks", "!duck"}) {
            if (t.regionMatches(true, 0, p, 0, p.length())
                    && (t.length() == p.length() || Character.isWhitespace(t.charAt(p.length())))) {
                return true;
            }
        }
        return false;
    }

    public static DuckCommand parse(String line) {
        if (!isDuck(line)) {
            return new DuckCommand(Kind.NOT_DUCK, null, null, 0, null, null);
        }
        String t = line.trim();
        boolean all = t.regionMatches(true, 0, "!ducks", 0, 6) && (t.length() == 6 || Character.isWhitespace(t.charAt(6)));
        String rest = t.substring(all ? 6 : 5).trim().toLowerCase(Locale.ROOT);
        String[] a = rest.isEmpty() ? new String[0] : rest.split("\\s+");
        if (all) {
            if (a.length == 0) {
                return error("usage: !ducks <cmd>  (every duck: walk|stop|sit|stand|follow|quack|left|right|reset|brain <id>|trick <id>)");
            }
            return command(a, 0, "all");
        }
        if (a.length == 0 || a[0].equals("help") || a[0].equals("?")) {
            return new DuckCommand(Kind.HELP, "help", null, 0, HELP, null);
        }
        switch (a[0]) {
            case "spawn":
                if (a.length > 2) {
                    return error("usage: !duck spawn [persona|all]");
                }
                if (a.length == 2 && !NAME.matcher(a[1]).matches()) {
                    return error("not a persona name: " + a[1]);
                }
                return new DuckCommand(Kind.SPAWN, "spawn", null, 0, null, a.length == 2 ? a[1] : null);
            case "remove":
                if (a.length != 2 || !NAME.matcher(a[1]).matches()) {
                    return error("usage: !duck remove <name|all>");
                }
                return new DuckCommand(Kind.REMOVE, "remove", null, 0, null, a[1]);
            case "list":
                return a.length == 1 ? new DuckCommand(Kind.LIST, "list", null, 0, null, null) : error("usage: !duck list");
            case "here":
                return new DuckCommand(Kind.HERE, "here", null, 0, null, null);
            case "brains":
                return new DuckCommand(Kind.BRAINS, "brains", null, 0, null, null);
            case "names":
                if (a.length != 2 || !(a[1].equals("on") || a[1].equals("off"))) {
                    return error("usage: !duck names on|off");
                }
                return new DuckCommand(Kind.NAMES, "names", null, a[1].equals("on") ? 1 : 0, null, null);
            case "scale":
                if (a.length < 2) {
                    return error("usage: !duck scale <" + MIN_SCALE + "-" + MAX_SCALE + ">");
                }
                double s;
                try {
                    s = Double.parseDouble(a[1]);
                } catch (NumberFormatException e) {
                    return error("not a number: " + a[1]);
                }
                if (!(s >= MIN_SCALE && s <= MAX_SCALE)) {
                    return error("scale must be between " + MIN_SCALE + " and " + MAX_SCALE);
                }
                return new DuckCommand(Kind.SCALE, "scale", null, s, null, null);
            default:
                break;
        }
        if (WORDS.contains(a[0])) {
            return command(a, 0, null);
        }
        // !duck <name> <cmd...>
        if (!NAME.matcher(a[0]).matches()) {
            return error("unknown command '" + a[0] + "'. " + HELP);
        }
        if (a.length == 1) {
            return error("unknown command '" + a[0] + "' (or a duck name without a command: !duck " + a[0] + " walk). " + HELP);
        }
        if (Set.of("here", "scale", "names", "spawn", "remove", "list", "brains", "help").contains(a[1])) {
            return error("'" + a[1] + "' is for the whole lab, not one duck: !duck " + a[1]);
        }
        return command(a, 1, a[0]);
    }

    /** a[i..] is a sidecar command for target {@code duck}. */
    private static DuckCommand command(String[] a, int i, String duck) {
        String c = a[i];
        int n = a.length - i;
        switch (c) {
            case "walk":
            case "stop":
            case "sit":
            case "stand":
            case "follow":
            case "quack":
            case "reset":
                if (c.equals("follow") && n == 2 && !a[i + 1].equals("me")) {
                    String tgt = a[i + 1];                // !duck rusty follow zippy: follow another duck
                    if (!NAME.matcher(tgt).matches()) {
                        return error("usage: !duck <name> follow [me|<other duck>]");
                    }
                    return sidecar("follow " + tgt, "{\"cmd\":\"follow\",\"target\":\"" + tgt + "\",\"from\":\"mc\"}", duck);
                }
                if (n > (c.equals("follow") ? 2 : 1)) {
                    return error("'" + c + "' takes no arguments");
                }
                return sidecar(c, "{\"cmd\":\"" + c + "\",\"from\":\"mc\"}", duck);
            case "left":
            case "right":
                if (n > 1) {
                    return error("'" + c + "' takes no arguments");
                }
                return sidecar("turn " + c, "{\"cmd\":\"turn\",\"dir\":\"" + c + "\",\"from\":\"mc\"}", duck);
            case "brain":
            case "trick":
                if (n != 2 || !BRAIN_ID.matcher(a[i + 1]).matches()) {
                    return error("usage: !duck " + (duck == null ? "" : duck.equals("all") ? "s " : duck + " ") + c
                            + " <id>  (ids: !duck brains)");
                }
                return sidecar(c + " " + a[i + 1], "{\"cmd\":\"" + c + "\",\"name\":\"" + a[i + 1] + "\",\"from\":\"mc\"}", duck);
            default:
                return error("unknown command '" + c + "'. " + HELP);
        }
    }

    /** Chat line for a POST /ducks reply (spawn / spawn all). */
    public static String spawnReply(com.google.gson.JsonObject o) {
        StringBuilder b = new StringBuilder();
        if (o.has("spawned") && o.get("spawned").isJsonArray() && o.getAsJsonArray("spawned").size() > 0) {
            b.append("spawned ");
            int i = 0;
            for (com.google.gson.JsonElement e : o.getAsJsonArray("spawned")) {
                com.google.gson.JsonObject d = e.getAsJsonObject();
                b.append(i++ > 0 ? ", " : "").append(str(d, "name")).append(" (").append(str(d, "voice")).append(")");
            }
            if (o.has("ducks")) {
                b.append("; ").append(o.get("ducks").getAsInt()).append(" duck(s) now");
            }
            if (o.has("errors") && o.get("errors").isJsonArray() && o.getAsJsonArray("errors").size() > 0) {
                b.append(" (").append(o.getAsJsonArray("errors").get(0).getAsString()).append(")");
            }
            return b.toString();
        }
        return "spawn: " + (o.has("error") ? str(o, "error") : "failed");
    }

    /** Chat line for a POST /ducks/remove reply. */
    public static String removeReply(com.google.gson.JsonObject o) {
        if (o.has("ok") && o.get("ok").getAsBoolean() && o.has("removed")) {
            StringBuilder b = new StringBuilder("removed ");
            int i = 0;
            for (com.google.gson.JsonElement e : o.getAsJsonArray("removed")) {
                b.append(i++ > 0 ? ", " : "").append(str(e.getAsJsonObject(), "name"));
            }
            return b.append("; ").append(o.has("ducks") ? o.get("ducks").getAsInt() : 0).append(" duck(s) left").toString();
        }
        return "remove: " + (o.has("error") ? str(o, "error") : "failed");
    }

    private static String str(com.google.gson.JsonObject o, String k) {
        return o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsString() : "?";
    }

    /** {@code json} with {@code "duck":"<sel>"} added (sel null or empty: unchanged). */
    public static String withDuck(String json, String sel) {
        if (sel == null || sel.isEmpty() || json == null || !json.startsWith("{")) {
            return json;
        }
        StringBuilder b = new StringBuilder("{\"duck\":\"");
        for (char ch : sel.toCharArray()) {
            if (ch == '"' || ch == '\\') {
                b.append('\\');
            }
            if (ch >= 0x20) {
                b.append(ch);
            }
        }
        b.append("\",");
        return b.append(json, 1, json.length()).toString();
    }

    private static DuckCommand sidecar(String cmd, String json, String duck) {
        return new DuckCommand(Kind.SIDECAR, cmd, json, 0, null, duck);
    }

    private static DuckCommand error(String msg) {
        return new DuckCommand(Kind.ERROR, null, null, 0, msg, null);
    }
}
