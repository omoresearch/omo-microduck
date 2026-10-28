package dev.yifan.microduck;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * `!duck brains`: fetches GET /brains from the sidecar (community brains catalogue, see microduck-sim/brains.py)
 * and formats it as chat lines. Pure Java (java.net.http + Gson), so the formatter is unit-tested in the harness.
 */
public final class BrainsClient {
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1).connectTimeout(Duration.ofSeconds(2)).build();

    private BrainsClient() {
    }

    /** Async GET {base}/brains; each formatted line goes to {@code reply} (on an HttpClient thread). */
    public static void fetch(String base, Consumer<String> reply) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base + "/brains")).timeout(Duration.ofSeconds(3)).GET().build();
        HTTP.sendAsync(req, HttpResponse.BodyHandlers.ofString()).whenComplete((r, err) -> {
            if (err != null) {
                reply.accept("sidecar not reachable at " + base);
                return;
            }
            if (r.statusCode() != 200) {
                reply.accept("brains: HTTP " + r.statusCode() + " (sidecar too old?)");
                return;
            }
            for (String line : format(r.body())) {
                reply.accept(line);
            }
        });
    }

    /**
     * {"brain":"sprint","trick":null,"robot":"default","brains":[{"id":..,"kind":"perpetual"|"episodic",
     * "variant":"default"|"rollers","status":"pass"|"wobbly",..}]} -> 3 short chat lines.
     */
    public static List<String> format(String json) {
        List<String> out = new ArrayList<>();
        try {
            JsonObject o = JsonParser.parseString(json).getAsJsonObject();
            List<String> walk = new ArrayList<>();
            List<String> tricks = new ArrayList<>();
            JsonArray arr = o.has("brains") && o.get("brains").isJsonArray() ? o.getAsJsonArray("brains") : new JsonArray();
            for (JsonElement el : arr) {
                if (!el.isJsonObject()) {
                    continue;
                }
                JsonObject b = el.getAsJsonObject();
                String id = str(b, "id");
                if (id == null || !DuckCommand.BRAIN_ID.matcher(id).matches()) {
                    continue;                                  // never echo junk into chat
                }
                String tag = id + ("rollers".equals(str(b, "variant")) ? " (skates)" : "")
                        + ("wobbly".equals(str(b, "status")) ? "~" : "");
                ("episodic".equals(str(b, "kind")) ? tricks : walk).add(tag);
            }
            String cur = str(o, "brain");
            String trick = str(o, "trick");
            out.add("brain now: " + (cur == null ? "?" : cur) + (trick != null ? ", doing trick " + trick : "")
                    + "  (!duck brain <id>|default, !duck trick <id>; ~ = wobbly)");
            out.add("brains: default, " + (walk.isEmpty() ? "-" : String.join(", ", walk)));
            out.add("tricks: " + (tricks.isEmpty() ? "-" : String.join(", ", tricks)));
        } catch (Exception e) {
            out.clear();
            out.add("brains: bad reply from sidecar");
        }
        return out;
    }

    private static String str(JsonObject o, String k) {
        JsonElement e = o.get(k);
        return e == null || e.isJsonNull() || !e.isJsonPrimitive() ? null : e.getAsString();
    }
}
