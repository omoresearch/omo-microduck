package dev.yifan.microduck;

/**
 * Sim (MuJoCo, z-up, metres) <-> Minecraft frame mapping from CONTRACT.md. Pure Java, no MC classes.
 *
 * Axis change C: (x, y, z) -> (x, z, -y). Position: A + s * C p. Rotation: R_mc = C R_sim C^T, which for a
 * quaternion [w, v] is simply [w, C v] because C is a proper rotation (det +1).
 */
public final class DuckMath {
    private DuckMath() {
    }

    /** C applied to a vector. */
    public static double[] axis(double x, double y, double z) {
        return new double[] {x, z, -y};
    }

    /** MC position of sim point p: (ax + s*x, ay + s*z, az - s*y). */
    public static double[] simToMc(double ax, double ay, double az, double s, double x, double y, double z) {
        return new double[] {ax + s * x, ay + s * z, az - s * y};
    }

    /** Player MC (px, pz) -> sim (x, y): x = (px-ax)/s, y = -(pz-az)/s. */
    public static double[] mcToSim(double ax, double az, double s, double px, double pz) {
        return new double[] {(px - ax) / s, -(pz - az) / s};
    }

    /** Sim quaternion [w,x,y,z] -> MC quaternion [w,x,y,z] (= C R C^T). */
    public static double[] quatSimToMc(double w, double x, double y, double z) {
        return new double[] {w, x, z, -y};
    }

    /** Rotate v by unit quaternion q = [w,x,y,z]. */
    public static double[] rotate(double[] q, double vx, double vy, double vz) {
        double w = q[0], qx = q[1], qy = q[2], qz = q[3];
        // t = 2 * cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)
        double tx = 2 * (qy * vz - qz * vy);
        double ty = 2 * (qz * vx - qx * vz);
        double tz = 2 * (qx * vy - qy * vx);
        return new double[] {
            vx + w * tx + (qy * tz - qz * ty),
            vy + w * ty + (qz * tx - qx * tz),
            vz + w * tz + (qx * ty - qy * tx)};
    }

    /** 3x3 rotation matrix (row-major) of unit quaternion [w,x,y,z]. */
    public static double[][] matrix(double[] q) {
        double w = q[0], x = q[1], y = q[2], z = q[3];
        return new double[][] {
            {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    }

    /**
     * MC position of a body-local sim vertex v for a body at sim pose (pos p, quat q), exactly the way the renderer
     * composes it: translate(A + s*C p) * rotate(q_mc) * scale(s) applied to the pre-axis-changed vertex C v.
     */
    public static double[] bodyVertexToMc(double ax, double ay, double az, double s, float[] pose, double vx, double vy,
                                          double vz) {
        double[] origin = simToMc(ax, ay, az, s, pose[0], pose[1], pose[2]);
        double[] qmc = quatSimToMc(pose[3], pose[4], pose[5], pose[6]);
        double[] cv = axis(vx, vy, vz);
        double[] r = rotate(qmc, s * cv[0], s * cv[1], s * cv[2]);
        return new double[] {origin[0] + r[0], origin[1] + r[1], origin[2] + r[2]};
    }

    /** Interpolate two [px,py,pz,qw,qx,qy,qz] poses: lerp position, shortest-arc nlerp rotation. */
    public static float[] lerpPose(float[] a, float[] b, float t, float[] out) {
        if (out == null) {
            out = new float[7];
        }
        for (int i = 0; i < 3; i++) {
            out[i] = a[i] + (b[i] - a[i]) * t;
        }
        float dot = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6];
        float sign = dot < 0 ? -1f : 1f;
        float n = 0;
        for (int i = 3; i < 7; i++) {
            out[i] = a[i] + (sign * b[i] - a[i]) * t;
            n += out[i] * out[i];
        }
        n = (float) Math.sqrt(n);
        if (n < 1e-6f) {
            System.arraycopy(b, 3, out, 3, 4);
        } else {
            for (int i = 3; i < 7; i++) {
                out[i] /= n;
            }
        }
        return out;
    }
}
