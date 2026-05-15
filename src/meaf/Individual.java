package meaf;

import java.util.Arrays;
import java.util.Random;

/**
 * Represents a candidate solution as a flat permutation of customer IDs 1..N.
 * Shared across GA, PSO, and used for pheromone seeding in ACS.
 *
 * fitness = evalPerm(route)  [lower = better internally]
 * We keep cost-based fitness (lower = better) to match ACS convention.
 */
public class Individual implements Cloneable {
    public int[]  route;      // permutation of customer IDs [1..N]
    public double cost;       // weighted cost (with penalty) – lower is better
    private static final Random rng = new Random();

    public Individual(int[] route, double cost) {
        this.route = route.clone();
        this.cost  = cost;
    }

    public Individual(int[] route, VRPTWInstance inst) {
        this.route = route.clone();
        this.cost  = inst.evalPerm(this.route);
    }

    /** Deep copy */
    @Override
    public Individual clone() {
        return new Individual(this.route.clone(), this.cost);
    }

    // ── PMX Crossover ────────────────────────────────────────────────────────
    /**
     * Partially Mapped Crossover (PMX).
     * Returns two offspring from parents p1 and p2.
     */
    public static Individual[] pmxCrossover(Individual p1, Individual p2,
                                             VRPTWInstance inst, Random rand) {
        int n = p1.route.length;
        int a = rand.nextInt(n);
        int b = rand.nextInt(n);
        if (a > b) { int tmp = a; a = b; b = tmp; }

        int[] c1 = pmxChild(p1.route, p2.route, a, b);
        int[] c2 = pmxChild(p2.route, p1.route, a, b);
        return new Individual[]{
            new Individual(c1, inst),
            new Individual(c2, inst)
        };
    }

    private static int[] pmxChild(int[] p1, int[] p2, int lo, int hi) {
        int n = p1.length;
        int[] child = new int[n];
        Arrays.fill(child, -1);
        // Copy segment from p1
        for (int i = lo; i <= hi; i++) child[i] = p1[i];
        // Fill rest from p2 using PMX mapping
        for (int i = lo; i <= hi; i++) {
            int val = p2[i];
            if (!inRange(child, val, lo, hi)) {
                // Find where val goes
                int pos = i;
                while (pos >= lo && pos <= hi) {
                    int mapped = p1[pos];
                    pos = indexOf(p2, mapped);
                }
                child[pos] = val;
            }
        }
        // Fill remaining positions with p2 order
        for (int i = 0; i < n; i++) {
            if (child[i] == -1) child[i] = p2[i];
        }
        return child;
    }

    private static boolean inRange(int[] arr, int val, int lo, int hi) {
        for (int i = lo; i <= hi; i++) if (arr[i] == val) return true;
        return false;
    }

    private static int indexOf(int[] arr, int val) {
        for (int i = 0; i < arr.length; i++) if (arr[i] == val) return i;
        return -1;
    }

    // ── Reverse Sequence Mutation (RSM) ──────────────────────────────────────
    /**
     * Reverse Sequence Mutation: choose two positions i < j, reverse arr[i..j].
     */
    public Individual mutate(VRPTWInstance inst, Random rand) {
        int n = route.length;
        int a = rand.nextInt(n);
        int b = rand.nextInt(n);
        if (a > b) { int tmp = a; a = b; b = tmp; }
        int[] newRoute = route.clone();
        // Reverse segment [a..b]
        while (a < b) {
            int tmp = newRoute[a]; newRoute[a] = newRoute[b]; newRoute[b] = tmp;
            a++; b--;
        }
        return new Individual(newRoute, inst);
    }

    // ── Swap (for PSO) ───────────────────────────────────────────────────────
    public Individual swap(int i, int j, VRPTWInstance inst) {
        int[] newRoute = route.clone();
        int tmp = newRoute[i]; newRoute[i] = newRoute[j]; newRoute[j] = tmp;
        return new Individual(newRoute, inst);
    }

    @Override
    public String toString() {
        return String.format("Individual[cost=%.2f]", cost);
    }
}