package meaf;

import java.util.ArrayList;
import java.util.List;

/**
 * 2-Opt Local Search applied to a VRPTW individual.
 * Applied per sub-route; only accepts moves that:
 *   1) Reduce total route distance
 *   2) Do NOT violate time windows or capacity
 */
public class TwoOpt {

    /**
     * Apply 2-Opt to every sub-route of an individual.
     * Returns a (possibly improved) new Individual.
     */
    public static Individual apply(Individual ind, VRPTWInstance inst) {
        List<List<Integer>> routes = inst.decode(ind.route);
        List<List<Integer>> improved = new ArrayList<>();
        for (List<Integer> sub : routes) {
            improved.add(optimizeSub(sub, inst));
        }
        // Flatten back to permutation
        List<Integer> flat = new ArrayList<>();
        for (List<Integer> sub : improved) flat.addAll(sub);
        // Append any missing customer (safety)
        boolean[] seen = new boolean[inst.numCustomers + 1];
        for (int c : flat) seen[c] = true;
        for (int i = 1; i <= inst.numCustomers; i++) {
            if (!seen[i]) flat.add(i);
        }
        int[] perm = flat.stream().mapToInt(x -> x).toArray();
        return new Individual(perm, inst);
    }

    // ── Internal: optimize a single sub-route ───────────────────────────────
    private static List<Integer> optimizeSub(List<Integer> route,
                                              VRPTWInstance inst) {
        if (route.size() < 3) return new ArrayList<>(route);
        List<Integer> best = new ArrayList<>(route);
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < best.size() - 1; i++) {
                for (int j = i + 2; j < best.size(); j++) {
                    int ci  = best.get(i);
                    int ci1 = best.get(i + 1);
                    int cj  = best.get(j);
                    int cj1 = (j + 1 < best.size()) ? best.get(j + 1) : 0;

                    double oldDist = inst.dist[ci][ci1] + inst.dist[cj][cj1];
                    double newDist = inst.dist[ci][cj]  + inst.dist[ci1][cj1];

                    if (newDist < oldDist - 1e-6) {
                        // Try the reversal
                        List<Integer> candidate = new ArrayList<>(best);
                        // Reverse segment [i+1 .. j]
                        reverseSegment(candidate, i + 1, j);
                        if (inst.isFeasible(candidate)) {
                            best     = candidate;
                            improved = true;
                        }
                    }
                }
            }
        }
        return best;
    }

    private static void reverseSegment(List<Integer> list, int lo, int hi) {
        while (lo < hi) {
            int tmp = list.get(lo);
            list.set(lo, list.get(hi));
            list.set(hi, tmp);
            lo++; hi--;
        }
    }
}