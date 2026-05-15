package meaf;

import java.util.*;

public class ACSAgent extends BaseAgent {

    private static final int    COLONY = 100;
    private static final double RHO    = 0.30;
    private static final double PHI    = 0.10;
    private static final double Q0     = 0.10;
    private static final double BETA   = 2.0;
    private static final double ALPHA  = 1.0;

    private double[][] tau;
    private double[][] eta;
    private double     tau0;

    private List<List<Integer>> bestRoutes;
    private double              bestRouteCost;

    private final int    N;
    private final Random rand;

    public ACSAgent(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        super(inst);
        this.rand = rand;
        this.N    = inst.numCustomers;

        int sz = N + 1;
        eta = new double[sz][sz];
        for (int i = 0; i < sz; i++) {
            for (int j = 0; j < sz; j++) {
                eta[i][j] = (inst.dist[i][j] > 1e-9) ? 1.0 / inst.dist[i][j] : 0.0;
            }
        }

        // CHỈ dùng sharedPop để tính L_nn, KHÔNG lưu bestRoutes hay gọi globalUpdate
        double bestCost = Double.MAX_VALUE;
        for (Individual ind : sharedPop) {
            List<List<Integer>> routes = inst.decode(ind.route);
            double cost = inst.evalRoutes(routes);
            if (cost < bestCost) { bestCost = cost; }
        }
        double Lnn = (bestCost > 0) ? bestCost : 1.0;
        tau0 = 1.0 / (N * Lnn);
        tau  = new double[sz][sz];
        for (double[] row : tau) Arrays.fill(row, tau0);

        bestRoutes = null;         // <-- không có lời giải ban đầu
        bestRouteCost = Double.MAX_VALUE;
        bestInd = null;
        recordHistory();
    }

    @Override
    public void step() {
        if (!active) return;

        List<List<Integer>> iterBest = null;
        double iterBestCost = Double.MAX_VALUE;

        for (int ant = 0; ant < COLONY; ant++) {
            List<List<Integer>> sol = buildSolution();
            double cost = inst.evalRoutes(sol);
            if (cost < iterBestCost) { iterBestCost = cost; iterBest = sol; }
        }

        if (iterBestCost < bestRouteCost) {
            bestRouteCost = iterBestCost;
            bestRoutes    = iterBest;
            bestInd       = routesToIndividual(bestRoutes);
            stableCount   = 0;
        } else {
            stableCount++;
        }

        globalUpdate();
        recordHistory();
    }

    // ── Communication Input (Fusion strategy for ACS, Section 5.2) ──────────
    @Override
    public void injectGlobalBest(List<Individual> pool) {
        if (!active || pool == null || pool.isEmpty()) return;

        for (int i = 0; i < Math.min(pool.size(), MEAF_K); i++) {
            Individual ext    = pool.get(i);
            List<List<Integer>> extRoutes = inst.decode(ext.route);
            double extCost    = inst.evalRoutes(extRoutes);

            if (extCost < bestRouteCost) {
                bestRouteCost = extCost;
                bestRoutes    = extRoutes;
                bestInd       = ext.clone();
            }
            // Reinforce pheromone on external route (as if it were global‑best)
            double delta = (extCost > 0) ? 1.0 / extCost : 0.0;
            for (List<Integer> sub : extRoutes) {
                int prev = 0;
                for (int c : sub) {
                    tau[prev][c] += RHO * delta;
                    prev = c;
                }
                tau[prev][0] += RHO * delta;
            }
        }
    }

    // ── Dissolve Rule (re‑seed pheromone from donor) ────────────────────────
    @Override
    public void dissolveFrom(BaseAgent donor) {
        if (!active) return;
        Individual donorInd  = donor.getBestInd();
        List<List<Integer>> donorRoutes = inst.decode(donorInd.route);
        double donorCost     = inst.evalRoutes(donorRoutes);

        // Re‑seed pheromone matrix based on donor best
        this.tau0 = 1.0 / (N * donorCost);
        for (double[] row : tau) Arrays.fill(row, tau0);

        if (donorCost < bestRouteCost) {
            bestRouteCost = donorCost;
            bestRoutes    = donorRoutes;
            bestInd       = donorInd.clone();
        }

        // Strongly reinforce donor's route
        double delta = (donorCost > 0) ? 5.0 / donorCost : 0.0;
        for (List<Integer> sub : donorRoutes) {
            int prev = 0;
            for (int c : sub) {
                tau[prev][c] += RHO * delta;
                prev = c;
            }
            tau[prev][0] += RHO * delta;
        }
        System.out.printf("    [ACS] Dissolved: re‑seeded pheromone from %s (cost=%.2f)%n",
                          donor.getClass().getSimpleName(), donorCost);
    }

    // ── Solution construction (theo đúng ACS) ───────────────────────────────
    private List<List<Integer>> buildSolution() {
        Set<Integer> unvisited = new LinkedHashSet<>();
        for (int i = 1; i <= N; i++) unvisited.add(i);

        List<List<Integer>> routes = new ArrayList<>();
        while (!unvisited.isEmpty()) {
            List<Integer> sub = new ArrayList<>();
            int cur = 0;
            while (true) {
                List<Integer> cands = new ArrayList<>();
                for (int c : unvisited) {
                    if (inst.canAppend(sub, c)) cands.add(c);
                }
                if (cands.isEmpty()) break;

                int next;
                double q = rand.nextDouble();
                if (q <= Q0) {
                    // Exploitation: argmax
                    next = cands.get(0);
                    double bestVal = Double.NEGATIVE_INFINITY;
                    for (int c : cands) {
                        double val = Math.pow(tau[cur][c], ALPHA)
                                   * Math.pow(eta[cur][c], BETA);
                        if (val > bestVal) { bestVal = val; next = c; }
                    }
                } else {
                    // Exploration: roulette wheel
                    double[] ws = new double[cands.size()];
                    double   sum = 0.0;
                    for (int i = 0; i < cands.size(); i++) {
                        int c = cands.get(i);
                        ws[i] = Math.pow(tau[cur][c], ALPHA)
                              * Math.pow(eta[cur][c], BETA);
                        sum  += ws[i];
                    }
                    double r = rand.nextDouble() * sum;
                    double cum = 0.0;
                    next = cands.get(cands.size() - 1);
                    for (int i = 0; i < cands.size(); i++) {
                        cum += ws[i];
                        if (r <= cum) { next = cands.get(i); break; }
                    }
                }

                sub.add(next);
                unvisited.remove(next);

                // Local pheromone update (Eq 16)
                tau[cur][next] = (1 - PHI) * tau[cur][next] + PHI * tau0;
                cur = next;
            }
            if (!sub.isEmpty()) routes.add(sub);
        }
        return routes;
    }

    // ── Global pheromone update (Eq 15) ─────────────────────────────────────
    private void globalUpdate() {
        if (bestRoutes == null) return;
        double delta = (bestRouteCost > 0) ? 1.0 / bestRouteCost : 0.0;
        // Evaporate all
        for (int i = 0; i < tau.length; i++) {
            for (int j = 0; j < tau[i].length; j++) {
                tau[i][j] *= (1 - RHO);
            }
        }
        // Reinforce global‑best route
        for (List<Integer> sub : bestRoutes) {
            int prev = 0;
            for (int c : sub) {
                tau[prev][c] += RHO * delta;
                prev = c;
            }
            tau[prev][0] += RHO * delta;
        }
    }

    // ── Utilities ────────────────────────────────────────────────────────────
    private Individual routesToIndividual(List<List<Integer>> routes) {
        List<Integer> flat = new ArrayList<>();
        for (List<Integer> sub : routes) flat.addAll(sub);
        // Append missing (safety)
        boolean[] seen = new boolean[N + 1];
        for (int c : flat) seen[c] = true;
        for (int i = 1; i <= N; i++) if (!seen[i]) flat.add(i);
        int[] perm = flat.stream().mapToInt(x -> x).toArray();
        return new Individual(perm, inst.evalRoutes(routes));
    }

    private void recordHistory() {
        if (bestRoutes != null) {
            history.add(inst.pureDistance(bestRoutes));
        } else {
            history.add(Double.MAX_VALUE);
        }
    }
}