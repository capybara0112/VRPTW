package meaf;

import java.util.*;
import java.util.stream.Collectors;

/**
 * MEAF Orchestrator – implements Algorithm 1 (MEAF loop) and
 * Algorithm 2 (Dissolve Rule) from the paper.
 *
 * Architecture (Fig. 3, paper):
 *   ┌────────────────────────────────────────────┐
 *   │           SKIN MEMBRANE (Global Pool)       │
 *   ├─────────────┬──────────────┬───────────────┤
 *   │  GA_Agent   │  PSO_Agent   │  ACS_Agent    │
 *   │  (+2-Opt)   │  (Discrete)  │ (Pheromone)   │
 *   └─────────────┴──────────────┴───────────────┘
 *
 * Every M=15 iterations:
 *   CommOut → GlobalPool → CommIn (replace K worst in each agent)
 *
 * After sditer=30 iterations:
 *   Dissolve: worst agent absorbs 50% best solutions from best agent
 */
public class Orchestrator {

    // ── MEAF parameters ───────────────────────────────────────────────────────
    public static final int    N_GEN          = 500;    // demo (paper: 500)
    public static final int    N_RUNS         = 21;     // demo (paper: 21)
    public static final int    MEAF_M         = 15;
    public static final int    MEAF_SD_ITER   = 50;    // demo value
    public static final int    MEAF_THRESHOLD = 2;
    public static final double MEAF_DF        = 0.85;
    public static final int    MEAF_K         = 1;

    private final VRPTWInstance inst;
    private final Random        rand;

    // Agents
    private GAAgent  ga;
    private PSOAgent pso;
    private ACSAgent acs;

    // Global pool (skin membrane)
    private final List<Individual> globalPool = new ArrayList<>();

    // Convergence history  [0..N_GEN]
    private final List<Double> meafHistory = new ArrayList<>();

    // ── Constructor ──────────────────────────────────────────────────────────
    public Orchestrator(VRPTWInstance inst, Random rand) {
        this.inst = inst;
        this.rand = rand;
    }

    // ── Main run loop ─────────────────────────────────────────────────────────
    /**
     * Run the full MEAF framework for N_GEN iterations.
     * Returns the global-best Individual found.
     */
    public RunResult run(List<Individual> sharedPop) {
        System.out.println("  ┌─────────────────────────────────────────────────┐");
        System.out.printf ("  │  MEAF: %s | KH=%d | Pop=100 | Gens=%d%n",
                           inst.name, inst.numCustomers, N_GEN);
        System.out.printf ("  │  M=%d | sditer=%d | dfpercent=%.2f%n",
                           MEAF_M, MEAF_SD_ITER, MEAF_DF);
        System.out.println("  └─────────────────────────────────────────────────┘");

        // Initialize agents with shared population
        ga  = new GAAgent (inst, sharedPop, new Random(rand.nextLong()));
        pso = new PSOAgent(inst, sharedPop, new Random(rand.nextLong()));
        acs = new ACSAgent(inst, sharedPop, new Random(rand.nextLong()));

        // Record G=0
        meafHistory.add(getGlobalBestPureDist());

        for (int gen = 1; gen <= N_GEN; gen++) {

            // ── 1. Isolated Evolution ─────────────────────────────────────
            if (ga.isActive())  ga.step();
            if (pso.isActive()) pso.step();
            if (acs.isActive()) acs.step();

            // ── 2. Communication Phase (every M iterations) ──────────────
            if (gen % MEAF_M == 0) {
                communicationPhase(gen);
            }

            // ── 3. Dissolve Rule ──────────────────────────────────────────
            dissolveCheck(gen);

            // ── 4. Record global best ──────────────────────────────────────
            double gBest = getGlobalBestPureDist();
            meafHistory.add(gBest);

            if (gen % 10 == 0 || gen == N_GEN) {
                printStatus(gen, gBest);
            }
        }

        Individual best = collectGlobalBest();
        System.out.printf("  [MEAF] Done. Best pure distance = %.2f%n",
                          inst.pureDistancePerm(best.route));
        return new RunResult(best, meafHistory);
    }

    // ── Communication Phase ───────────────────────────────────────────────────
    /**
     * CommOut: each active agent contributes its best to the global pool.
     * CommIn:  each active agent receives top-K from pool → replaces K worst.
     */
    private void communicationPhase(int gen) {
        // CommOut: collect best from each active agent
        globalPool.clear();
        Map<String, Double> agentCosts = new LinkedHashMap<>();

        if (ga.isActive()) {
            globalPool.add(ga.getBestInd().clone());
            agentCosts.put("GA",  ga.getBestPureDistance());
        }
        if (pso.isActive()) {
            globalPool.add(pso.getBestInd().clone());
            agentCosts.put("PSO", pso.getBestPureDistance());
        }
        if (acs.isActive()) {
            globalPool.add(acs.getBestInd().clone());
            agentCosts.put("ACS", acs.getBestPureDistance());
        }

        // Sort pool: best (lowest pure dist) first
        globalPool.sort(Comparator.comparingDouble(
            ind -> inst.pureDistancePerm(ind.route)));

        // CommIn: inject global best into every active agent
        if (!globalPool.isEmpty()) {
            List<Individual> topK = globalPool.subList(0, Math.min(MEAF_K, globalPool.size()));
            if (ga.isActive())  ga.injectGlobalBest(topK);
            if (pso.isActive()) pso.injectGlobalBest(topK);
            if (acs.isActive()) acs.injectGlobalBest(topK);
        }

        double globalBest = globalPool.isEmpty() ? Double.MAX_VALUE
                : inst.pureDistancePerm(globalPool.get(0).route);
        System.out.printf("  [MEAF] Gen %3d | Comm | GlobalBest=%.2f | %s%n",
            gen, globalBest,
            agentCosts.entrySet().stream()
                .map(e -> e.getKey() + "=" + String.format("%.1f", e.getValue()))
                .collect(Collectors.joining(" | ")));
    }

    // ── Dissolve Rule (Algorithm 2) ──────────────────────────────────────────
    private void dissolveCheck(int gen) {
        if (gen < MEAF_SD_ITER) return;

        // Count active agents
        List<BaseAgent> activeAgents = getActiveAgents();
        if (activeAgents.size() <= MEAF_THRESHOLD) return;

        // Find best and worst by pure distance
        BaseAgent bestAgent  = null, worstAgent = null;
        double bestCost  = Double.MAX_VALUE, worstCost = Double.MIN_VALUE;
        for (BaseAgent ag : activeAgents) {
            double c = ag.getBestPureDistance();
            if (c < bestCost)  { bestCost  = c; bestAgent  = ag; }
            if (c > worstCost) { worstCost = c; worstAgent = ag; }
        }

        if (worstCost <= 0 || bestAgent == worstAgent) return;

        double ratio = bestCost / worstCost;
        if (ratio < MEAF_DF) {
            String worstName = worstAgent.getClass().getSimpleName();
            String bestName  = bestAgent.getClass().getSimpleName();
            System.out.printf("  [MEAF] Gen %d: Dissolve %s (ratio=%.3f < %.2f) ← inject from %s%n",
                              gen, worstName, ratio, MEAF_DF, bestName);
            worstAgent.dissolveFrom(bestAgent);
        }
    }

    // ── Global best helpers ───────────────────────────────────────────────────
    private double getGlobalBestPureDist() {
        double best = Double.MAX_VALUE;
        for (BaseAgent ag : getAllAgents()) {
            if (ag.isActive()) {
                double d = ag.getBestPureDistance();
                if (d < best) best = d;
            }
        }
        return best;
    }

    private Individual collectGlobalBest() {
        Individual best = null;
        double bestDist = Double.MAX_VALUE;
        for (BaseAgent ag : getAllAgents()) {
            if (ag.isActive() && ag.getBestInd() != null) {
                double d = inst.pureDistancePerm(ag.getBestInd().route);
                if (d < bestDist) { bestDist = d; best = ag.getBestInd(); }
            }
        }
        return best != null ? best : ga.getBestInd();
    }

    private List<BaseAgent> getActiveAgents() {
        List<BaseAgent> list = new ArrayList<>();
        for (BaseAgent ag : getAllAgents()) if (ag.isActive()) list.add(ag);
        return list;
    }

    private List<BaseAgent> getAllAgents() {
        return Arrays.asList(ga, pso, acs);
    }

    private void printStatus(int gen, double gBest) {
        List<String> active = getActiveAgents().stream()
            .map(a -> a.getClass().getSimpleName().replace("Agent",""))
            .collect(Collectors.toList());
        System.out.printf("  Gen %3d | MEAF Best=%.2f | Active=%s%n",
                          gen, gBest, active);
    }

    // ── Accessor for individual agent histories ───────────────────────────────
    public List<Double> getGAHistory()   { return ga  != null ? ga.getHistory()  : List.of(); }
    public List<Double> getPSOHistory()  { return pso != null ? pso.getHistory() : List.of(); }
    public List<Double> getACSHistory()  { return acs != null ? acs.getHistory() : List.of(); }
    public List<Double> getMEAFHistory() { return meafHistory; }

    // ── Inner result record ───────────────────────────────────────────────────
    public static class RunResult {
        public final Individual   bestInd;
        public final List<Double> meafHistory;
        RunResult(Individual bestInd, List<Double> meafHistory) {
            this.bestInd     = bestInd;
            this.meafHistory = new ArrayList<>(meafHistory);
        }
    }
}