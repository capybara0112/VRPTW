package meaf;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

/**
 * ══════════════════════════════════════════════════════════════════════════════
 *  MEAVRPTW – Main Entry Point
 *  Replicates the experiment structure from the paper (Applied Intelligence 2026)
 *
 *  Runs:
 *    GA standalone  (with 2-Opt LS)
 *    PSO standalone (Discrete, swap-based)
 *    ACS standalone (Pheromone-guided)
 *    MEAF           (Membrane-inspired framework combining all three)
 *
 *  Each algorithm is run N_RUNS times independently.
 *  Outputs:
 *    STDOUT: Summary table (matches paper Table 3 style)
 *    CSV:    convergence history → results/C101_convergence.csv
 *
 *  Demo config: N_GEN=50, N_RUNS=2, dataset=C101
 * ══════════════════════════════════════════════════════════════════════════════
 */
public class Main {

    static final int N_GEN  = Orchestrator.N_GEN;
    static final int N_RUNS = Orchestrator.N_RUNS;

    public static void main(String[] args) throws Exception {
        // ── Load instance ──────────────────────────────────────────────────
        String txtPath = args.length > 0 ? args[0] : "C101.txt"; 
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  MEAVRPTW – Membrane-Inspired EA Framework (Java)        ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.printf("[INFO] Loading: %s%n", txtPath);

        VRPTWInstance inst = VRPTWInstance.load(txtPath);
        System.out.printf("[INFO] Instance: %s | Customers: %d | Vehicles: %d | Cap: %d%n",
                          inst.name, inst.numCustomers, inst.numVehicles, inst.vehicleCapacity);
        System.out.printf("[INFO] Demo config: N_GEN=%d, N_RUNS=%d%n", N_GEN, N_RUNS);
        System.out.printf("[INFO] MEAF: M=%d, sditer=%d, dfpercent=%.2f%n%n",
                          Orchestrator.MEAF_M, Orchestrator.MEAF_SD_ITER, Orchestrator.MEAF_DF);

        // ── Run each algorithm N_RUNS times ───────────────────────────────
        AlgoResult gaResult   = runAlgo("GA",   inst, Main::runGA);
        AlgoResult psoResult  = runAlgo("PSO",  inst, Main::runPSO);
        AlgoResult acsResult  = runAlgo("ACS",  inst, Main::runACS);
        AlgoResult meafResult = runAlgo("MEAF", inst, Main::runMEAF);

        // ── Print summary table ────────────────────────────────────────────
        printSummary(inst, gaResult, psoResult, acsResult, meafResult);

        // ── Export CSV ────────────────────────────────────────────────────
        exportCSV(inst, gaResult, psoResult, acsResult, meafResult);

        System.out.println("\n[✔] Hoàn thành. Kết quả lưu tại results/");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Algorithm runners (return convergence history per run)
    // ══════════════════════════════════════════════════════════════════════════

    @FunctionalInterface
    interface AlgoRunner {
        List<Double> run(VRPTWInstance inst, List<Individual> sharedPop, Random rand);
    }

    static List<Double> runGA(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        GAAgent agent = new GAAgent(inst, sharedPop, rand);
        for (int g = 0; g < N_GEN; g++) agent.step();
        return padHistory(agent.getHistory(), N_GEN + 1);
    }

    static List<Double> runPSO(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        PSOAgent agent = new PSOAgent(inst, sharedPop, rand);
        for (int g = 0; g < N_GEN; g++) agent.step();
        return padHistory(agent.getHistory(), N_GEN + 1);
    }

    static List<Double> runACS(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        ACSAgent agent = new ACSAgent(inst, sharedPop, rand);
        for (int g = 0; g < N_GEN; g++) agent.step();
        return padHistory(agent.getHistory(), N_GEN + 1);
    }

    static List<Double> runMEAF(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        Orchestrator orch = new Orchestrator(inst, rand);
        Orchestrator.RunResult result = orch.run(sharedPop);
        return padHistory(result.meafHistory, N_GEN + 1);
    }

    // ── Generic experiment runner ─────────────────────────────────────────────
    static AlgoResult runAlgo(String name, VRPTWInstance inst, AlgoRunner runner) {
        System.out.printf("%n┌─────────────────────────────────────────────────────┐%n");
        System.out.printf("│  %s – Running %d independent runs × %d generations%n", name, N_RUNS, N_GEN);
        System.out.printf("└─────────────────────────────────────────────────────┘%n");

        List<Double>         allBestCosts = new ArrayList<>();
        List<Long>           allTimes     = new ArrayList<>();
        List<List<Double>>   allHistories = new ArrayList<>();

        for (int r = 0; r < N_RUNS; r++) {
            Random rand = new Random((long)(r * 42 + 7));
            // Fresh shared population for each run
            List<Individual> sharedPop = BaseAgent.buildInitialPopulation(inst, 100, rand);

            long t0 = System.currentTimeMillis();
            List<Double> hist = runner.run(inst, sharedPop, rand);
            long elapsed = System.currentTimeMillis() - t0;

            double bestDist = hist.get(hist.size() - 1);
            allBestCosts.add(bestDist);
            allTimes.add(elapsed);
            allHistories.add(hist);

            System.out.printf("  [Run %02d] Final=%.2f | Time=%.2fs%n",
                              r + 1, bestDist, elapsed / 1000.0);
        }

        // Average history across runs
        int nPts = allHistories.stream().mapToInt(List::size).min().orElse(0);
        List<Double> avgHistory = new ArrayList<>();
        for (int i = 0; i < nPts; i++) {
            final int idx = i;
            avgHistory.add(allHistories.stream()
                .mapToDouble(h -> h.get(idx)).average().orElse(0.0));
        }

        double mean = allBestCosts.stream().mapToDouble(d -> d).average().orElse(0.0);
        double std  = stdDev(allBestCosts, mean);
        double best = allBestCosts.stream().mapToDouble(d -> d).min().orElse(0.0);
        double avgT = allTimes.stream().mapToLong(l -> l).average().orElse(0.0) / 1000.0;

        return new AlgoResult(name, mean, std, best, avgT, avgHistory);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Output helpers
    // ══════════════════════════════════════════════════════════════════════════

    static void printSummary(VRPTWInstance inst,
                              AlgoResult ga, AlgoResult pso,
                              AlgoResult acs, AlgoResult meaf) {
        System.out.printf("%n%s%n", "═".repeat(78));
        System.out.printf(" SUMMARY – %s | Pop=100 | Gens=%d | Runs=%d%n",
                          inst.name, N_GEN, N_RUNS);
        System.out.printf("%s%n", "═".repeat(78));
        System.out.printf(" %-10s | %12s | %10s | %12s | %10s%n",
                          "Algorithm", "Mean Dist", "Std Dev", "Avg Time(s)", "Best Dist");
        System.out.printf("%s%n", "─".repeat(78));

        double bestMean = Math.min(ga.mean, Math.min(pso.mean, Math.min(acs.mean, meaf.mean)));
        for (AlgoResult r : new AlgoResult[]{ga, pso, acs, meaf}) {
            String star = (Math.abs(r.mean - bestMean) < 1e-6) ? " ★" : "";
            System.out.printf(" %-10s | %12.2f | %10.2f | %12.2f | %10.2f%s%n",
                              r.name, r.mean, r.std, r.avgTime, r.best, star);
        }

        double singleBest = Math.min(ga.mean, Math.min(pso.mean, acs.mean));
        double reduction  = (singleBest - meaf.mean) / singleBest * 100.0;
        System.out.printf("%n  MEAF Cost Reduction vs Best Single-Algo = %.2f%%%n", reduction);
        System.out.printf("%s%n%n", "═".repeat(78));
    }

    static void exportCSV(VRPTWInstance inst,
                           AlgoResult ga, AlgoResult pso,
                           AlgoResult acs, AlgoResult meaf) throws IOException {
        new File("results").mkdirs();
        String path = "results/" + inst.name + "_convergence.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("Generation,GA_Cost,PSO_Cost,ACS_Cost,MEAF_Cost");
            int nPts = Stream.of(ga, pso, acs, meaf)
                .mapToInt(r -> r.avgHistory.size()).min().orElse(0);
            for (int i = 0; i < nPts; i++) {
                pw.printf("%d,%.4f,%.4f,%.4f,%.4f%n",
                    i,
                    ga.avgHistory.get(i),
                    pso.avgHistory.get(i),
                    acs.avgHistory.get(i),
                    meaf.avgHistory.get(i));
            }
        }
        System.out.printf("[✔] CSV hội tụ lưu tại: %s%n", path);
    }

    // ── Utility ───────────────────────────────────────────────────────────────
    static List<Double> padHistory(List<Double> hist, int target) {
        List<Double> out = new ArrayList<>(hist);
        while (out.size() < target) out.add(out.get(out.size() - 1));
        if (out.size() > target) out = out.subList(0, target);
        return out;
    }

    static double stdDev(List<Double> vals, double mean) {
        if (vals.size() < 2) return 0.0;
        double sumSq = vals.stream().mapToDouble(v -> (v - mean) * (v - mean)).sum();
        return Math.sqrt(sumSq / (vals.size() - 1));
    }

    // ── Result record ──────────────────────────────────────────────────────────
    static class AlgoResult {
        final String       name;
        final double       mean, std, best, avgTime;
        final List<Double> avgHistory;

        AlgoResult(String name, double mean, double std, double best,
                   double avgTime, List<Double> avgHistory) {
            this.name       = name;
            this.mean       = mean;
            this.std        = std;
            this.best       = best;
            this.avgTime    = avgTime;
            this.avgHistory = avgHistory;
        }
    }
}