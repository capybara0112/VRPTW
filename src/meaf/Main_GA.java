package meaf;

import java.io.*;
import java.util.*;
import java.util.stream.*;

/**
 * Main entry point chỉ để chạy GA độc lập.
 * Xuất file CSV và vẽ biểu đồ như bình thường (các cột PSO, ACS, MEAF sẽ là 0).
 */
public class Main_GA {

    static final int N_GEN  = Orchestrator.N_GEN;
    static final int N_RUNS = Orchestrator.N_RUNS;

    public static void main(String[] args) throws Exception {
        String txtPath = args.length > 0 ? args[0] : "C101.txt";
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  MEAVRPTW – GA Standalone Run                           ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.printf("[INFO] Loading: %s%n", txtPath);

        VRPTWInstance inst = VRPTWInstance.load(txtPath);
        System.out.printf("[INFO] Instance: %s | Customers: %d | Vehicles: %d | Cap: %d%n",
                inst.name, inst.numCustomers, inst.numVehicles, inst.vehicleCapacity);
        System.out.printf("[INFO] Config: N_GEN=%d, N_RUNS=%d%n%n", N_GEN, N_RUNS);

        // Chỉ chạy GA
        AlgoResult gaResult = runAlgo("GA", inst, Main::runGA);

        // Tạo dummy result cho các thuật toán khác
        List<Double> zeroHistory = new ArrayList<>();
        for (int i = 0; i <= N_GEN; i++) zeroHistory.add(0.0);
        AlgoResult psoResult  = new AlgoResult("PSO",  0, 0, 0, 0, zeroHistory);
        AlgoResult acsResult  = new AlgoResult("ACS",  0, 0, 0, 0, zeroHistory);
        AlgoResult meafResult = new AlgoResult("MEAF", 0, 0, 0, 0, zeroHistory);

        // In bảng tổng kết
        printSummary(inst, gaResult, psoResult, acsResult, meafResult);

        // Xuất CSV
        exportCSV(inst, gaResult, psoResult, acsResult, meafResult);

        System.out.println("\n[✔] Hoàn thành. Kết quả lưu tại results/");
    }

    // ========== Các hàm runner (chỉ dùng runGA) ==========
    @FunctionalInterface
    interface AlgoRunner {
        List<Double> run(VRPTWInstance inst, List<Individual> sharedPop, Random rand);
    }

    // Chạy một lần GA (dùng code từ Main.java)
    static List<Double> runGA(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        GAAgent agent = new GAAgent(inst, sharedPop, rand);
        for (int g = 0; g < N_GEN; g++) agent.step();
        return padHistory(agent.getHistory(), N_GEN + 1);
    }

    // ========== Hàm chạy thí nghiệm ==========
    static AlgoResult runAlgo(String name, VRPTWInstance inst, AlgoRunner runner) {
        System.out.printf("%n┌─────────────────────────────────────────────────────┐%n");
        System.out.printf("│  %s – Running %d independent runs × %d generations%n", name, N_RUNS, N_GEN);
        System.out.printf("└─────────────────────────────────────────────────────┘%n");

        List<Double>         allBestCosts = new ArrayList<>();
        List<Long>           allTimes     = new ArrayList<>();
        List<List<Double>>   allHistories = new ArrayList<>();

        for (int r = 0; r < N_RUNS; r++) {
            Random rand = new Random((long)(r * 42 + 7));
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

        // Trung bình lịch sử
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

    // ========== In bảng tổng kết ==========
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

    // ========== Xuất CSV ==========
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

    // ========== Tiện ích ==========
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

    // ========== Lớp lưu kết quả ==========
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