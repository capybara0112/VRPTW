package meaf;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Abstract base for all evolutionary agents (GA, PSO, ACS).
 */
public abstract class BaseAgent {

    protected final VRPTWInstance inst;
    protected boolean  active       = true;
    protected int      stableCount  = 0;
    protected List<Double> history  = new ArrayList<>();

    protected Individual bestInd;

    // MEAF parameters (from Table 1 & Table 2 of the paper)
    protected static final int    MEAF_T            = 5;
    protected static final int    MEAF_M            = 15;
    protected static final int    MEAF_SD_ITER      = 50;
    protected static final int    MEAF_EA_THRESHOLD = 2;
    protected static final double MEAF_DF_PERCENT   = 0.85;
    protected static final int    MEAF_K            = 1;

    protected BaseAgent(VRPTWInstance inst) {
        this.inst = inst;
    }

    public abstract void step();
    public abstract void injectGlobalBest(List<Individual> pool);
    public abstract void dissolveFrom(BaseAgent donor);

    public boolean isActive()   { return active; }
    public void    deactivate() { active = false; }
    public boolean isStable()   { return stableCount >= MEAF_T; }

    public double getBestPureDistance() {
        if (bestInd == null) return Double.MAX_VALUE;
        return inst.pureDistancePerm(bestInd.route);
    }

    public Individual getBestInd() { return bestInd; }
    public List<Double> getHistory() { return history; }

    // Nearest Neighbor heuristic (phiên bản cũ, có kiểm tra quay về depot)
    protected static int[] nearestNeighborSolution(VRPTWInstance inst, java.util.Random rand) {
        int N = inst.numCustomers;
        List<Integer> unvisited = new ArrayList<>();
        for (int i = 1; i <= N; i++) unvisited.add(i);
        Collections.shuffle(unvisited, rand);

        List<Integer> seq = new ArrayList<>();
        while (!unvisited.isEmpty()) {
            int cur = 0;
            double elapsed = 0.0, load = 0.0;
            boolean moved = true;
            while (moved && !unvisited.isEmpty()) {
                moved = false;
                int bestC = -1;
                double bestD = Double.MAX_VALUE;
                for (int c : unvisited) {
                    double dem = inst.customers[c].demand;
                    double d = inst.dist[cur][c];
                    double arr = elapsed + d;
                    // Kiểm tra khả năng quay về depot sau khi phục vụ
                    double tEnd = Math.max(arr, inst.customers[c].readyTime)
                                  + inst.customers[c].serviceTime
                                  + inst.dist[c][0];
                    if (load + dem <= inst.vehicleCapacity && tEnd <= inst.depot.dueTime) {
                        if (d < bestD) {
                            bestD = d;
                            bestC = c;
                        }
                    }
                }
                if (bestC >= 0) {
                    double arr = elapsed + inst.dist[cur][bestC];
                    elapsed = Math.max(arr, inst.customers[bestC].readyTime)
                              + inst.customers[bestC].serviceTime;
                    load += inst.customers[bestC].demand;
                    seq.add(bestC);
                    unvisited.remove(Integer.valueOf(bestC));
                    cur = bestC;
                    moved = true;
                }
            }
        }
        // Đảm bảo không thiếu khách hàng
        for (int i = 1; i <= N; i++) {
            if (!seq.contains(i)) seq.add(i);
        }
        int[] arr = new int[N];
        for (int i = 0; i < N; i++) arr[i] = seq.get(i);
        return arr;
    }

    // Khởi tạo quần thể với 90% NN + 10% ngẫu nhiên
    protected static List<Individual> buildInitialPopulation(VRPTWInstance inst,
                                                              int popSize,
                                                              java.util.Random rand) {
        List<Individual> pop = new ArrayList<>();
        int nHeuristic = (int) (popSize * 0.9);   // 90% NN
        for (int i = 0; i < nHeuristic; i++) {
            int[] perm = nearestNeighborSolution(inst, rand);
            pop.add(new Individual(perm, inst));
        }
        List<Integer> base = new ArrayList<>();
        for (int i = 1; i <= inst.numCustomers; i++) base.add(i);
        while (pop.size() < popSize) {
            Collections.shuffle(base, rand);
            int[] perm = base.stream().mapToInt(x -> x).toArray();
            pop.add(new Individual(perm, inst));
        }
        Collections.shuffle(pop, rand);
        return pop;
    }
}