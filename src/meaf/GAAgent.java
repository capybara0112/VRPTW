package meaf;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Genetic Algorithm Agent (standard version – strictly follows the paper).
 *
 * Paper Table 2 parameters:
 *   pop = 100, ts = 0.20 (truncation selection), mr = 0.50
 * Operators: PMX crossover, Reverse Sequence Mutation (RSM)
 * No 2‑Opt, no elitism.
 */
public class GAAgent extends BaseAgent {

    private static final int    POP_SIZE  = 100;
    private static final double TS        = 0.20;
    private static final double MUT_RATE  = 0.50;
    // The paper does not specify crossover probability; 0.85 is a common default.
    private static final double CX_PB     = 0.85;

    private List<Individual> pop;
    private final Random rand;

    public GAAgent(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        super(inst);
        this.rand = rand;
        // Khởi tạo từ shared population (được tạo bởi BaseAgent.buildInitialPopulation)
        this.pop = sharedPop.stream()
                .map(Individual::clone)
                .collect(Collectors.toList());
        updateBest();
        recordHistory();
    }

    @Override
    public void step() {
        if (!active) return;

        // 1. Truncation selection: giữ lại top ts% làm tập ứng viên cha mẹ
        int nTrunc = Math.max(2, (int) (POP_SIZE * TS));
        List<Individual> candidates = pop.stream()
                .sorted(Comparator.comparingDouble(i -> i.cost)) // chi phí thấp = tốt
                .limit(nTrunc)
                .collect(Collectors.toList());

        // 2. Fitness‑proportionate selection (roulette wheel) trên tập ứng viên
        double maxCost = candidates.stream()
                .mapToDouble(i -> i.cost)
                .max().orElse(1.0);
        double[] weights = candidates.stream()
                .mapToDouble(i -> maxCost - i.cost + 1e-9) // fitness = maxCost - cost
                .toArray();
        double sumW = Arrays.stream(weights).sum();
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= sumW;
        }

        // 3. Sinh offspring
        List<Individual> offspring = new ArrayList<>();
        while (offspring.size() < POP_SIZE) {
            Individual p1 = rouletteSelect(candidates, weights);
            Individual p2 = rouletteSelect(candidates, weights);

            Individual c1 = p1.clone();
            Individual c2 = p2.clone();

            // PMX Crossover
            if (rand.nextDouble() < CX_PB) {
                Individual[] children = Individual.pmxCrossover(p1, p2, inst, rand);
                c1 = children[0];
                c2 = children[1];
            }

            // Reverse Sequence Mutation
            if (rand.nextDouble() < MUT_RATE) {
                c1 = c1.mutate(inst, rand);
            }
            if (rand.nextDouble() < MUT_RATE) {
                c2 = c2.mutate(inst, rand);
            }

            offspring.add(c1);
            if (offspring.size() < POP_SIZE) {
                offspring.add(c2);
            }
        }

        // 4. Thay thế quần thể cũ (không elitism)
        pop = offspring;

        // 5. Cập nhật giải pháp tốt nhất
        double prevCost = bestInd != null ? bestInd.cost : Double.MAX_VALUE;
        updateBest();
        if (bestInd.cost < prevCost - 1e-9) {
            stableCount = 0;
        } else {
            stableCount++;
        }

        recordHistory();
    }

    // ------------------------------------------------------------
    // MEAF Communication Input (Fusion strategy for GA, Section 5.2)
    // ------------------------------------------------------------
    @Override
    public void injectGlobalBest(List<Individual> pool) {
        if (!active || pool == null || pool.isEmpty()) return;

        // Sắp xếp quần thể: tệ nhất (cost cao nhất) đầu tiên
        List<Integer> sortedIdx = new ArrayList<>();
        for (int i = 0; i < pop.size(); i++) sortedIdx.add(i);
        sortedIdx.sort((a, b) -> Double.compare(pop.get(b).cost, pop.get(a).cost));

        int k = Math.min(pool.size(), MEAF_K);
        for (int i = 0; i < k; i++) {
            Individual ext = pool.get(i).clone();
            ext.cost = inst.evalPerm(ext.route);    // đảm bảo cost được cập nhật
            pop.set(sortedIdx.get(i), ext);
        }
        updateBest();
    }

    // ------------------------------------------------------------
    // Dissolve Rule (thay thế một nửa quần thể bằng bản sao của donor)
    // ------------------------------------------------------------
    @Override
    public void dissolveFrom(BaseAgent donor) {
        if (!active) return;
        int nCopy = POP_SIZE / 2;

        List<Integer> sortedIdx = new ArrayList<>();
        for (int i = 0; i < pop.size(); i++) sortedIdx.add(i);
        sortedIdx.sort((a, b) -> Double.compare(pop.get(b).cost, pop.get(a).cost));

        Individual donorBest = donor.getBestInd();
        for (int i = 0; i < nCopy; i++) {
            Individual ext = donorBest.clone();
            // Thêm nhiễu nhẹ cho đa dạng
            if (rand.nextDouble() < 0.4) {
                ext = ext.mutate(inst, rand);
            }
            pop.set(sortedIdx.get(i), ext);
        }
        updateBest();
        System.out.printf("    [GA] Dissolved: re-seeded %d individuals from %s%n",
                nCopy, donor.getClass().getSimpleName());
    }

    // ------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------
    private Individual rouletteSelect(List<Individual> candidates, double[] weights) {
        double r = rand.nextDouble();
        double cum = 0.0;
        for (int i = 0; i < candidates.size(); i++) {
            cum += weights[i];
            if (r <= cum) {
                return candidates.get(i);
            }
        }
        return candidates.get(candidates.size() - 1);
    }

    private void updateBest() {
        Individual localBest = pop.stream()
                .min(Comparator.comparingDouble(i -> i.cost))
                .orElse(null);
        if (localBest != null) {
            if (bestInd == null || localBest.cost < bestInd.cost) {
                bestInd = localBest.clone();
            }
        }
    }

    private void recordHistory() {
        history.add(inst.pureDistancePerm(bestInd.route));
    }

    /** Trả về danh sách cá thể (để tương thích với các thành phần khác). */
    public List<Individual> getPop() {
        return new ArrayList<>(pop);
    }
}