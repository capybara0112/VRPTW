package meaf;

import java.util.*;

/**
 * Discrete Particle Swarm Optimization Agent cho bài toán VRPTW.
 * Triển khai theo Wang (2003) swap‑operator formulation, đúng như mô tả
 * trong Mục 3.5.2 của bài báo "A novel membrane‑inspired evolutionary
 * algorithm framework for VRPTW".
 *
 * Tham số (Bảng 2): swarm = 100, w = 0.10.
 * Công thức cập nhật (Eq. 18):
 *   V_i(k+1) = w V_i(k) ⊕ r1 (pbest_i − X_i(k)) ⊕ r2 (gbest − X_i(k))
 *   X_i(k+1) = X_i(k) + V_i(k+1)
 */
public class PSOAgent extends BaseAgent {

    // ── Tham số từ bài báo ─────────────────────────────────────────────────
    private static final int    SWARM_SIZE = 100;   // kích thước bầy
    private static final double W          = 0.10;   // trọng số quán tính (inertia weight)

    // ── Giới hạn vận tốc (theo thực nghiệm, không có trong bài báo) ──────────
    private static final int    MAX_VELOCITY = 30;   // ngăn velocity phình vô hạn

    // ── Trạng thái của bầy ─────────────────────────────────────────────────
    private final int[][]   particles;      // vị trí các hạt (hoán vị khách hàng)
    private final double[]  particleCost;   // chi phí của từng hạt
    private final int[][]   pbest;          // vị trí tốt nhất cá nhân (personal best)
    private final double[]  pbestCost;      // chi phí của pbest
    private int[]   gbest;                  // vị trí tốt nhất toàn cục (global best)
    private double  gbestCost;              // chi phí của gbest

    private final int       N;              // số khách hàng (không tính depot)
    private final Random    rand;           // bộ sinh số ngẫu nhiên riêng
    private List<int[]>[]   velocities;     // vận tốc của từng hạt (chuỗi swap)

    // ── Constructor ─────────────────────────────────────────────────────────
    @SuppressWarnings("unchecked")
    public PSOAgent(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        super(inst);
        this.rand = rand;
        this.N = inst.numCustomers;

        // 1. Khởi tạo vị trí ban đầu bằng Nearest‑Neighbor heuristic (trang 13)
        particles    = new int[SWARM_SIZE][N];
        particleCost = new double[SWARM_SIZE];
        pbest        = new int[SWARM_SIZE][N];
        pbestCost    = new double[SWARM_SIZE];
        for (int i = 0; i < SWARM_SIZE; i++) {
            Individual ind = sharedPop.get(i % sharedPop.size());
            particles[i]    = ind.route.clone();
            particleCost[i] = inst.evalPerm(particles[i]);   // đánh giá fitness
            pbest[i]        = particles[i].clone();          // pbest ban đầu = chính nó
            pbestCost[i]    = particleCost[i];
        }

        // 2. Xác định global best ban đầu
        int bestIdx = 0;
        for (int i = 1; i < SWARM_SIZE; i++) {
            if (pbestCost[i] < pbestCost[bestIdx]) bestIdx = i;
        }
        gbest     = pbest[bestIdx].clone();
        gbestCost = pbestCost[bestIdx];
        bestInd   = new Individual(gbest, gbestCost);       // để BaseAgent theo dõi

        // 3. Khởi tạo vận tốc ban đầu (vài swap ngẫu nhiên)
        velocities = new ArrayList[SWARM_SIZE];
        for (int i = 0; i < SWARM_SIZE; i++) {
            velocities[i] = new ArrayList<>();
            for (int k = 0; k < 5; k++) {
                velocities[i].add(new int[]{rand.nextInt(N), rand.nextInt(N)});
            }
        }

        recordHistory();
    }

    // ── Vòng lặp chính (một thế hệ) ─────────────────────────────────────────
    @Override
    public void step() {
        if (!active) return;
        double prevBest = gbestCost;

        for (int i = 0; i < SWARM_SIZE; i++) {
            // === 1. Inertia (quán tính) ===
            List<int[]> newVelocity = new ArrayList<>();
            for (int[] swap : velocities[i]) {
                if (rand.nextDouble() < W) {
                    newVelocity.add(swap);
                }
            }

            // === 2. Cognitive (học từ pbest) ===
            double r1 = rand.nextDouble();                       // r1 ~ U(0,1)
            List<int[]> diffToPbest = getSwapSequence(particles[i], pbest[i]);
            for (int[] swap : diffToPbest) {
                if (rand.nextDouble() < r1) {
                    newVelocity.add(swap);
                }
            }

            // === 3. Social (học từ gbest) ===
            double r2 = rand.nextDouble();                       // r2 ~ U(0,1)
            List<int[]> diffToGbest = getSwapSequence(particles[i], gbest);
            for (int[] swap : diffToGbest) {
                if (rand.nextDouble() < r2) {
                    newVelocity.add(swap);
                }
            }

            // === 4. Giới hạn vận tốc (clamp) để tránh nhiễu loạn ===
            if (newVelocity.size() > MAX_VELOCITY) {
                Collections.shuffle(newVelocity, rand);
                newVelocity = new ArrayList<>(newVelocity.subList(0, MAX_VELOCITY));
            }
            // Reset nếu velocity tích lũy quá lớn qua các thế hệ
            if (velocities[i].size() > MAX_VELOCITY * 2) {
                velocities[i].clear();
            }

            velocities[i] = newVelocity;

            // === 5. Áp dụng vận tốc → vị trí mới (X + V) ===
            int[] newPos = particles[i].clone();
            for (int[] swap : velocities[i]) {
                int tmp = newPos[swap[0]];
                newPos[swap[0]] = newPos[swap[1]];
                newPos[swap[1]] = tmp;
            }
            particles[i]    = newPos;
            particleCost[i] = inst.evalPerm(newPos);

            // === 6. Cập nhật pbest và gbest ===
            if (particleCost[i] < pbestCost[i]) {
                pbest[i]     = particles[i].clone();
                pbestCost[i] = particleCost[i];
                if (particleCost[i] < gbestCost) {
                    gbest     = particles[i].clone();
                    gbestCost = particleCost[i];
                }
            }
        }

        // === 7. Cập nhật trạng thái hội tụ (dùng cho MEAF) ===
        if (gbestCost < prevBest - 1e-9) {
            stableCount = 0;
            bestInd = new Individual(gbest, gbestCost);
        } else {
            stableCount++;
        }
        recordHistory();
    }

    // ── Fusion (Eq. 21) – chỉ dùng khi PSO nằm trong MEAF ──────────────────
    @Override
    public void injectGlobalBest(List<Individual> pool) {
        if (!active || pool == null || pool.isEmpty()) return;

        for (Individual extInd : pool) {
            int[] extRoute = extInd.route.clone();
            double extCost = inst.evalPerm(extRoute);

            // Cập nhật gbest nếu giải pháp ngoại lai tốt hơn
            if (extCost < gbestCost) {
                gbest     = extRoute.clone();
                gbestCost = extCost;
                bestInd   = new Individual(gbest, gbestCost);
            }

            // Chỉ ảnh hưởng đến khoảng 25% bầy (giữ diversity)
            int affected = Math.max(1, SWARM_SIZE / 4);
            for (int z = 0; z < affected; z++) {
                int idx = rand.nextInt(SWARM_SIZE);
                List<int[]> seqToExternal = getSwapSequence(particles[idx], extRoute);
                double rExt = rand.nextDouble();
                for (int[] swap : seqToExternal) {
                    if (rand.nextDouble() < rExt) {
                        velocities[idx].add(swap);
                    }
                }
            }
        }
    }

    // ── Dissolve Rule (khi bị MEAF loại bỏ) ─────────────────────────────────
    @Override
    public void dissolveFrom(BaseAgent donor) {
        if (!active) return;
        int nCopy = SWARM_SIZE / 2;
        Integer[] idx = new Integer[SWARM_SIZE];
        for (int i = 0; i < SWARM_SIZE; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(pbestCost[b], pbestCost[a]));

        int[] donorRoute = donor.getBestInd().route;
        for (int i = 0; i < nCopy; i++) {
            int wi = idx[i];
            int[] ext = donorRoute.clone();
            // Thêm nhiễu nhẹ
            if (rand.nextDouble() < 0.3) {
                int a = rand.nextInt(N), b = rand.nextInt(N);
                int tmp = ext[a]; ext[a] = ext[b]; ext[b] = tmp;
            }
            double extCost = inst.evalPerm(ext);
            particles[wi]   = ext;
            pbestCost[wi]   = extCost;
            pbest[wi]       = ext.clone();
            if (extCost < gbestCost) {
                gbest     = ext.clone();
                gbestCost = extCost;
                bestInd   = new Individual(gbest, gbestCost);
            }
        }
        System.out.printf("    [PSO] Dissolved: re‑seeded %d particles from %s%n",
                nCopy, donor.getClass().getSimpleName());
    }

    // ── Helper: tạo chuỗi swap tối thiểu để biến current → target (O(N)) ───
    private List<int[]> getSwapSequence(int[] current, int[] target) {
        List<int[]> sequence = new ArrayList<>();
        int[] temp = current.clone();
        // Dùng map vị trí để đạt O(N)
        Map<Integer, Integer> pos = new HashMap<>();
        for (int k = 0; k < temp.length; k++) {
            pos.put(temp[k], k);
        }
        for (int i = 0; i < temp.length; i++) {
            if (temp[i] != target[i]) {
                int j = pos.get(target[i]);
                sequence.add(new int[]{i, j});
                // cập nhật map
                pos.put(temp[i], j);
                pos.put(temp[j], i);
                // swap
                int tmp = temp[i];
                temp[i] = temp[j];
                temp[j] = tmp;
            }
        }
        return sequence;
    }

    // ── Ghi lại lịch sử hội tụ (pure distance) ─────────────────────────────
    private void recordHistory() {
        history.add(inst.pureDistancePerm(bestInd.route));
    }
}