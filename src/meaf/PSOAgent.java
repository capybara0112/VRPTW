package meaf;

import java.util.*;

public class PSOAgent extends BaseAgent {

    // Tham số đúng paper: swarm=100, w=0.10 (Bảng 2)
    private static final int    SWARM_SIZE = 100;
    private static final double W          = 0.10;   // inertia weight
    // Giới hạn vận tốc (không có trong paper nhưng cần để ổn định – common practice)
    private static final int    MAX_VELOCITY = 30;

    private final int[][]  particles;      // vị trí các hạt (hoán vị)
    private final double[] particleCost;
    private final int[][]  pbest;          // personal best
    private final double[] pbestCost;
    private int[]   gbest;                 // global best
    private double  gbestCost;

    private final int N;
    private final Random rand;
    private List<int[]>[] velocities;      // vận tốc = chuỗi swap operator

    @SuppressWarnings("unchecked")
    public PSOAgent(VRPTWInstance inst, List<Individual> sharedPop, Random rand) {
        super(inst);
        this.rand = rand;
        this.N = inst.numCustomers;

        // 1. Khởi tạo vị trí từ shared population (NN heuristic)
        particles    = new int[SWARM_SIZE][N];
        particleCost = new double[SWARM_SIZE];
        pbest        = new int[SWARM_SIZE][N];
        pbestCost    = new double[SWARM_SIZE];
        for (int i = 0; i < SWARM_SIZE; i++) {
            Individual ind = sharedPop.get(i % sharedPop.size());
            particles[i]    = ind.route.clone();
            particleCost[i] = inst.evalPerm(particles[i]);
            pbest[i]        = particles[i].clone();
            pbestCost[i]    = particleCost[i];
        }

        // 2. Xác định global best ban đầu
        int bestIdx = 0;
        for (int i = 1; i < SWARM_SIZE; i++) {
            if (pbestCost[i] < pbestCost[bestIdx]) bestIdx = i;
        }
        gbest     = pbest[bestIdx].clone();
        gbestCost = pbestCost[bestIdx];
        bestInd   = new Individual(gbest, gbestCost);

        // 3. Khởi tạo vận tốc ngẫu nhiên (vài swap ban đầu)
        velocities = new ArrayList[SWARM_SIZE];
        for (int i = 0; i < SWARM_SIZE; i++) {
            velocities[i] = new ArrayList<>();
            for (int k = 0; k < 5; k++) {   // 5 swap khởi tạo là đủ
                velocities[i].add(new int[]{rand.nextInt(N), rand.nextInt(N)});
            }
        }
        recordHistory();
    }

    @Override
    public void step() {
        if (!active) return;
        double prevBest = gbestCost;

        for (int i = 0; i < SWARM_SIZE; i++) {
            List<int[]> newVelocity = new ArrayList<>();

            // --- 1. Inertia ---
            for (int[] swap : velocities[i]) {
                if (rand.nextDouble() < W) {
                    newVelocity.add(swap);
                }
            }

            // --- 2. Cognitive ---
            double r1 = rand.nextDouble();   // r1 ~ U(0,1)
            List<int[]> diffToPbest = getSwapSequence(particles[i], pbest[i]);
            for (int[] swap : diffToPbest) {
                if (rand.nextDouble() < r1) {
                    newVelocity.add(swap);
                }
            }

            // --- 3. Social ---
            double r2 = rand.nextDouble();   // r2 ~ U(0,1)
            List<int[]> diffToGbest = getSwapSequence(particles[i], gbest);
            for (int[] swap : diffToGbest) {
                if (rand.nextDouble() < r2) {
                    newVelocity.add(swap);
                }
            }

            // --- 4. Giới hạn vận tốc (bounded velocity) ---
            if (newVelocity.size() > MAX_VELOCITY) {
                Collections.shuffle(newVelocity, rand);
                newVelocity = new ArrayList<>(newVelocity.subList(0, MAX_VELOCITY));
            }

            // Reset nếu velocity quá lớn (tránh explosion)
            if (velocities[i].size() > MAX_VELOCITY * 2) {
                velocities[i].clear();
            }

            velocities[i] = newVelocity;

            // --- 5. Áp dụng vận tốc để tạo vị trí mới ---
            int[] newPos = particles[i].clone();
            for (int[] swap : velocities[i]) {
                int tmp = newPos[swap[0]];
                newPos[swap[0]] = newPos[swap[1]];
                newPos[swap[1]] = tmp;
            }
            particles[i]    = newPos;
            particleCost[i] = inst.evalPerm(newPos);

            // --- 6. Cập nhật pbest & gbest ---
            if (particleCost[i] < pbestCost[i]) {
                pbest[i]     = particles[i].clone();
                pbestCost[i] = particleCost[i];
                if (particleCost[i] < gbestCost) {
                    gbest     = particles[i].clone();
                    gbestCost = particleCost[i];
                }
            }
        }

        // --- 7. Cập nhật trạng thái hội tụ ---
        if (gbestCost < prevBest - 1e-9) {
            stableCount = 0;
            bestInd = new Individual(gbest, gbestCost);
        } else {
            stableCount++;
        }
        recordHistory();
    }

    // ==================== Fusion (Eq. 21) ====================
    @Override
    public void injectGlobalBest(List<Individual> pool) {
        if (!active || pool == null || pool.isEmpty()) return;

        for (Individual extInd : pool) {
            int[] extRoute = extInd.route.clone();
            double extCost = inst.evalPerm(extRoute);

            if (extCost < gbestCost) {
                gbest = extRoute.clone();
                gbestCost = extCost;
                bestInd = new Individual(gbest, gbestCost);
            }

            // Chỉ inject vào khoảng 25% swarm (giữ diversity)
            int affected = Math.max(1, SWARM_SIZE / 4);
            for (int z = 0; z < affected; z++) {
                int i = rand.nextInt(SWARM_SIZE);
                List<int[]> seqToExternal = getSwapSequence(particles[i], extRoute);
                double r_ext = rand.nextDouble();
                for (int[] swap : seqToExternal) {
                    if (rand.nextDouble() < r_ext) {
                        velocities[i].add(swap);
                    }
                }
            }
        }
    }

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
            if (rand.nextDouble() < 0.3) {
                int a = rand.nextInt(N), b = rand.nextInt(N);
                int tmp = ext[a]; ext[a] = ext[b]; ext[b] = tmp;
            }
            double extCost = inst.evalPerm(ext);
            particles[wi] = ext;
            pbestCost[wi]  = extCost;
            pbest[wi]      = ext.clone();
            if (extCost < gbestCost) {
                gbest = ext.clone();
                gbestCost = extCost;
                bestInd = new Individual(gbest, gbestCost);
            }
        }
        System.out.printf("    [PSO] Dissolved: re-seeded %d particles from %s%n",
                nCopy, donor.getClass().getSimpleName());
    }

    // ==================== Helper ====================
    // Tạo swap sequence tối thiểu bằng cách dùng map vị trí (O(N))
    private List<int[]> getSwapSequence(int[] current, int[] target) {
        List<int[]> sequence = new ArrayList<>();
        int[] temp = current.clone();
        Map<Integer, Integer> posMap = new HashMap<>();
        for (int k = 0; k < temp.length; k++) {
            posMap.put(temp[k], k);
        }
        for (int i = 0; i < temp.length; i++) {
            if (temp[i] != target[i]) {
                int j = posMap.get(target[i]);
                sequence.add(new int[]{i, j});
                // cập nhật map
                posMap.put(temp[i], j);
                posMap.put(temp[j], i);
                // swap
                int tmp = temp[i];
                temp[i] = temp[j];
                temp[j] = tmp;
            }
        }
        return sequence;
    }

    private void recordHistory() {
        history.add(inst.pureDistancePerm(bestInd.route));
    }
}