package meaf;

import java.io.*;
import java.util.*;

public class VRPTWInstance {

    public final String   name;
    public final int      numVehicles;
    public final int      vehicleCapacity;
    public final Customer depot;
    public final Customer[] customers;
    public final int      numCustomers;
    public final double[][] dist;

    public static final double UNIT_COST  = 1.0;
    public static final double INIT_COST  = 0.0;
    public static final double WAIT_COST  = 0.0;
    public static final double DELAY_COST = 0.0;

    private VRPTWInstance(String name, int numVehicles, int vehicleCapacity,
                          Customer[] customers) {
        this.name            = name;
        this.numVehicles     = numVehicles;
        this.vehicleCapacity = vehicleCapacity;
        this.customers       = customers;
        this.depot           = customers[0];
        this.numCustomers    = customers.length - 1;

        int n = customers.length;
        dist = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dx = customers[i].x - customers[j].x;
                double dy = customers[i].y - customers[j].y;
                dist[i][j] = Math.sqrt(dx * dx + dy * dy);
            }
        }
    }

    public static VRPTWInstance load(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String instanceName = br.readLine().trim();
            br.readLine();
            br.readLine();
            br.readLine();
            String[] vh = br.readLine().trim().split("\\s+");
            int numVeh = Integer.parseInt(vh[0]);
            int cap    = Integer.parseInt(vh[1]);
            br.readLine();
            br.readLine();
            br.readLine();
            br.readLine();

            List<Customer> custs = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] tok = line.split("\\s+");
                if (tok.length < 7) continue;
                int id = Integer.parseInt(tok[0]);
                double x = Double.parseDouble(tok[1]);
                double y = Double.parseDouble(tok[2]);
                int dem = Integer.parseInt(tok[3]);
                int rt  = Integer.parseInt(tok[4]);
                int due = Integer.parseInt(tok[5]);
                int svc = Integer.parseInt(tok[6]);
                custs.add(new Customer(id, x, y, dem, rt, due, svc));
            }
            Customer[] arr = custs.toArray(new Customer[0]);
            return new VRPTWInstance(instanceName, numVeh, cap, arr);
        }
    }

    /**
     * Sequential decode + merge single‑customer routes only.
     * B1: duyệt tuần tự, nếu vượt tải hoặc đến trễ → tạo route mới.
     * B2: với mỗi route chỉ có 1 khách hàng, thử chèn khách hàng đó
     *     vào các route khác (theo first feasible). Nếu chèn được, xoá route 1‑customer.
     */
    public List<List<Integer>> decode(int[] perm) {
        // === B1: sequential decoding ===
        List<List<Integer>> routes = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        int load = 0;
        double currentTime = 0;
        int prev = 0;

        for (int c : perm) {
            int dem = customers[c].demand;
            double travelTime = dist[prev][c];
            double arrivalTime = currentTime + travelTime;

            // Nếu vượt tải hoặc đến trễ → tạo route mới
            if (load + dem > vehicleCapacity || arrivalTime > customers[c].dueTime) {
                if (!cur.isEmpty()) {
                    routes.add(cur);
                }
                cur = new ArrayList<>();
                load = 0;
                currentTime = 0;
                prev = 0;
                travelTime = dist[0][c];
                arrivalTime = travelTime;
            }

            double serviceStart = Math.max(arrivalTime, customers[c].readyTime);
            cur.add(c);
            load += dem;
            currentTime = serviceStart + customers[c].serviceTime;
            prev = c;
        }
        if (!cur.isEmpty()) routes.add(cur);

        // === B2: merge single‑customer routes (size == 1) ===
        for (int i = routes.size() - 1; i >= 0; i--) {
            List<Integer> small = routes.get(i);
            if (small.size() == 1) {
                int c = small.get(0);
                boolean inserted = false;
                for (int j = 0; j < routes.size(); j++) {
                    if (j == i) continue;
                    List<Integer> target = routes.get(j);
                    // Kiểm tra tải trọng
                    int targetLoad = 0;
                    for (int id : target) targetLoad += customers[id].demand;
                    if (targetLoad + customers[c].demand > vehicleCapacity) continue;

                    // Thử chèn vào từng vị trí, lấy vị trí đầu tiên feasible
                    for (int pos = 0; pos <= target.size(); pos++) {
                        List<Integer> candidate = new ArrayList<>(target);
                        candidate.add(pos, c);
                        if (isFeasible(candidate)) {
                            target.add(pos, c);
                            inserted = true;
                            break;
                        }
                    }
                    if (inserted) break;
                }
                if (inserted) {
                    routes.remove(i);   // xoá tuyến 1‑customer vì đã chèn được
                }
            }
        }

        return routes;
    }

    // ── Fitness / cost evaluation ─────────────────────────────────────────────
    public double evalRoutes(List<List<Integer>> routes) {
        if (routes == null || routes.isEmpty()) return Double.MAX_VALUE;
        double totalDistance = 0.0;
        for (List<Integer> sub : routes) {
            int prev = 0;
            for (int c : sub) {
                totalDistance += dist[prev][c];
                prev = c;
            }
            totalDistance += dist[prev][0];
        }
        return totalDistance;
    }

    public double evalPerm(int[] perm) {
        return evalRoutes(decode(perm));
    }

    // ── Pure Euclidean distance ──────────────────────────────────────────────
    public double pureDistance(List<List<Integer>> routes) {
        return evalRoutes(routes);
    }

    public double pureDistancePerm(int[] perm) {
        return pureDistance(decode(perm));
    }

    // ── Feasibility check ────────────────────────────────────────────────────
    public boolean isFeasible(List<Integer> sub) {
        int load = 0;
        double elapsed = 0.0;
        int prev = 0;
        for (int c : sub) {
            load += customers[c].demand;
            if (load > vehicleCapacity) return false;
            double arr = elapsed + dist[prev][c];
            if (arr > customers[c].dueTime) return false;
            elapsed = Math.max(arr, customers[c].readyTime) + customers[c].serviceTime;
            prev = c;
        }
        return elapsed + dist[prev][0] <= depot.dueTime;
    }

    public boolean canAppend(List<Integer> sub, int c) {
        int load = 0;
        for (int x : sub) load += customers[x].demand;
        if (load + customers[c].demand > vehicleCapacity) return false;
        List<Integer> test = new ArrayList<>(sub);
        test.add(c);
        return isFeasible(test);
    }
}