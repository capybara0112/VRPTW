package meaf;

/**
 * Represents a single customer (or depot) in the VRPTW instance.
 */
public class Customer {
    public final int    id;
    public final double x, y;
    public final int    demand;
    public final int    readyTime;
    public final int    dueTime;
    public final int    serviceTime;

    public Customer(int id, double x, double y,
                    int demand, int readyTime, int dueTime, int serviceTime) {
        this.id          = id;
        this.x           = x;
        this.y           = y;
        this.demand      = demand;
        this.readyTime   = readyTime;
        this.dueTime     = dueTime;
        this.serviceTime = serviceTime;
    }
}