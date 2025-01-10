class ElevatorSystem{
    List<Elevator> elevators;
    Date datatime;
    private static final ElevatorSystem instance = new ElevatorSystem()
    private ElevatorSystem(){
        elevators = new ArrayList<>();
    }
    public static ElevatorSystem getInstance(){
        return instance
    }
    public void addElevator(Elevator e){
        elevators.add(e);
    }
    public Elevator assignElevator(Request req){
        Elevator res = elevators.get(0);
        for (Elevator tmp: elevators){
            if (tmp.status == Direction.IDLE) return tmp;
            else if (tmp.checkWorkload() < res.checkWorkload()) res = tmp;
        }
        return res;
    }
}

class Elevator{
    int currFloor;
    Direction status;
    int maxFloor;
    int maxWeight;
    HashSet<Integer> upQueue = new HashSet<>();
    HashSet<Integer> downQueue = new HashSet<>();
    public Elevator(int maxFloor, int maxWeight){
        this.maxFloor = maxFloor;
        this.maxWeight = maxWeight;
        currFloor = 1;
        status = Direction.IDLE
    }
    public void handleRequest(Request req){
        if (req.dir == Direction.DOWN){
            downQueue.add(req.floor);
        }
        else {
            upQueue.add(req.floor);
        }
    }
    public int checkWorkload(){
        return downQueue.size() + upQueue.size();
    }
    public boolean run(){
        try{
            // status refresh
            // when reach 1st floor then switch status to up
            // when reach top floor than switch status to down
            // if status is down but not req in downQ then switch status to UP
            // if status is up but not req in upQ then swith status to Down
            // if no req then switch to IDLE
            if (currFloor==1 && status==Direction.DOWN) {
                status = Direction.UP;
            } else if (currFloor==maxFloor && status==Direction.UP) {
                status = Direction.DOWN;
            } else if (checkWorkLoad()==0) {
                status = Direction.IDLE;
            } else if (upQueue.size()==0) {
                status = Direction.DOWN;
            } else if (downQueue.size()==0) {
                status = Direction.UP;
            }
            // move to next stop
            // if status is down then find next stop and move
            // if status is up then find next stop and move
            if (status == Direction.DOWN) {
                for (int i = currFloor; i>=1; i--) {
                    if (downQueue.contains(i)) {
                        downQueue.remove(i);
                        currFloor = i;
                        return true;
                    }
                }
            } else if (status == Direction.UP) {
                for(int i = currFloor; i<=maxFloor; i++) {
                    if (upQueue.contains(i)) {
                        upQueue.remove(i);
                        currFloor = i;
                        return true;
                    }
                }
            }
        }
        catch (Exception ex){
            System.lout.println("alert!call office")
        }
    }
    return false
}

enum Direction{
    UP, DOWN, IDLE
}

abstract class Request{
    int floor;
    Direction dir;
}

class ExternalRequest extends Request {
    int floor;
    Direction dir;
    public ExternalRequest(int f, Direction d){
        floor = f;
        dir = d;
    }
}

class InternalRequest extends Request {
    int floor;
    Direction dir;
    public ExternalRequest(int f, Direction d){
        floor = f;
        dir = d;
    }
}

class Passenger {
    ElevatorSystem instance = ElevatorSystem.getInstance();
    Elevator assignElv = null;
    public boolean pressExtButton(int floor, Direction dir){
        assignElv = instance.assignElevator(new ExternalRequest(floor, dir));
        return true;
    }
    public boolean pressIntButton(int floor){
        Direction dir = Direction.IDLE;
        if (floor > assignElv.currFloor){
            assignElv.handleRequest(new InternalRequest(floor, Direction.UP));
        }
        else if (floor > assignElv.currFloor){
            assignElv.handleRequest(new InternalRequest(floor. Direction.DOWN))
        }
        else return false;
        retun true
    }
}

/*
根据需求来创建相应函数
电梯系统类 - ElevatorSystem
将外部请求调度给系统中的一架电梯 - assignElevator(Request)
电梯类 - Elevator
处理按键请求 - handleRequest(Request)
根据所有请求运行电梯 - runElevator()
(非必要) 乘客类 - Passenger
按外部按键 - pressExtButton()
按内部按键 - pressIntButton()

加分要求
引入Enum类来定义常量状态
引入单例设计模式来定义ElevatorSystem类
增加电梯种类(货运电梯，载人电梯，VIP电梯) 不同电梯有不同策略
优化电梯调度
闲置电梯优先
如果没有闲置，选取请求数最少的电梯
优化电梯请求处理业务
如果非高峰期，可以采用FIFO方式处理请求
如果达到高峰期，可以根据当前电梯状态按照楼层升序处理相同状态的所有请求
根据电梯工作高峰期来使用不同的请求处理算法，使用Strategy设计模式     
*/