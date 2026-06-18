#!/usr/bin/env python

import os
import select
import sys
import rclpy
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from rclpy.node import Node
import numpy as np
import heapq
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile
import time
#from time import time
#from time import ctime
import csv
import threading
from geometry_msgs.msg import PoseWithCovarianceStamped
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84
LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

msg = """
CAU AISL EXPERIMENT
Press 'o' to start the experiment
Press 'p' to pause the experiment
Press 'q' to quit
"""
e = """
Communications Failed
"""

#Turtlebot Dynamics region
def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key
def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))
def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input

    return output
def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel
def check_linear_limit_velocity(velocity):
    return constrain(velocity, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
def check_angular_limit_velocity(velocity):
    return constrain(velocity, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
 
################################### 학생들이 작성할 부분 ###############################
############# Guidance: Reference Path(Points) Generation Class ############
# 여기에 Reference Path(Points)를 형성하는 Class를 작성하세요.

def concat_paths(*segments, atol=1e-9):
    segs = [np.asarray(s, dtype=float) for s in segments if len(s) > 0]
    if not segs:
        return np.empty((0, 2), dtype=float)
    out = segs[0]
    for seg in segs[1:]:
        if np.allclose(out[-1], seg[0], atol=atol):
            out = np.vstack([out, seg[1:]])
        else:
            out = np.vstack([out, seg])
    return out

def line(x0, y0, x1, y1, ds):
    P0 = np.array([x0, y0]); P1 = np.array([x1, y1])
    L = np.linalg.norm(P1 - P0)
    n = max(2, int(np.ceil(L/ds)))
    t = np.linspace(0, 1, n, endpoint=True)
    XY = (1-t)[:,None]*P0 + t[:,None]*P1
    return XY

def arc(cx, cy, R, th0, th1, ds):
    dtheta = ds / max(R, 1e-9)
    if th1 >= th0: thetas = np.arange(th0, th1,  dtheta)
    else:          thetas = np.arange(th0, th1, -dtheta)
    thetas = np.append(thetas, th1)
    x = cx + R*np.cos(thetas)
    y = cy + R*np.sin(thetas)
    return np.stack([x, y], axis=1)

def build_path(ds=0.01, R=0.3, straight=0.3):
    pts1 = line(0.0, 0.0, 0.0, straight, ds)
    pts2 = arc( R, straight , R, np.pi, 0, ds)
    pts3 = line( 2*R, straight, 2*R, 0, ds)
    pts4 = arc( 3*R, 0, R, -np.pi, 0, ds )
    pts5 = line( 4*R, 0, 4*R, straight, ds)
    pts = concat_paths(pts1, pts2, pts3, pts4, pts5)
    
    return pts  # (N,2)

def rasterize_path_to_grid(path_xy, res=0.01, margin=0.15, corridor_width=0.10,
                           as_costmap=False, cost_free=1, cost_outside=220, cost_obstacle=255):
    xmin, ymin = path_xy.min(axis=0) - margin
    xmax, ymax = path_xy.max(axis=0) + margin
    Wm, Hm = (xmax - xmin), (ymax - ymin)
    grid_w = int(np.ceil(Wm / res))
    grid_h = int(np.ceil(Hm / res))
    ox, oy = xmin, ymin

    xs = ox + (np.arange(grid_w) + 0.5) * res
    ys = oy + (np.arange(grid_h) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)
    min_d2 = np.full((grid_h, grid_w), np.inf, dtype=np.float32)
    P = path_xy
    for i in range(len(P)-1):
        x0,y0 = P[i]; x1,y1 = P[i+1]
        vx,vy = x1-x0, y1-y0
        seg_len2 = vx*vx + vy*vy + 1e-12
        t = ((XX - x0)*vx + (YY - y0)*vy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t*vx; proj_y = y0 + t*vy
        d2 = (XX - proj_x)**2 + (YY - proj_y)**2
        min_d2 = np.minimum(min_d2, d2)

    half_w = corridor_width * 0.5
    corridor_mask = (min_d2 <= (half_w*half_w))
    if as_costmap:
        cost = np.full((grid_h, grid_w), cost_outside, np.uint8)
        dist = np.sqrt(min_d2, dtype=np.float32)
        alpha = np.clip(dist / half_w, 0.0, 1.0)
        cost[corridor_mask] = (alpha[corridor_mask]*cost_outside + (1-alpha[corridor_mask])*cost_free).astype(np.uint8)
        cost[0,:]=cost_obstacle; cost[-1,:]=cost_obstacle
        cost[:,0]=cost_obstacle; cost[:,-1]=cost_obstacle
        return cost, dict(res=res, origin=(ox, oy))
    else:
        occ = np.ones((grid_h, grid_w), np.uint8)
        occ[corridor_mask] = 0
        return occ, dict(res=res, origin=(ox, oy))

# ---- A* / 좌표 변환 / 재표본화 ----
def astar_occ(occ, start, goal):
    H, W = occ.shape
    def inb(r,c): return 0 <= r < H and 0 <= c < W
    if occ[start] or occ[goal]: raise ValueError("start/goal must be free")
    moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
    def h(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])
    g = np.full((H,W), np.inf, np.float32)
    came = np.full((H,W,2), -1, np.int32)
    vis = np.zeros((H,W), bool)
    g[start] = 0.0
    from heapq import heappush, heappop
    pq = [(h(start,goal), 0.0, start)]
    while pq:
        fcur, gcur, cur = heappop(pq)
        if vis[cur]: continue
        vis[cur] = True
        if cur == goal:
            path = []
            n = cur
            while n[0] != -1:
                path.append(n)
                pr, pc = came[n]
                if pr == -1: break
                n = (pr, pc)
            path.reverse()
            return path
        r,c = cur
        for dr,dc,cost in moves:
            nr,nc = r+dr, c+dc
            if not inb(nr,nc) or occ[nr,nc]: continue
            ng = gcur + cost
            if ng < g[nr,nc]:
                g[nr,nc] = ng
                came[nr,nc] = (r,c)
                heappush(pq, (ng + h((nr,nc),goal), ng, (nr,nc)))
    return []

def ij_to_world(path_ij, res, origin):
    if not path_ij: return np.array([]), np.array([])
    rows = np.array([p[0] for p in path_ij], float)
    cols = np.array([p[1] for p in path_ij], float)
    xs = origin[0] + (cols + 0.5)*res
    ys = origin[1] + (rows + 0.5)*res
    return xs, ys

def world_to_index(x, y, res, origin):
    col = (x - origin[0]) / res - 0.5
    row = (y - origin[1]) / res - 0.5
    return int(round(row)), int(round(col))

def resample_equal(x, y, ds=0.05):
    if len(x) < 2: return x, y
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    S = np.arange(0.0, s[-1]+1e-9, ds)
    xi = np.interp(S, s, x)
    yi = np.interp(S, s, y)
    return xi, yi

def compute_yaws(xs, ys):
    dx = np.gradient(xs); dy = np.gradient(ys)
    return np.unwrap(np.arctan2(dy, dx))

def compute_curvature(xs, ys):
    dx = np.gradient(xs); dy = np.gradient(ys)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5 + 1e-9
    return num / den
# ---- (2) 컨트롤러/추정기 ----
class VelocityEstimator:
    def __init__(self):
        self._prev_t = None; self._prev = None
    def update(self, x, y, yaw, tnow):
        if self._prev_t is None:
            self._prev_t = tnow; self._prev = (x,y,yaw); return 0.0, 0.0, 0.0
        dt = max(1e-3, tnow - self._prev_t)
        vx = (x - self._prev[0])/dt; vy = (y - self._prev[1])/dt
        v = float(np.hypot(vx, vy))
        dyaw = (yaw - self._prev[2] + np.pi)%(2*np.pi) - np.pi
        w = float(dyaw/dt)
        self._prev_t = tnow; self._prev = (x,y,yaw)
        return v, w, dt

class SimpleRef:
    """rx, ry를 받아 yaw를 미리 계산해두는 얇은 래퍼"""
    def __init__(self, xs, ys):
        self.x = np.asarray(xs, float); self.y = np.asarray(ys, float)
        self.yaw = compute_yaws(self.x, self.y)
    def nearest_index(self, x, y, last_idx=0, win_back=20, win_ahead=60):
        i0 = max(last_idx - win_back, 0)
        i1 = min(last_idx + win_ahead, len(self.x))
        dx = self.x[i0:i1] - x; dy = self.y[i0:i1] - y
        local = int(np.argmin(dx*dx + dy*dy))
        return i0 + local

def _wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def align_heading_once(pub, imuodom_subscriber, ref_xs, ref_ys,
                       tol_deg=5.0, w_gain=1.5, w_max=0.6, timeout=3.0,
                       hz=50.0):
    """
    ref_xs, ref_ys의 초반 접선(경로 진행각)에 현재 yaw를 맞춘 뒤 출발.
    tol_deg 안에 0.2 s 이상 유지되면 정렬 완료 처리.
    """
    tol = math.radians(tol_deg)
    yaw_ref0 = math.atan2(ref_ys[1] - ref_ys[0], ref_xs[1] - ref_xs[0])  
    t0 = time.time()
    stable_cnt = 0
    period = 1.0 / hz

    while True:
        yaw = imuodom_subscriber.yaw
        err = _wrap(yaw_ref0 - yaw)
        w_cmd = float(np.clip(w_gain * err, -w_max, w_max))

        tw = Twist()
        tw.linear.x = 0.0           
        tw.angular.z = w_cmd
        pub.publish(tw)

        stable_cnt = stable_cnt + 1 if abs(err) < tol else 0
        
        if stable_cnt >= int(0.2 / period):
            break
        
        if (time.time() - t0) > timeout:
            break

        time.sleep(period)

    
    pub.publish(Twist())
    time.sleep(0.1)


class Stanley:
    """
    Diff-drive path follower (Stanley)
    - 최근접점 윈도우 탐색 + look-ahead
    - 횡오차 부호 뒤집기 옵션(flip_cross_sign)
    - last_idx 내부 보관(상태ful)
    """
    def __init__(self,
                 k=2.2, ks=0.15, k_heading=0.9,
                 look_ahead_pts=1, omega_max=1.2,
                 flip_cross_sign=False,
                 win_back=20, win_ahead=60):
        self.k = k
        self.ks = ks
        self.k_heading = k_heading
        self.look_ahead_pts = look_ahead_pts
        self.omega_max = omega_max
        self.flip = flip_cross_sign
        self.win_back = win_back
        self.win_ahead = win_ahead
        self.last_idx = 0  

    def step(self, x, y, yaw, v, ref_xs, ref_ys, ref_yaws):
        
        i0 = max(self.last_idx - self.win_back, 0)
        i1 = min(self.last_idx + self.win_ahead, len(ref_xs))
        dx = ref_xs[i0:i1] - x
        dy = ref_ys[i0:i1] - y
        local = int(np.argmin(dx*dx + dy*dy))
        idx = min(i0 + local + self.look_ahead_pts, len(ref_xs) - 1)
        self.last_idx = idx

        
        gamma = ref_yaws[idx]
        psi   = _wrap(yaw - gamma)                 
        nx, ny = -np.sin(gamma), np.cos(gamma)     
        e = (x - ref_xs[idx]) * nx + (y - ref_ys[idx]) * ny
        if self.flip: e = -e

        
        cte   = np.arctan2(self.k * e, v + self.ks)
        omega = np.clip(self.k_heading * psi + cte, -self.omega_max, self.omega_max)
        return float(omega), int(idx), float(psi), float(e)

class PID1D:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, u_min=-np.inf, u_max=np.inf, tau=0.05, Kaw=1.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_min, self.u_max = u_min, u_max
        self.tau, self.Kaw = tau, Kaw
        self.I = 0.0; self.d_f = 0.0; self.e_prev = 0.0; self.u_prev = 0.0
    def reset(self):
        self.I = self.d_f = self.e_prev = self.u_prev = 0.0
    def update(self, sp, meas, dt):
        e = sp - meas
        if dt <= 0.0: d = 0.0
        else:
            d_raw = (e - self.e_prev)/dt
            alpha = self.tau/(self.tau + dt)
            self.d_f = alpha*self.d_f + (1-alpha)*d_raw
            d = self.d_f
        u_unsat = self.Kp*e + self.I + self.Kd*d
        u = float(np.clip(u_unsat, self.u_min, self.u_max))
        self.I += (self.Ki*e + self.Kaw*(u - u_unsat)) * dt
        self.e_prev = e; self.u_prev = u
        return u


######################################################################################

#Sensor region
spin_thread=None
stop_event = threading.Event()
class ImuOdomSubscriber(Node):
    def __init__(self):
        super().__init__('imu_odom_subscriber')
        #print('Activate Initialization')
        # self.imu_seubscription = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        
        #print('End Initialization')
        self.mapPose_subscription = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 1)
        self.pose = np.array([0.0,0.0])
        self.yaw=0
        
    def euler_from_quaternion(self, z, w):
        t3 = +2.0 * (w * z )
        t4 = +1.0 - 2.0 * ( z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians
            
    def pose_callback(self, msg):
        pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.pose[0] = pose.x
        self.pose[1] = pose.y
        #print(f'Position: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})')
        #print(f'Orientation : ({orientation.x:.3f}, {orientation.y:.3f}, {orientation.z:.3f}, {orientation.w:.3f}')

        self.yaw = self.euler_from_quaternion(orientation.z,orientation.w)
def spin_threadfunction(node):
    while not stop_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.01)
    rclpy.shutdown()

def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)
    rclpy.init()

    qos = QoSProfile(depth=10)
    imuodom_subscriber = ImuOdomSubscriber()
    node = rclpy.create_node('sine_turtlebot')
    
    pub = node.create_publisher(Twist, 'cmd_vel', qos)

    spin_thread = threading.Thread(target=spin_threadfunction ,args=(imuodom_subscriber,))
    spin_thread.start()

    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0


    ################################ 학생들이 작성할 부분 #################################
    ################################ Reference Path 형성 #################################
    # 위의 정의된 Reference Path Class를 호출하여서 Reference Path를 생성하도록 코드를 작성하세요.
    path_world = build_path(ds=0.01, R=0.30, straight=0.30)   
    occ, meta   = rasterize_path_to_grid(path_world, res=0.01, margin=0.15, corridor_width=0.10, as_costmap=False)


    grid_start = world_to_index(path_world[0,0],  path_world[0,1],  meta["res"], meta["origin"])
    grid_goal  = world_to_index(path_world[-1,0], path_world[-1,1], meta["res"], meta["origin"])

    path_ij = astar_occ(occ, grid_start, grid_goal)
    if not path_ij:
        raise RuntimeError("A* failed: no path from start to goal in corridor")
    px, py  = ij_to_world(path_ij, meta["res"], meta["origin"])

    
    rx, ry = resample_equal(px, py, ds=0.05)
    rx, ry = ry, -rx
    ref    = SimpleRef(rx, ry)  # yaw 계산 내장

    
    ref_xs, ref_ys, ref_yaws = ref.x, ref.y, ref.yaw
    kappa_ref = compute_curvature(ref_xs, ref_ys)

    ######################## Waypoint와 Obstacle 좌표를 입력하세요 ########################
    #Waypoint=[]
    #Obstacle=[]
    ########################## PID Controller Initialization #############################
    # 위의 정의된 PID Controller를 호출하여서 PID Controller를 초기화하세요.
    stanley = Stanley(k=2.2, ks=0.15, k_heading=0.9,
                      look_ahead_pts=1, omega_max=min(1.0, BURGER_MAX_ANG_VEL),
                      flip_cross_sign=False)

    pid_v = PID1D(Kp=0.8, Ki=0.00, Kd=0.04, u_min=-0.20, u_max=0.20, tau=0.05, Kaw=1.0)
    pid_w = PID1D(Kp=1.0, Ki=0.10, Kd=0.02, u_min=-0.60, u_max=0.60, tau=0.05, Kaw=1.0)
    vel_est = VelocityEstimator()

    
    V_PLAN = 0.08
    V_MAX  = BURGER_MAX_LIN_VEL
    W_MAX  = min(1.0, BURGER_MAX_ANG_VEL)
    ALPHA_FF = 0.6 
    

    try:
        print(msg)
        start = False
        
        startTime =time.time()
        with open('test.csv',mode='w', newline='') as file:
            writer= csv.writer(file)
            writer.writerow(['starttime','time','desired position','present position','present heading angle']) 
            while True:
                key = get_key(settings)

                if key == 'o':
                    print("Starting")
                    align_heading_once(pub, imuodom_subscriber, ref_xs, ref_ys,
                                    tol_deg=5.0, w_gain=1.5, w_max=0.6, timeout=3.0, hz=50.0)

                    
                    stanley.last_idx = int(ref.nearest_index(imuodom_subscriber.pose[0],
                                                            imuodom_subscriber.pose[1]))

                    
                    warmup_t = time.time()
                    stanley.look_ahead_pts = 0   
                    V_PLAN = 0.05                
                    ALPHA_FF = 0.0               

                    start = True
                    startTime = time.time()
            

                elif key == 'p':
                    print("Paused by user input.")
                    start = False
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    pub.publish(twist)

                elif key == 'q':
                    print("Quitting...")
                    stop_event.set()
                    return
                
                if start:
                    robot_position = np.array([imuodom_subscriber.pose[0], imuodom_subscriber.pose[1]])
                    robot_yaw = imuodom_subscriber.yaw
                    
                   
                    tnow = time.time()
                    v_meas, w_meas, dt_est = vel_est.update(
                        robot_position[0], robot_position[1], robot_yaw, tnow
                    )
                    dt_ctrl = max(0.03, min(0.10, dt_est))  
                    if 'warmup_t' in locals():
                        if time.time() - warmup_t > 2.0:     
                            stanley.look_ahead_pts = 1
                            V_PLAN = 0.08
                            ALPHA_FF = 0.6
                            del warmup_t

                    
                    omega_d, idx, psi, e = stanley.step(
                        robot_position[0], robot_position[1], robot_yaw,
                        max(v_meas, 1e-3), 
                        ref_xs, ref_ys, ref_yaws
                    )
                    
                    v_d = V_PLAN
                    goal_xy = np.array([ref_xs[-1], ref_ys[-1]])
                    dist_to_goal = float(np.linalg.norm(robot_position - goal_xy))

                    
                    if abs(psi) > np.deg2rad(20):
                        v_d = 0.06
                    else:
                        kap = abs(kappa_ref[idx])
                        if   kap > 2.0: v_d = 0.06
                        elif kap > 1.0: v_d = 0.07
                        else:           v_d = 0.09


                    if ALPHA_FF != 0.0:
                        omega_d = np.clip(
                            omega_d + ALPHA_FF * v_d * float(kappa_ref[idx]),
                            -W_MAX, W_MAX
                        )

                    
                    u_v = pid_v.update(v_d, v_meas, dt_ctrl)
                    u_w = pid_w.update(omega_d, w_meas, dt_ctrl)

                    target_linear_velocity  = float(np.clip(v_d + u_v, -V_MAX, V_MAX))
                    target_angular_velocity = float(np.clip(omega_d + u_w, -W_MAX, W_MAX))

                    
                    desired_position = np.array([ref_xs[idx], ref_ys[idx]]) 
                    goal_xy = np.array([ref_xs[-1], ref_ys[-1]])
                    dist_to_goal = float(np.linalg.norm(robot_position - goal_xy))
                    GOAL_TOL = 0.05   

                    if dist_to_goal < GOAL_TOL:
                        print("Goal reached. Stopping.")
                        start = False                 # 메인 루프 논리 플래그 내려서 제어 중단
                        pub.publish(Twist())          # 즉시 정지 명령
                        continue   
                    
                    #csv 저장
                    writer.writerow([startTime,time.time(),str(desired_position),str(robot_position),str(robot_yaw)]) 
                    
                    print(f"position : {robot_position}, yaw : {str(robot_yaw)}, desired : {desired_position}, controll : {(target_linear_velocity,target_angular_velocity)}")

                    # 속도 프로파일 및 제어 명령 생성
                    control_linear_velocity = make_simple_profile(control_linear_velocity, target_linear_velocity, (LIN_VEL_STEP_SIZE / 2.0))
                    control_angular_velocity = make_simple_profile(control_angular_velocity, target_angular_velocity, (ANG_VEL_STEP_SIZE / 2.0))

                    twist = Twist()
                    twist.linear.x = float(control_linear_velocity)
                    twist.angular.z = float(control_angular_velocity)
                    pub.publish(twist)

    except Exception as ex:
        print(ex)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    main()
