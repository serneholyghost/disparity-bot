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
from time import time
from time import ctime
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
def resample_equal_arc(xs, ys, ds=0.025):
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    
    dx = np.diff(xs); dy = np.diff(ys)
    keep = np.hstack([[True], (dx*dx + dy*dy) > 1e-12])
    xs = xs[keep]; ys = ys[keep]
    if len(xs) < 2:
        return xs.copy(), ys.copy()

    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.hstack([0.0, np.cumsum(seg)])
    s_new = np.arange(0.0, s[-1] + 1e-12, ds)
    if s_new[-1] < s[-1] - 1e-9:
        s_new = np.hstack([s_new, s[-1]])
    x_new = np.interp(s_new, s, xs)
    y_new = np.interp(s_new, s, ys)
    return x_new, y_new

def arc_lookahead(xs, ys, nearest_idx, Ld, search_ahead=60): #80
    N = len(xs)
    j = nearest_idx
    s_acc = 0.0
    max_j = min(nearest_idx + search_ahead, N-2)  
    while j <= max_j:
        dx = xs[j+1]-xs[j]; dy = ys[j+1]-ys[j]
        seg = math.hypot(dx, dy)
        if s_acc + seg >= Ld:
            remain = Ld - s_acc
            t = 0.0 if seg < 1e-9 else (remain/seg)
            gx = xs[j] + t*dx
            gy = ys[j] + t*dy
            return gx, gy, j + t  
        s_acc += seg
        j += 1
    return xs[max_j+1], ys[max_j+1], (max_j+1)


def build_occ_grid(map_size_m=2.0, resolution=0.03, origin_xy=(-1.0, -1.5)):
    
    n_cells = int(round(map_size_m / resolution))
    occ = np.zeros((n_cells, n_cells), dtype=np.uint8)

    def world_to_grid(x, y):
        r = int(round((x - origin_xy[0]) / resolution))
        c = int(round((y - origin_xy[1]) / resolution))
        return r, c

    #벽1
    wall_x1, wall_x2 = -0.6, 0.50 #0.45   
    wall_y  = -0.3                 
    r1, c1 = world_to_grid(wall_x1, wall_y)
    r2, c2 = world_to_grid(wall_x2, wall_y)
    r_lo, r_hi = sorted([r1, r2])
    r_lo = max(r_lo, 0); r_hi = min(r_hi, n_cells-1)
    c1  = max(min(c1, n_cells-1), 0)
    occ[r_lo:r_hi+1, c1] = 1

    #사각형
    box_x_low,  box_x_high  = -0.30, 0.04   
    box_y_low,  box_y_high  = -1.3, -0.70  
    r_a, c_a = world_to_grid(box_x_low,  box_y_low)
    r_b, c_b = world_to_grid(box_x_high, box_y_high)
    r_lo, r_hi = sorted([r_a, r_b])
    c_lo, c_hi = sorted([c_a, c_b])
    r_lo = max(r_lo, 0); r_hi = min(r_hi, n_cells-1)
    c_lo = max(c_lo, 0); c_hi = min(c_hi, n_cells-1)
    occ[r_lo:r_hi+1, c_lo:c_hi+1] = 1

    # 벽2
    w2_x1, w2_x2 = 0.04, 0.75 #0.8, 0.84
    w2_y = -0.95
    rr1, cc1 = world_to_grid(w2_x1, w2_y)
    rr2, cc2 = world_to_grid(w2_x2, w2_y)
    r_lo, r_hi = sorted([rr1, rr2])
    r_lo = max(r_lo, 0); r_hi = min(r_hi, n_cells-1)
    cc1  = max(min(cc1, n_cells-1), 0)
    occ[r_lo:r_hi+1, cc1] = 1
    thick = 6 #2 , 1  

    for dc in range(-thick, thick+1):
        cc_thick = cc1 + dc
        if 0 <= cc_thick < n_cells:
            occ[r_lo:r_hi+1, cc_thick] = 1

    meta = {
        "resolution": resolution,
        "origin_xy": origin_xy,
        "n_cells": n_cells
    }

    return occ, meta

def world_to_grid_pt(x, y, meta):
    res = meta["resolution"]
    ox, oy = meta["origin_xy"]
    r = int(round((x - ox)/res))
    c = int(round((y - oy)/res))
    return r, c

def grid_to_world_pt(r, c, meta):
    res = meta["resolution"]
    ox, oy = meta["origin_xy"]
    x = ox + r*res
    y = oy + c*res
    return x, y

def astar_path(start_xy, goal_xy, occ, meta):
    """
    A* on grid (8방향)
    실패 시 빈 array 반환
    """
    sr, sc = world_to_grid_pt(start_xy[0], start_xy[1], meta)
    gr, gc = world_to_grid_pt(goal_xy[0],  goal_xy[1],  meta)

    n_cells = meta["n_cells"]
    def in_bounds(r,c):
        return (0 <= r < n_cells and 0 <= c < n_cells)
    def is_free(r,c):
        return occ[r,c] == 0

    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    open_heap = []
    heapq.heappush(open_heap, (0.0, (sr,sc)))
    g_cost = {(sr,sc): 0.0}
    parent = {(sr,sc): None}

    found = False
    while open_heap:
        _, (cr,cc) = heapq.heappop(open_heap)
        if (cr,cc) == (gr,gc):
            found = True
            break
        for dr,dc in moves:
            nr, nc = cr+dr, cc+dc
            if not in_bounds(nr,nc): continue
            if not is_free(nr,nc):   continue
            step_cost = math.hypot(dr,dc)
            ng = g_cost[(cr,cc)] + step_cost
            if (nr,nc) not in g_cost or ng < g_cost[(nr,nc)]:
                g_cost[(nr,nc)] = ng
                h = math.hypot(nr-gr, nc-gc)
                f = ng + h
                parent[(nr,nc)] = (cr,cc)
                heapq.heappush(open_heap, (f, (nr,nc)))

    if not found:
        print("[A*] 경로를 찾지 못했습니다.")
        return np.zeros((0,2), dtype=float)

    # backtrack
    path_cells = []
    node = (gr,gc)
    while node is not None:
        path_cells.append(node)
        node = parent[node]
    path_cells.reverse()

    xs = []
    ys = []
    for (r,c) in path_cells:
        xw, yw = grid_to_world_pt(r,c,meta)
        xs.append(xw)
        ys.append(yw)

    return np.column_stack([xs,ys])

def make_line_path(a_xy, b_xy, num_points=50):
    ax, ay = a_xy
    bx, by = b_xy
    xs = np.linspace(ax, bx, num_points)
    ys = np.linspace(ay, by, num_points)
    return np.column_stack([xs, ys])

def concat_paths(*paths):
    segs = []
    for i, p in enumerate(paths):
        if p.shape[0] == 0:
            continue

        if i>0 and segs and np.allclose(segs[-1][-1], p[0]):
            segs.append(p[1:])
        else:
            segs.append(p)
    return np.vstack(segs)

def smooth_path(xs, ys, window=3, start_fix=None, goal_fix=None):
    xs_s=[]
    ys_s=[]
    N=len(xs)
    for i in range(N):
        i0=max(0,i-window)
        i1=min(N,i+window+1)
        xs_s.append(np.mean(xs[i0:i1]))
        ys_s.append(np.mean(ys[i0:i1]))
    xs_s=np.array(xs_s)
    ys_s=np.array(ys_s)

    if start_fix is not None:
        xs_s[0]=start_fix[0]
        ys_s[0]=start_fix[1]
    if goal_fix is not None:
        xs_s[-1]=goal_fix[0]
        ys_s[-1]=goal_fix[1]

    return xs_s, ys_s
################## Lateral Error & PID Controller Class ####################
# 여기에 Tracking Error를 계산하는 Class를 작성하세요.
"""
class PurePursuitController:
    def __init__(self, lookahead_dist=0.20, idx_advance_limit=5, omega_max=1.0):
        self.Ld = lookahead_dist
        self.idx_prev = 0
        self.advance_limit = idx_advance_limit
        self.omega_max = omega_max

    def step(self, x, y, yaw, xs_ref, ys_ref, v_cmd):
        
        i0 = self.idx_prev
        i1 = min(i0 + 50, len(xs_ref)-1)
        dists = (xs_ref[i0:i1+1]-x)**2 + (ys_ref[i0:i1+1]-y)**2
        rel_best = np.argmin(dists)
        nearest_idx = i0 + rel_best

        if nearest_idx < self.idx_prev:
            nearest_idx = self.idx_prev
        self.idx_prev = nearest_idx

        goal_idx = nearest_idx
        for j in range(nearest_idx,
                       min(nearest_idx + self.advance_limit, len(xs_ref))):
            dx = xs_ref[j] - x
            dy = ys_ref[j] - y
            dist = math.hypot(dx, dy)
            if dist >= self.Ld:
                goal_idx = j
                break
        gx = xs_ref[goal_idx]
        gy = ys_ref[goal_idx]


        dx = gx - x
        dy = gy - y
        Xr =  dx*math.cos(yaw) + dy*math.sin(yaw)   
        Yr = -dx*math.sin(yaw) + dy*math.cos(yaw)    

        Ld = max(1e-3, math.hypot(Xr, Yr))
        kappa = 2.0 * Yr / (Ld*Ld)   
        omega_cmd = v_cmd * kappa    

        if omega_cmd > self.omega_max:
            omega_cmd = self.omega_max
        elif omega_cmd < -self.omega_max:
            omega_cmd = -self.omega_max

       return goal_idx, v_cmd, omega_cmd, Yr
"""    
    
class PurePursuitController:
    def __init__(self, lookahead_dist=0.12, idx_advance_limit=80, omega_max=1.0,
                 near_goal_dist=0.12):
        self.Ld = lookahead_dist
        self.idx_prev = 0
        self.advance_limit = idx_advance_limit
        self.omega_max = omega_max
        self.near_goal_dist = near_goal_dist

    def step(self, x, y, yaw, xs_ref, ys_ref, v_cmd):
        gx_goal, gy_goal = xs_ref[-1], ys_ref[-1]
        Srem = math.hypot(gx_goal - x, gy_goal - y)

        i0 = self.idx_prev
        i1 = min(i0 + 50, len(xs_ref) - 1)
        d2 = (xs_ref[i0:i1 + 1] - x)**2 + (ys_ref[i0:i1 + 1] - y)**2
        nearest_idx = i0 + int(np.argmin(d2))
        if nearest_idx < self.idx_prev:
            nearest_idx = self.idx_prev
        self.idx_prev = nearest_idx

        # 골 근처 헤딩 정렬 + 저속
        if Srem < self.near_goal_dist:
            hdg_goal = math.atan2(gy_goal - y, gx_goal - x)
            e_yaw = (hdg_goal - yaw + math.pi) % (2*math.pi) - math.pi
            omega = float(np.clip(1.8 * e_yaw, -self.omega_max, self.omega_max))
            v_eff = min(v_cmd, 0.03)
            return gx_goal, gy_goal, (len(xs_ref) - 1), v_eff, omega, 0.0

        Ld_eff = min(self.Ld, max(0.05, 0.5 * Srem))
        gx, gy, j_float = arc_lookahead(xs_ref, ys_ref, nearest_idx, Ld_eff, self.advance_limit)
        goal_idx = int(round(j_float))  

        dx = gx - x; dy = gy - y
        Xr =  dx*math.cos(yaw) + dy*math.sin(yaw)
        Yr = -dx*math.sin(yaw) + dy*math.cos(yaw)

        Ld_m = max(1e-3, math.hypot(Xr, Yr))
        kappa = 2.0 * Yr / (Ld_m * Ld_m)

        v_eff = v_cmd / (1.0 + 0.8 * abs(kappa))
        v_eff = float(np.clip(v_eff, 0.02, v_cmd))  
        omega = float(np.clip(v_eff * kappa, -self.omega_max, self.omega_max))

        
        return gx, gy, goal_idx, v_eff, omega, float(Yr)

# 여기에 PID Controller Class를 작성하세요.
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0,
                 output_limit=None):
    
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.output_limit = output_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time  = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time  = None

    def update(self, error, current_time):


        if self.prev_time is None:
            dt = 0.0
        else:
            dt = max(1e-6, current_time - self.prev_time)

        self.integral += error * dt

        derivative = 0.0
        if dt > 0.0:
            derivative = (error - self.prev_error) / dt

        u = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

        if self.output_limit is not None:
            limit = self.output_limit
            if u >  limit: u =  limit
            if u < -limit: u = -limit

        self.prev_error = error
        self.prev_time  = current_time

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
    """
    START = (0.0, 0.0)
    MID   = (0.5, 0.07)  #(0.4,0.1 성공)      
    WP1   = (0.8, 0.0)  #(0.85)
    WP2   = (0.13, -0.6)  #0.2,-0.6
    GOAL  = (0.3, -1.2)

    occ, meta = build_occ_grid(map_size_m=2.0, resolution=0.05)

    p0 = astar_path(START, MID, occ, meta)
    if p0.shape[0] == 0:
        p0 = make_line_path(START, MID, num_points=50)

    p1 = astar_path(MID, WP1, occ, meta)
    if p1.shape[0] == 0:
        p1 = make_line_path(MID, WP1, num_points=50)

    p2 = astar_path(WP1, WP2, occ, meta)
    if p2.shape[0] == 0:
        p2 = make_line_path(WP1, WP2, num_points=50)

    p3 = astar_path(WP2, GOAL, occ, meta)
    if p3.shape[0] == 0:
        p3 = make_line_path(WP2, GOAL, num_points=50)

    full = concat_paths(p0, p1, p2, p3)
    """
    

    START = (0.0, 0.0)
    MID   = (0.5, 0.07) #0.4, 0.1 (성공),
    WP1   = (0.75, 0.0) 
    MID1   = (0.6, -0.3)
    WP2   = (0.13, -0.6)  #0.2,-0.6
    GOAL  = (0.3, -1.2)

    
    occ, meta = build_occ_grid(map_size_m=2.0, resolution=0.03)

    p0 = astar_path(START, MID, occ, meta)
    if p0.shape[0] == 0:
        p0 = make_line_path(START, MID, num_points=50)

    p1 = astar_path(MID, WP1, occ, meta)
    if p1.shape[0] == 0:
        p1 = make_line_path(MID, WP1, num_points=50)

    p2 = astar_path(WP1, MID1, occ, meta)
    if p2.shape[0] == 0:
        p2 = make_line_path(WP1, MID1, num_points=50)

    p3 = astar_path(MID1, WP2, occ, meta)
    if p3.shape[0] == 0:
        p3 = make_line_path(WP1, WP2, num_points=50)

    p4 = astar_path(WP2, GOAL, occ, meta)
    if p4.shape[0] == 0:
        p4 = make_line_path(WP2, GOAL, num_points=50)

    full = concat_paths(p0, p1, p2, p3, p4)



    xs_raw = full[:,0]
    ys_raw = full[:,1]


#    xs_ref, ys_ref = smooth_path(
#        xs_raw, ys_raw,
#        window=1,
#        start_fix=START,
#        goal_fix=GOAL
#    )
    # 스무딩 → 등호장 재샘플
    xs_s, ys_s = smooth_path(xs_raw, ys_raw, window=2,
                            start_fix=START, goal_fix=GOAL)
    xs_ref, ys_ref = resample_equal_arc(xs_s, ys_s, ds=0.025)


#    pp = PurePursuitController(
#        lookahead_dist=0.1,  #0.1
#        idx_advance_limit=30,
#        omega_max=1.0
#    )
    
    pp = PurePursuitController(
        lookahead_dist=0.12,
        idx_advance_limit=60, #80
        omega_max=1.0,
        near_goal_dist=0.10
    )
    ######################## Waypoint와 Obstacle 좌표를 입력하세요 ########################
    Waypoint=[]
    Obstacle=[]
    ########################## PID Controller Initialization #############################
    # 위의 정의된 PID Controller를 호출하여서 PID Controller를 초기화하세요.
    pid_w = PIDController(
        Kp=0.8,    
        Ki=0.0,    
        Kd=0.02,    
        output_limit=0.5  # rad/s
    )

    ######################################################################################
    V_BASE   = 0.06   
    GOAL_TOL = 0.03 
    
    try:
        print(msg)
        start = False
        
        startTime =time()
        with open('test.csv',mode='w', newline='') as file:
            writer= csv.writer(file)
            writer.writerow(['starttime','time','desried position','present position','present heading angle']) ######## 마음대로 수정 가능!
            while True:
                key = get_key(settings)

                if key == 'o':
                    print("Starting")
                    start = True
                    startTime =time()

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
                    

#                    goal_idx, v_cmd, omega_cmd, lat_err = pp.step(
#                        robot_position[0],   # x
#                        robot_position[1],   # y
#                        robot_yaw,           # yaw 
#                        xs_ref,              # ref x
#                        ys_ref,              # ref y
#                        V_BASE               
#                    )
#                    desired_position = np.array([xs_ref[goal_idx], ys_ref[goal_idx]])  
                    
                    
                    gx, gy, goal_idx, v_cmd, omega_cmd, lat_err = pp.step(
                        robot_position[0], robot_position[1], robot_yaw,
                        xs_ref, ys_ref, V_BASE
                    )
                    desired_position = np.array([gx, gy]) 


                
                    goal_xy = np.array([xs_ref[-1], ys_ref[-1]])
                    dist_to_goal = float(np.linalg.norm(robot_position - goal_xy))
                    if dist_to_goal < GOAL_TOL:
                        print("Goal reached. Stopping.")
                        start = False
                        twist = Twist()
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        pub.publish(twist)
                        continue     
                    ################################ 학생들이 작성할 부분 #################################
                    #################### Lateral Error & PID Controller Generation ###################
                    # 위의 정의된 Tracking Error Class를 호출하여서 Error를 Output으로 하는 코드를 작성하세요.

                    # Output으로 나온 Error를 가지고 위의 정의된 PID Controller를 호출하여서 제어명령을 수립하는 코드를 작성하세요.
                    delta_w = pid_w.update(lat_err, time())
                    omega_cmd_corrected = omega_cmd + delta_w
                    target_linear_velocity =  check_linear_limit_velocity(v_cmd)
                    target_angular_velocity = check_angular_limit_velocity(omega_cmd_corrected)
                    ##################################################################################
                    
                    #csv 저장
                    #writer.writerow([startTime,time(),str(desired_position),str(robot_position),str(robot_yaw)])
                    writer.writerow([
                        startTime,
                        time(),     
                        str(desired_position),        
                        str(robot_position),          
                        str(robot_yaw),               
                        str(lat_err),                
                        str(dist_to_goal),            
                        str(control_linear_velocity), 
                        str(control_angular_velocity) 
                    ])
                    #print(f"position : {robot_position}, yaw : {str(robot_yaw)}, desired : {desired_position}, controll : {(target_linear_velocity,target_angular_velocity)}")
                    print(
                        f"pos:{robot_position} yaw:{robot_yaw} "
                        f"des:{desired_position} "
                        f"err(lat={lat_err:.3f}) "
                        f"goal_dist:{dist_to_goal:.2f} "
                        f"target_cmd:({target_linear_velocity:.2f},{target_angular_velocity:.2f}) "
                        f"sent_cmd:({control_linear_velocity:.2f},{control_angular_velocity:.2f})"
                    )
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
