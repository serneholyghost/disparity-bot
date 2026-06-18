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

class PathGenerator:
    def __init__(self, Waypoint=None):
        if Waypoint is not None:
            self.path = np.array(Waypoint, dtype=float)
        else:
            self.path = np.array(self.generate_path(), dtype=float)

        self.target_wp_idx = 0
        self.len_path = len(self.path)

        self.goal_thresh = 0.02

    def generate_path(self):
        R       = 0.3
        res     = 0.02  
        path    = []

        for x in np.arange(0.0, R+res, res):
            path.append([x, 0.0])

        center_x_1 = R
        center_y_1 = -R
        for theta in np.arange(math.pi/2, -math.pi/2 - res/R, -res/R):
            x = center_x_1 + R * math.cos(theta)
            y = center_y_1 + R * math.sin(theta)
            path.append([x, y])

        for x in np.arange(R, 0.0, -res):
            path.append([x, -2*R])

        center_x_2 = 0.0
        center_y_2 = -3*R
        for theta in np.arange(math.pi/2, 3*math.pi/2 + res/R, res/R):
            x = center_x_2 + R * math.cos(theta)
            y = center_y_2 + R * math.sin(theta)
            path.append([x, y])

        for x in np.arange(0.0, R+0.2+res, res):
            path.append([x, -4*R])
        
        return path

    
    def get_waypoint(self, robot_position, lookahead_dist):
        if np.linalg.norm(robot_position - self.path[-1]) < self.goal_thresh:
            self.target_wp_idx = self.len_path - 1
            return self.path[-1], True

        if self.target_wp_idx >= self.len_path:
            return self.path[-1], True

        self.target_wp_idx = min(self.target_wp_idx, self.len_path - 1)

        for i in range(self.target_wp_idx, self.len_path):
            waypoint_pos = self.path[i]
            dist = np.linalg.norm(robot_position - waypoint_pos)

            if dist >= lookahead_dist:
                self.target_wp_idx = i
                return waypoint_pos, False

        self.target_wp_idx = self.len_path - 1
        return self.path[-1], True


################## Lateral Error & PID Controller Class ####################
# 여기에 Tracking Error를 계산하는 Class를 작성하세요.
class PurePursuitController:
    def __init__(self, Ld):
        self.Ld = Ld

    def calculate_curv(self, robot_position, robot_yaw, desired_position):
        delta_xg = desired_position[0] - robot_position[0]
        delta_yg = desired_position[1] - robot_position[1]

        x_L = delta_xg * math.cos(robot_yaw) + delta_yg * math.sin(robot_yaw)
        y_L = -delta_xg * math.sin(robot_yaw) + delta_yg * math.cos(robot_yaw)

        if self.Ld == 0.0:
            kappa_R = 0.0
        else:
            kappa_R = 2.0 * y_L / (self.Ld ** 2)

        return kappa_R
# 여기에 PID Controller Class를 작성하세요.
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.integral_err = 0.0
        self.prev_err = 0.0

        self.integral_limit = 1.0

    def calculate_control_output(self, err):
        self.integral_err += err * self.dt
        self.integral_err = np.clip(self.integral_err, -self.integral_limit, self.integral_limit)

        self.derivative_err = (err - self.prev_err) / self.dt

        output = (self.Kp * err) + (self.Ki * self.integral_err) + (self.Kd * self.derivative_err)

        self.prev_err = err

        return output
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
    path_generator = PathGenerator()

    ######################## Waypoint와 Obstacle 좌표를 입력하세요 ########################
    Waypoint=[]
    Obstacle=[]

    ########################## PID Controller Initialization #############################
    # 위의 정의된 PID Controller를 호출하여서 PID Controller를 초기화하세요.
    lookahead_distance = 0.35
    constant_linear_velocity = 0.10

    pure_pursuit = PurePursuitController(Ld=lookahead_distance)

    KP = 0.8
    KI = 0.05
    KD = 0.05
    DT = 0.01

    pid_controller = PIDController(Kp=KP, Ki=KI, Kd=KD, dt=DT)

    ######################################################################################

    try:
        print(msg)
        start = False
        
        startTime =time()
        prev_time = time()
        with open('test.csv',mode='w', newline='') as file:
            writer= csv.writer(file)
            writer.writerow(['starttime',
                             'time',
                             'desired position',
                             'present position',
                             'present heading angle',
                             'target_wp_idx',
                             'ctrl_ref_omg',
                             'pid_error',
                             'dist_to_goal',
                             'duration',
                             'cmd_lin_vel',
                             'cmd_ang_vel']) ######## 마음대로 수정 가능!
            while True:
                key = get_key(settings)

                if key == 'o':
                    print("Starting")
                    start = True
                    startTime =time()
                    prev_yaw = imuodom_subscriber.yaw
                    prev_time = time()

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
                    now_time = time()
                    dt = now_time - prev_time
                    prev_time = now_time
                    pid_controller.dt = dt
                    
                    robot_position = np.array([imuodom_subscriber.pose[0], imuodom_subscriber.pose[1]])
                    robot_yaw = imuodom_subscriber.yaw
                    
                    ################################ 학생들이 작성할 부분 #################################
                    desired_waypoint, path_complete = path_generator.get_waypoint(robot_position, lookahead_distance)
                    desired_position = desired_waypoint     ###### 현재 터틀봇이 목표로 하는 Desired Waypoint Position
                    
                    theta_dot_d = 0.0
                    error_input = 0.0
                    
                    if path_complete:
                        target_linear_velocity = 0.0
                        target_angular_velocity = 0.0
                    else:
                        kappa_R = pure_pursuit.calculate_curv(robot_position, robot_yaw, desired_position)
                        theta_dot_d = constant_linear_velocity * kappa_R

                        dyaw = math.atan2(math.sin(robot_yaw - prev_yaw), math.cos(robot_yaw - prev_yaw))
                        yaw_rate_meas = dyaw / dt
                        prev_yaw = robot_yaw
                        
                        error_input = theta_dot_d - yaw_rate_meas
                        angular_cmd_pid = pid_controller.calculate_control_output(error_input)

                    #################### Lateral Error & PID Controller Generation ###################
                    # 위의 정의된 Tracking Error Class를 호출하여서 Error를 Output으로 하는 코드를 작성하세요.

                    # Output으로 나온 Error를 가지고 위의 정의된 PID Controller를 호출하여서 제어명령을 수립하는 코드를 작성하세요.

                        target_linear_velocity = check_linear_limit_velocity(constant_linear_velocity)   ####### 선속도 제어명령 (등속도 명령을 인가하세요)
                        target_angular_velocity = check_angular_limit_velocity(angular_cmd_pid)           ####### 각속도 제어명령
                    ##################################################################################
                    
                    #csv 저장
                    writer.writerow([
                        startTime,
                        time(),
                        str(desired_position),
                        str(robot_position),
                        str(robot_yaw),
                        path_generator.target_wp_idx,
                        theta_dot_d,
                        error_input,
                        np.linalg.norm(robot_position - desired_position),
                        time() - startTime,
                        float(control_linear_velocity),
                        float(control_angular_velocity)
                    ])

                    
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
