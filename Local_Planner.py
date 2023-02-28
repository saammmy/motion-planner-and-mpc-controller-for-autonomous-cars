import numpy as np
import math
from matplotlib import pyplot as plt
import glob
import sys
import os
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

class QuinticPolynomial:

    def __init__(self, xi, vi, ai, xf, vf, af, T):
        # calculate coefficient of quintic polynomial
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T** 4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xf - self.a0 - self.a1*T - self.a2*T**2,
                      vf - self.a1 - 2*self.a2*T,
                      af - 2*self.a2])

        #print A.dtype, b.dtype
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    # calculate postition info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5 * t ** 5
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
        return j


class QuarticPolynomial:

    def __init__(self, xi, vi, ai, vf, af, T):
        # calculate coefficient of quartic polynomial
        # used for longitudinal trajectory
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[3*T**2, 4*T**3],
                             [6*T, 12*T**2]])
        b = np.array([vf - self.a1 - 2*self.a2*T,
                             af - 2*self.a2])

        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    # calculate postition info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t
        return j


class LocalPlanner:
    def __init__(self,world,sp,s,x,y, yaw, curvature):
        self.world = world
        self.planning_horizon = 20 #meters
        self.planning_duration = 2 #seconds
        self.dt = 0.2
        self.refrence_path = sp
        self.s = s
        self.x = x
        self.y = y
        self.yaw = yaw
        self.curvature = curvature
        self.lon_traj_frenet = None
        self.lon_traj_frenet = None
        self.traj_cartesian = None
        self.max_acceleration = 2 #m/s2
        self.target_velocity = None
    
    def update_refrence(self,sp,s,x,y, yaw, curvature):
        self.refrence_path = sp
        self.s = s
        self.x = x
        self.y = y
        self.yaw = yaw
        self.curvature = curvature

    def get_dist(self, x, y, _x, _y):
        return np.sqrt((x - _x)**2 + (y - _y)**2)

    def cross2(self,a:np.ndarray,b:np.ndarray)->np.ndarray:
        return np.cross(a,b)
        
    def cartesian_to_frenet(self, x, y):

        dx = [x - ix for ix in self.x]
        dy = [y - iy for iy in self.y]

        d = np.min(np.hypot(dx,dy))

        closest_index = np.argmin(np.hypot(dx,dy))

        map_vec = [self.x[closest_index+1] - self.x[closest_index], self.y[closest_index+1] - self.y[closest_index]]

        ego_vec = [x - self.x[closest_index], y - self.y[closest_index]]
        
        direction = np.sign(np.dot(map_vec,ego_vec))

        if direction >= 0:
            idx =  closest_index + 1
        else:
            idx = closest_index
        if idx == 0:
            idx = 1

        n_x = self.x[idx] - self.x[idx - 1]
        n_y = self.y[idx] - self.y[idx - 1]
        x_x = x - self.x[idx]
        x_y = y - self.y[idx]

        proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
        proj_x = proj_norm*n_x
        proj_y = proj_norm*n_y   

        d = self.get_dist(x_x,x_y,proj_x,proj_y)

        ego_vec = [x - self.x[idx-1], y - self.y[idx-1],0]
        map_vec = [n_x, n_y, 0]

        d_cross = self.cross2(ego_vec,map_vec)

        if d_cross[-1] > 0:
            d = -d
        
        s = self.s[idx-1] + self.get_dist(0,0,proj_x,proj_y)
        return s,d

    def frenet_to_cartesian(self, s, d):

        xy = self.refrence_path.calc_position(s)
        ref_yaw = self.refrence_path.calc_yaw(s)

        fx = xy[0] + d*math.cos(ref_yaw + math.pi/2.0)
        fy = xy[1] + d*math.sin(ref_yaw + math.pi/2.0)

        return fx, fy

    def calculate_global_path(self):
        
        s = [self.lon_traj_frenet.calc_pos(t) for t in np.arange(0,self.planning_duration + self.dt, self.dt)]
        d = [self.lat_traj_frenet.calc_pos(t) for t in np.arange(0,self.planning_duration + self.dt, self.dt)]
        s_d = [self.lon_traj_frenet.calc_vel(t) for t in np.arange(0,self.planning_duration + self.dt, self.dt)]
        d_d = [self.lat_traj_frenet.calc_vel(t) for t in np.arange(0,self.planning_duration + self.dt, self.dt)]


        xy = [self.refrence_path.calc_position(s_i) for s_i in s]
        ref_yaw = [ self.refrence_path.calc_yaw(s_i) for s_i in s]
        x = []
        y = []
        v = []
        yaw = []
        for i in range(len(xy)):
            fx = xy[i][0] + d[i]*math.cos(ref_yaw[i] + math.pi/2.0)
            fy = xy[i][1] + d[i]*math.sin(ref_yaw[i] + math.pi/2.0)
            x.append(fx)
            y.append(fy)

        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            yaw.append(math.atan2(dy,dx))
        yaw.append(yaw[-1])

        for i in range(len(s_d)):
            vi = math.sqrt(s_d[i]**2 + d_d[i]**2)
            v.append(vi)

        return x , y, yaw, v


    def generate_goal_sets(self,waypoint,goal_s,goal_d):

        goal_sets = []
        lane_width = waypoint.lane_width

        goal_sets.append((goal_s,goal_d))
        left = waypoint.get_left_lane()
        right = waypoint.get_right_lane()
        temp_d = 0
        while left is not None and left.get_left_lane().lane_id*left.lane_id >=0 and left.lane_type == carla.LaneType.Driving :
            temp_d = goal_d - lane_width
            goal_sets.append((goal_s,temp_d))
            left = left.get_left_lane()
        
        temp_d = 0
        while right is not None and right.get_right_lane().lane_id*right.lane_id >=0 and right.lane_type == carla.LaneType.Driving:
            temp_d = goal_d + lane_width
            goal_sets.append((goal_s,temp_d))
            right = right.get_right_lane()

        return goal_sets


    def run_step(self,current_state,target_lon_vel):

        # Calculate current s and d
        current_s,current_d = self.cartesian_to_frenet(current_state['x'], current_state['y'])
        self.planning_horizon = current_state["speed"]*self.planning_duration + (self.max_acceleration*self.planning_duration**2)/2.0
        self.planning_horizon = 20
        goal_s = current_s + self.planning_horizon
        goal_d = 0

        self.lon_traj_frenet = QuarticPolynomial(round(current_s,2),current_state["long_vel"],
                                current_state["long_acc"],target_lon_vel,0,self.planning_duration)

        self.lat_traj_frenet = QuinticPolynomial(round(current_d,2),current_state["lat_vel"],
                                current_state["lat_acc"],goal_d,0,0,self.planning_duration)

        x , y, yaw, v = self.calculate_global_path()

        return x, y, yaw, v