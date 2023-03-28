import numpy as np
import math
from matplotlib import pyplot as plt
import glob
import sys
import os
import time
from plotter import *
from utils import *

from velocity_generator.ramp_profile import *
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


class FrenetPath:

    def __init__(self):
        self.T = 0
        # time
        self.t = []
        self.dt = 0

        # lateral traj in Frenet frame
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []

        # longitudinal traj in Frenet frame
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []

        # cost
        self.c_lat = 0.0
        self.c_lon = 0.0
        self.c_tot = 0.0

        # combined traj in global frame
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.kappa = []
        self.v = []
        self.a = []
        self.j = []


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
        self.planning_time = 1.6 #seconds

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
        self.max_acceleration = 5 #m/s2
        self.min_acceleration = -5
        self.target_velocity = None
        self.velocity_generator = RampGenerator(max_accel= self.max_acceleration, min_accel = self.min_acceleration)

        self.K_LAT = 1.0 
        self.K_LON = 1.0 
        self.K_Di = 20000

        self.V_MAX = 60
        self.ACC_MAX = 5 
        self.K_MAX = 30
        self.t = time.time()
    
        # self.trajectory_plot = plt.subplots(2, 2)
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

    def calc_global_paths(self, fplist):

        # transform trajectory from Frenet to Global
        for fp in fplist:

            xy = [self.refrence_path.calc_position(s_i) for s_i in fp.s]
            ref_yaw = [ self.refrence_path.calc_yaw(s_i) for s_i in fp.s]

            for i in range(len(fp.s)):
                _d = fp.d[i]
                _x = xy[i][0] + _d*math.cos(ref_yaw[i] + math.pi/2.0)
                _y = xy[i][1] + _d*math.sin(ref_yaw[i] + math.pi/2.0)
                fp.x.append(_x)
                fp.y.append(_y)

            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(np.arctan2(dy, dx))
                fp.ds.append(np.hypot(dx, dy))
            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                yaw_diff = fp.yaw[i + 1] - fp.yaw[i]
                yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
                fp.kappa.append(yaw_diff / fp.ds[i])
            # calc velocity 

            for i in range(len(fp.s)):
                vi = math.sqrt(fp.s_d[i]**2 + fp.d_d[i]**2)
                fp.v.append(vi)

            #calc acceleration
            for i in range(len(fp.s)):
                ai = math.sqrt(fp.s_dd[i]**2 + fp.d_dd[i]**2)
                fp.a.append(ai)

            #calc jerk
            for i in range(len(fp.s)):
                ji = math.sqrt(fp.s_ddd[i]**2 + fp.d_ddd[i]**2)
                fp.j.append(ji)

        return fplist



    def generate_goal_sets(self,waypoint):

        goal_sets = []
        lane_width = waypoint.lane_width
        goal_sets.append(0)
        left = waypoint.get_left_lane()
        right = waypoint.get_right_lane()

        if left is not None and left.lane_id*waypoint.lane_id >=0 and left.lane_type == carla.LaneType.Driving :
            goal_sets.append(-lane_width)
        
        if right is not None and right.lane_id*waypoint.lane_id >=0 and right.lane_type == carla.LaneType.Driving:
            goal_sets.append(lane_width)

        return goal_sets


    def run_step(self,current_state, target_s, target_d, behavior):

        frenet_paths = []
        target_speed = current_state["target_speed"]

        curr_speed = current_state["speed"]

        req_accel = (target_speed - curr_speed)/self.planning_time

        if req_accel > self.max_acceleration:
            print("Breaching maximum acceleation limits")
            req_accel = self.max_acceleration

        if req_accel < self.min_acceleration:
            print("Breaching minimum acceleation limits")
            req_accel = self.min_acceleration

        achievable_speed = curr_speed + req_accel*self.planning_time
        
        fp = FrenetPath()
        fp.dt = self.dt
        fp.T = self.planning_time
        self.lat_traj_frenet = QuinticPolynomial(current_state["d"],current_state["lat_vel"], current_state["lat_acc"],target_d,0,0,self.planning_time)

        fp.t = [t for t in np.arange(0.0, self.planning_time + self.dt , self.dt)]
        # print(fp.t)
        fp.d = [self.lat_traj_frenet.calc_pos(t) for t in fp.t]
        fp.d_d = [self.lat_traj_frenet.calc_vel(t) for t in fp.t]
        fp.d_dd = [self.lat_traj_frenet.calc_acc(t) for t in fp.t]
        fp.d_ddd = [self.lat_traj_frenet.calc_jerk(t) for t in fp.t]

        self.lon_traj_frenet = QuarticPolynomial(current_state["s"],current_state["long_vel"], current_state["long_acc"],achievable_speed,0,self.planning_time)

        fp.s = [self.lon_traj_frenet.calc_pos(t) for t in fp.t]
        fp.s_d = [self.lon_traj_frenet.calc_vel(t) for t in fp.t] 
        fp.s_dd = [self.lon_traj_frenet.calc_acc(t) for t in fp.t]
        fp.s_ddd = [self.lon_traj_frenet.calc_jerk(t) for t in fp.t]

        frenet_paths.append(fp)

        frenet_paths = self.calc_global_paths(frenet_paths)

        opt_traj = frenet_paths[0]

        superelevation = 6
        radius_feet = m_to_feet(1/max(np.array(opt_traj.kappa)))
        
        if behavior!="lane_change":
            if achievable_speed < mph_to_ms(50):
                curvature_speed = 0.5*(-0.015*radius_feet+((.015*radius_feet)**2 + 4*radius_feet*((15*superelevation/100) + 2.85))**(1/2)) * 0.44704
            else:
                curvature_speed = 0.5*(-0.03*radius_feet+((.03*radius_feet)**2 + 4*radius_feet*((15*superelevation/100) + 3.6))**(1/2)) * 0.44704
        else:
            curvature_speed = target_speed
        
        # curvature_speed = 200
        
        print("Target Speed set by Behavior Planner: ",ms_to_mph(target_speed))
        print("Achievable Target Speed set by Behavior Planner: ", ms_to_mph(achievable_speed))
        print("Speed limit due to Curvature: ",  ms_to_mph(curvature_speed))
        
        final_speed = min(achievable_speed, curvature_speed, target_speed)

        print("Final Speed: ",ms_to_mph(final_speed))

        # if use FOT speed profile
        #x,y,yaw,v = opt_traj.x , opt_traj.y, opt_traj.yaw, opt_traj.v

        # if use ramp speed profile
        ds = opt_traj.s - opt_traj.s[0]
        v, a = self.velocity_generator.plan(current_state["speed"], final_speed, ds)
        x,y,yaw = opt_traj.x , opt_traj.y, opt_traj.yaw
        
        # plot_trajectory(self.trajectory_plot ,opt_traj.t, opt_traj.x, opt_traj.y, v*np.ones((len(opt_traj.x))), a*np.ones((len(opt_traj.x))), opt_traj.j)
        return x, y, yaw, v, opt_traj.dt, opt_traj.T, final_speed

