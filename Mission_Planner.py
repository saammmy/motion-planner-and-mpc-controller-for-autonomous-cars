
from spline import Spline2D
import math

import os
import sys
import glob

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import numpy as np
from params import *
from utils import *



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


class RefrencePath:
    def __init__(self, ds=0.5):
        self.refrencePath = None
        self.ds = ds
        self.s = []
        self.x = []
        self.y = []
        self.yaw = []
        self.curv = []
        self.direction = []
        self.waypoints = []
        self.waypoints_x = []
        self.waypoints_y = []

    def frenet_to_cartesian(self, s, d):

        xy = self.refrencePath.calc_position(s)
        ref_yaw = self.refrencePath.calc_yaw(s)

        fx = xy[0] + d*math.cos(ref_yaw + math.pi/2.0)
        fy = xy[1] + d*math.sin(ref_yaw + math.pi/2.0)

        return fx, fy

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
    
    def get_dist(self, x, y, _x, _y):
        return np.sqrt((x - _x)**2 + (y - _y)**2)

    def cross2(self,a:np.ndarray,b:np.ndarray)->np.ndarray:
        return np.cross(a,b)

class MissionPlanner:
    def __init__(self, world, Ego, global_start_point, global_end_point):

        self.world = world
        self.Ego = Ego
        self.end_point = global_end_point
        self.dao = GlobalRoutePlannerDAO(world.get_map(), 3)   
        self.grp = GlobalRoutePlanner(self.dao)
        self.grp.setup()
        self.planned_waypoints = None
        self.refrence_path_global = RefrencePath()
        self.refrence_path_local = RefrencePath()
        self.current_state = None
        self.s_future = LOCAL_GLOBAL_PLAN_MIN
        self.s_last_wrt_global = 0

        self.initial_path(global_start_point, global_end_point)

    def get_start_end_carla_point(self, start_s, start_d):
        #Convert to cartesian based on global path
        start_x, start_y = self.refrence_path_local.frenet_to_cartesian(start_s, start_d)
        goal_x, goal_y = self.refrence_path_global.frenet_to_cartesian(self.s_future, 0)
        # draw_trajectory([start_x], [start_y], self.world, 0.5, 1, 0, "point")
        # draw_trajectory([goal_x], [goal_y], self.world, 0.5, 1, 0, "point")
        start_point = carla.Transform(carla.Location(x=start_x, y=start_y))
        end_point = carla.Transform(carla.Location(x=goal_x, y=goal_y))

        return start_point, end_point 
    
    def initial_path(self, global_start_point, global_end_point):
        self.refrence_path_global = self.route(global_start_point, global_end_point)
        start_x, start_y = self.refrence_path_global.frenet_to_cartesian(0, 0)
        goal_x, goal_y = self.refrence_path_global.frenet_to_cartesian(self.s_future, 0)

        start_point = carla.Transform(carla.Location(x=start_x, y=start_y))
        end_point = carla.Transform(carla.Location(x=goal_x, y=goal_y))
        self.refrence_path_local = self.route(start_point, end_point)

        draw_trajectory(self.refrence_path_global.x, self.refrence_path_global.y, self.world, 0 + 0.05, len(self.refrence_path_global.x)-1, 0, "line", carla.Color(0,0,0))
        # draw_trajectory(self.refrence_path_local.x, self.refrence_path_local.y, self.world, 0 + 0.2, len(self.refrence_path_local.x)-1, 0, "line", carla.Color(0,128,0))


    def process_waypoints(self, waypoints):
        waypoints_ = []
        i =0
        while i< len(waypoints)-1:
            if distance(waypoints[i][0],waypoints[i+1][0]) > 2:
                waypoints_.append(waypoints[i])
            i += 1
        if distance(waypoints[-1][0],waypoints[-2][0]) > 2:
                waypoints_.append(waypoints[-1])
        return waypoints_


    def setup_refrence_path(self):
        path = RefrencePath()
        for waypoint in self.planned_waypoints:
            path.x.append(waypoint[0].transform.location.x)
            path.y.append(waypoint[0].transform.location.y)
            path.direction.append(waypoint[1])
            path.waypoints.append(waypoint[0])
            path.waypoints_x.append(waypoint[0].transform.location.x)
            path.waypoints_y.append(waypoint[0].transform.location.y)

        path.refrencePath = Spline2D(path.x, path.y)
        path.x = []
        path.y = []
        path.s = np.arange(0, path.refrencePath.s[-1], path.ds)

        for i_s in path.s:
            ix, iy = path.refrencePath.calc_position(i_s)
            path.x.append(ix)
            path.y.append(iy)
            path.yaw.append(path.refrencePath.calc_yaw(i_s))
            path.curv.append(path.refrencePath.calc_curvature(i_s))
        
        return path

    def route(self, start_point, end_point):

        try:
            waypoints_ = self.grp.trace_route(start_point.location, end_point.location)
        except Exception as e:
            return self.refrence_path_local

        self.planned_waypoints = self.process_waypoints(waypoints_)
        
        return self.setup_refrence_path()
    
    def re_route(self, start_s, start_d):
        s_total = start_s + self.s_last_wrt_global
        delta = min(max(LOCAL_GLOBAL_PLAN_MIN, LOCAL_GLOBAL_PLAN_MIN + future_s(self.current_state["speed"], self.current_state["long_acc"], PLANNING_DURATION)), LOCAL_GLOBAL_PLAN_MAX)

        self.s_future = min(s_total + delta, self.refrence_path_global.s[-1])
        start_point, end_point = self.get_start_end_carla_point(start_s, start_d)
        
        self.refrence_path_local = self.route(start_point, end_point)

        # draw_trajectory(self.refrence_path_local.x, self.refrence_path_local.y, self.world, self.current_state["z"] + 0.5, len(self.refrence_path_local.x)-1, 0, "line", carla.Color(0,128,0))

        self.s_last_wrt_global = self.refrence_path_global.cartesian_to_frenet(self.current_state["x"], self.current_state["y"])[0]  #self.current_state["s"]
        self.current_state["s"], self.current_state["d"] = self.refrence_path_local.cartesian_to_frenet(self.current_state["x"], self.current_state["y"])

    def is_reroute(self, current_state):
        self.current_state = current_state
        s_current = self.current_state["s"]
        s_future_pred = s_current + future_s(self.current_state["speed"], self.current_state["long_acc"], PLANNING_DURATION) + self.s_last_wrt_global
        # print("     Remaining local global path: ", self.s_future - s_future_pred)
        if self.s_future - s_future_pred < 50:
            self.re_route(self.current_state["s"], self.current_state["d"]) 
            return True
        
    