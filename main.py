
import sys
import glob
import os
import random
from Behavior_Planner import BehaviorPlanner
from Local_Planner import LocalPlanner
from math import *
from spline import Spline2D
from matplotlib import pyplot as plt
import numpy as np
import time
from Controller import PIDController
from Controller import MPC
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

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

def distance(waypoint1, waypoint2):

    x = waypoint1.transform.location.x - waypoint2.transform.location.x
    y = waypoint1.transform.location.y - waypoint2.transform.location.y

    return sqrt(x * x + y * y)

def preprocess(waypoints):
    waypoints_ = []
    i =0
    while i< len(waypoints)-1:
        if distance(waypoints[i][0],waypoints[i+1][0]) > 2:
            waypoints_.append(waypoints[i][0])
        i += 1
    if distance(waypoints[-1][0],waypoints[-2][0]) > 2:
            waypoints_.append(waypoints[-1][0])
    return waypoints_

def get_current_states(Ego):
    ego_vehicle_loc = Ego.get_location()
    ego_wpt = world.get_map().get_waypoint(ego_vehicle_loc)
    ego_vel = Ego.get_velocity()
    ego_speed = sqrt(ego_vel.x**2 + ego_vel.y**2)
    ego_acc = Ego.get_acceleration()
    ego_transform = Ego.get_transform()
    transform_matrix = np.linalg.inv(ego_transform.get_matrix())
    ego_vel = transform_matrix @ np.array([ego_vel.x, ego_vel.y, ego_vel.z, 0.0])
    ego_acc = transform_matrix @ np.array([ego_acc.x, ego_acc.y, ego_acc.z, 0.0])
    current_state= {
        "x": round(Ego.get_transform().location.x,2),
        "y": round(Ego.get_transform().location.y,2),
        "yaw": round(Ego.get_transform().rotation.yaw*pi/180,2),
        "long_vel": round(ego_vel[0],2),
        "lat_vel" : round(ego_vel[1],2),
        "long_acc": round(ego_acc[0],2),
        "lat_acc": round(ego_acc[1],2),
        "lane": ego_wpt.lane_id,
        "waypoint": ego_wpt,
        "speed": ego_speed
    }
    return current_state


if __name__ == "__main__":


    # Setup CARLA Simualtor
    client = carla.Client('localhost',2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    actors = []


    # Get start and end points
    spawn_points = world.get_map().get_spawn_points()
    start_point = carla.Transform(carla.Location(x=-10, y=40.80, z=0.5), carla.Rotation(yaw=90))
    obs = carla.Transform(carla.Location(x=-10, y=80.80, z=0.5), carla.Rotation(yaw=90))
    end_point = random.choice(spawn_points)    


    #Generate Global Path Waypoints
    dao = GlobalRoutePlannerDAO(world.get_map(), 3)   
    grp = GlobalRoutePlanner(dao)
    grp.setup() 
    waypoints_ = grp.trace_route(start_point.location, end_point.location)
    waypoints_ = preprocess(waypoints_)


    # Generate refrence path using spline fitting on waypoints.
    x = [i.transform.location.x for i in waypoints_]
    y = [i.transform.location.y for i in waypoints_]
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.5)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    # Spawn ego vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('model3'))
    Ego = world.spawn_actor(vehicle_bp,start_point)
    # obs_veh = world.spawn_actor(vehicle_bp,obs)
    actors.append(Ego)
    # actors.append(obs_veh)


    # Setup parameters
    reached_goal = False


    # Setup Behavior Planner
    behavior_planner = BehaviorPlanner()
    local_planner = LocalPlanner(sp,s,rx,ry,ryaw,rk)
    controller = MPC(Ego)
    time.sleep(5)
    while  not reached_goal:
        current_state = get_current_states(Ego)
        # vehicles = [get_current_states(obs_veh)]
        # state = behavior_planner.get_next_behavior(current_state,None,vehicles)
        # if state == "lane_change":
        #     waypoint0 = current_state["waypoint"]
        #     waypoint1 = current_state["waypoint"].next(4.0)
        #     waypoint2 = current_state["waypoint"].next(8.0)
        #     waypoint2 = waypoint2[0].get_left_lane()
        #     waypoints_ = grp.trace_route(waypoint2.transform.location, end_point.location)
        #     waypoints_ = preprocess(waypoints_)
        #     waypoints_.insert(0,waypoint1[0])
        #     waypoints_.insert(0,waypoint0)
        #     x = [i.transform.location.x for i in waypoints_]
        #     y = [i.transform.location.y for i in waypoints_]
        #     sp = Spline2D(x, y)
        #     s = np.arange(0, sp.s[-1], 0.5)
        #     for i_s in s:
        #         ix, iy = sp.calc_position(i_s)
        #         rx.append(ix)
        #         ry.append(iy)
        #         ryaw.append(sp.calc_yaw(i_s))
        #         rk.append(sp.calc_curvature(i_s))
        #     local_planner.update_refrence(sp,s,rx,ry,ryaw,rk)
        

        x, y, yaw,v = local_planner.run_step(current_state,2)
        controller.update_waypoints(x, y, yaw, v)
        controller.run_step()
        # plt.plot(x,y)
        
        # plt.axis('equal')
        # plt.show()
        # break


       

'''
possible error check:
hard coded idx = 1 if idx==0
'''