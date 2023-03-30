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
from numpy import savetxt
import time
from Controller import MPC
from utils import *
from params import *

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

def get_current_states(Ego, local_planner):
    ego_vehicle_loc = Ego.get_location()
    # ego_wpt = world.get_map().get_waypoint(ego_vehicle_loc)
    ego_vel = Ego.get_velocity()
    ego_speed = sqrt(ego_vel.x**2 + ego_vel.y**2)
    ego_acc = Ego.get_acceleration()
    ego_acc1 = sqrt(ego_acc.x**2 + ego_acc.y**2)
    ego_transform = Ego.get_transform()
    transform_matrix = np.linalg.inv(ego_transform.get_matrix())
    ego_vel = transform_matrix @ np.array([ego_vel.x, ego_vel.y, ego_vel.z, 0.0])
    ego_acc = transform_matrix @ np.array([ego_acc.x, ego_acc.y, ego_acc.z, 0.0])
    s,d = local_planner.cartesian_to_frenet(Ego.get_transform().location.x, Ego.get_transform().location.y) #None, None
    current_state= {
        "x": round(Ego.get_transform().location.x,2),
        "y": round(Ego.get_transform().location.y,2),
        "z": round(Ego.get_transform().location.z,2),
        "yaw": round(Ego.get_transform().rotation.yaw*pi/180,2),
        "long_vel": round(ego_vel[0],2),
        "lat_vel" : round(ego_vel[1],2),
        "long_acc": round(ego_acc[0],2),
        "lat_acc": round(ego_acc[1],2),
        # "lane": ego_wpt.lane_id,
        # "waypoint": ego_wpt,
        "speed": ego_speed,
        "s": s,
        "d":d,
        "target_speed": kmph_to_ms(Ego.get_speed_limit()),
        "total_acc": ego_acc1
    }
    return current_state

if __name__ == "__main__":

    # Setup CARLA Simualtor
    client = carla.Client('localhost',CLIENT)
    client.set_timeout(10.0)
    world = client.load_world(TOWN)
    spawn_points = world.get_map().get_spawn_points()

    if RANDOM_POINT:
        start_point = random.choice(spawn_points) 
        end_point = random.choice(spawn_points)
    else:
        start_point = carla.Transform(carla.Location(x=START_X, y=START_Y, z=0.3), carla.Rotation(pitch=0.0, yaw=START_YAW, roll=0.0))
        end_point = carla.Transform(carla.Location(x=END_X, y=END_Y, z=0.3))

    obstacle1 = carla.Transform(carla.Location(x=OBS1_X, y=OBS1_Y, z=0.3), carla.Rotation(pitch=0.0, yaw=OBS1_YAW, roll=0.0))

    if SYNCHRONOUS_MODE:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TICK_TIME
        world.apply_settings(settings)
    
    actors = []
    # Get start and end points
    spawn_points = world.get_map().get_spawn_points()

    #Generate Global Path Waypoints
    dao = GlobalRoutePlannerDAO(world.get_map(), 3)   
    grp = GlobalRoutePlanner(dao)
    grp.setup() 
    waypoints_ = grp.trace_route(start_point.location, end_point.location)
    waypoints_ = preprocess(waypoints_)
    vehicles = []
    # Generate refrence path using spline fitting on waypoints.
    global_x = [i.transform.location.x for i in waypoints_]
    global_y = [i.transform.location.y for i in waypoints_]
    sp = Spline2D(global_x, global_y)
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
    actors.append(Ego)

    if SPAWN_OBSTACLE:
        obs_veh = world.spawn_actor(vehicle_bp,obstacle1)
        actors.append(obs_veh)
        vehicles.append(obs_veh)

    # Setup parameters
    reached_goal = False

    time.sleep(1)

    # Setup Behavior Planner
    local_planner = LocalPlanner(world,sp,s,rx,ry,ryaw,rk)
    behavior_planner = BehaviorPlanner(local_planner)
    controller = MPC(Ego, world, global_x, global_y)

    if SPAWN_OBSTACLE:
        pass
        # obs_veh.set_target_velocity(target_velocity)

    if SYNCHRONOUS_MODE:
        world.tick()
    
    spectator = world.get_spectator()
    spectator.set_transform(Ego.get_transform())
    while  not reached_goal:
        # timestamp = world.get_snapshot()
        # print(f"elapsed_seconds: {timestamp.elapsed_seconds}, delta_seconds: {timestamp.delta_seconds}")
        s = time.time()
        current_state = get_current_states(Ego, local_planner)
        e1 = time.time()
    
        behavior, target_s, target_d = behavior_planner.get_next_behavior(current_state, lookahead_path= 10 + current_state["speed"]*3, vehicles=vehicles)
        # print("EGo Speed: ", current_state["speed"])
        # print("Current State", state)
        if behavior == "SAFETY_STOP":
            print("Applying Emergency Brake")
            Ego.apply_control((carla.VehicleControl(throttle=0, brake=1)))
            continue

        x, y, yaw, v, final_speed = local_planner.run_step(current_state, target_s, target_d, behavior) 
        e2 = time.time()
        # draw_trajectory(world, x,y, current_state["z"]+0.5)
        e3 = time.time()
        controller.update_waypoints(x, y, yaw, v, current_state, final_speed)
        # print("Current Velocity for Local planner", v[0])
        print("Current Velocity of vehicle", round(ms_to_mph(current_state["speed"]),2))
        # print("CUrrent Acceleration of Ego: ", round(current_state["long_acc"],2))
        # print("X: ", current_state["x"])
        # print("Y: ", current_state["y"])
        controller.run_step()
        e4 = time.time()
        if CONTROL_HORIZON == 1:
            print("TOTAL TIME: ", round(e4-s,4))
        # print("Get State time: ", round(e1-s,3))
        # print("Local Planner Time: ", round(e2-e1,3))
        # print("Drawing Time: ", round(e3-e2,3))
        # print("Controller Time: ", round(e4-e3,3))
        print("---------------------------")