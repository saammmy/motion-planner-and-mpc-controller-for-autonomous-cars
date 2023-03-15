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

def get_current_states(Ego, local_planner):
    ego_vehicle_loc = Ego.get_location()
    ego_wpt = world.get_map().get_waypoint(ego_vehicle_loc)
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
        "yaw": round(Ego.get_transform().rotation.yaw*pi/180,2),
        "long_vel": round(ego_vel[0],2),
        "lat_vel" : round(ego_vel[1],2),
        "long_acc": round(ego_acc[0],2),
        "lat_acc": round(ego_acc[1],2),
        "lane": ego_wpt.lane_id,
        "waypoint": ego_wpt,
        "speed": ego_speed,
        "s": s,
        "d":d,
        "target_speed": Ego.get_speed_limit(),
        "total_acc": ego_acc1
    }
    return current_state

def draw_trajectory(world, x,y, height=0.5, time=0.3):
    for i in range(len(x)): #len(x)):
            begin = carla.Location(x=x[i],y=y[i],z=height)
            world.debug.draw_point(begin,size=0.2,life_time=time)

def find_junction(waypoints_,world):
    junction = []
    x = []
    y = []
    for waypoint in waypoints_:
        if waypoint[0].is_junction:
            print(waypoint[0].is_junction)
            junction.append(waypoint[0])
            x.append(waypoint[0].transform.location.x)
            y.append(waypoint[0].transform.location.y)
    draw_trajectory(world, x,y,time=20)
    return junction
if __name__ == "__main__":


    # Setup CARLA Simualtor
    client = carla.Client('localhost',2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    actors = []


    # Get start and end points
    spawn_points = world.get_map().get_spawn_points()
    start_point_car = carla.Transform(carla.Location(x=-10, y=45.80, z=0.5), carla.Rotation(yaw=90))
    start_point_route = carla.Transform(carla.Location(x=-10, y=40.80, z=0.5), carla.Rotation(yaw=90))
    # obs = carla.Transform(carla.Location(x=-10, y=70.80, z=0.5), carla.Rotation(yaw=90))
    obs = carla.Transform(carla.Location(x= 171, y= 93, z=0.5), carla.Rotation(yaw=270))
    # end_point = carla.Transform(carla.Location(x=-105, y=132, z=0.5))
    # end_point = carla.Transform(carla.Location(x=210, y=63, z=0.5))
    end_point = carla.Transform(carla.Location(x= 171, y= 70, z=0.5))
    # end_point = random.choice(spawn_points)    

    #Generate Global Path Waypoints
    dao = GlobalRoutePlannerDAO(world.get_map(), 3)   
    grp = GlobalRoutePlanner(dao)
    grp.setup() 
    waypoints_ = grp.trace_route(start_point_route.location, end_point.location)
    # junction = find_junction(waypoints_,world)
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
    Ego = world.spawn_actor(vehicle_bp,start_point_car)
    actors.append(Ego)

    spawn_obstacles = False

    if spawn_obstacles:
        obs_veh = world.spawn_actor(vehicle_bp,obs)
        actors.append(obs_veh)
        vehicles.append(obs_veh)
    target_velocity = carla.Vector3D(x=0, y=4, z=0)


    # Setup parameters
    reached_goal = False


    # Setup Behavior Planner
    local_planner = LocalPlanner(world,sp,s,rx,ry,ryaw,rk)
    behavior_planner = BehaviorPlanner(local_planner)
    controller = MPC(Ego, world, global_x, global_y)

    time.sleep(5)
    # obs_veh.set_target_velocity(target_velocity)
    while  not reached_goal:
        s = time.time()
        current_state = get_current_states(Ego, local_planner)
        # TODO: Calculate lookahead distance based on velocity
        state, target_s, target_s_d, target_s_dd = behavior_planner.get_next_behavior(current_state, lookahead_path= 10+ current_state["speed"]*1.6, vehicles=vehicles)
        # print("EGo Speed: ", current_state["speed"])
        # print("Current State", state)
        if state == "SAFETY_STOP":
             print("Applying Emergency Brake")
             Ego.apply_control((carla.VehicleControl(throttle=0, brake=1)))
             continue

        x, y, yaw,v = local_planner.run_step(current_state, state, target_s, target_s_d, target_s_dd) 
        draw_trajectory(world, x,y)
        controller.update_waypoints(x, y, yaw, v, current_state)
        # print("------")
        # print("Current Velocity for Local planner", v[0])
        # print("Current Velocity of vehicle", current_state["speed"])
        # print("CUrrent Acceleration of Ego: ", current_state["long_acc"])
        controller.run_step()
        e = time.time()
        # print(e-s)

#TODO: Calculate Planning duration based on S 

    # s = time.time()
    # while time.time() - s < 5:
    #      Ego.apply_control(carla.VehicleControl(1, 0, 0))
    #      state = get_current_states(Ego,None)
    #     #  print("Acc :",state["total_acc"])
    #     #  print("Vel :",state["speed"])
    # print("Applying brake")
    # s = time.time()
    # while time.time() - s < 5:
    #      Ego.apply_control(carla.VehicleControl(0, 0, 0.1))
    #      state = get_current_states(Ego, None)
    #      print("Acc :",state["total_acc"])
    #      print("Vel :",state["speed"])
