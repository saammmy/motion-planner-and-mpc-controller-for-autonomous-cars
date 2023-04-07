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
import logging
from Mission_Planner import MissionPlanner
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
from carla import VehicleLightState as vls

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


def get_current_states(Ego, refrence_path):
    # ego_wpt = world.get_map().get_waypoint(ego_vehicle_loc)
    ego_vel = Ego.get_velocity()
    ego_speed = sqrt(ego_vel.x**2 + ego_vel.y**2)
    ego_acc = Ego.get_acceleration()
    ego_acc1 = sqrt(ego_acc.x**2 + ego_acc.y**2)
    ego_transform = Ego.get_transform()
    ego_vehicle_loc = ego_transform.location
    ego_forward_vector = ego_transform.get_forward_vector()
    transform_matrix = np.linalg.inv(ego_transform.get_matrix())
    ego_vel = transform_matrix @ np.array([ego_vel.x, ego_vel.y, ego_vel.z, 0.0])
    ego_acc = transform_matrix @ np.array([ego_acc.x, ego_acc.y, ego_acc.z, 0.0])
    s,d = refrence_path.cartesian_to_frenet(Ego.get_transform().location.x, Ego.get_transform().location.y) #None, None
    current_state= {
        "x": round(Ego.get_transform().location.x,2),
        "y": round(Ego.get_transform().location.y,2),
        "z": round(Ego.get_transform().location.z,2),
        "transform": ego_transform,
        "forward_vector": ego_forward_vector,
        "location_vector": ego_vehicle_loc,
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
    trafficManager = client.get_trafficmanager()
    tm_port = trafficManager.get_port()
    if RANDOM_POINT:
        start_point = random.choice(spawn_points) 
        end_point = random.choice(spawn_points)
    else:
        start_point = carla.Transform(carla.Location(x=START_X, y=START_Y, z=0.3), carla.Rotation(pitch=0.0, yaw=START_YAW, roll=0.0))
        # start_point_temp = carla.Transform(carla.Location(x=START_X-50, y=START_Y-6, z=0.3), carla.Rotation(pitch=0.0, yaw=START_YAW, roll=0.0))
        end_point = carla.Transform(carla.Location(x=END_X, y=END_Y, z=0.3))


    if SYNCHRONOUS_MODE:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = TICK_TIME
        world.apply_settings(settings)
    
    actors = []
    obstacles = []
    # Get start and end points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn ego vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('model3'))
    Ego = world.spawn_actor(vehicle_bp,start_point)
    actors.append(Ego)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    if SPAWN_OBSTACLE:
        obstacle1 = carla.Transform(carla.Location(x=OBS1_X, y=OBS1_Y, z=0.3), carla.Rotation(pitch=0.0, yaw=OBS1_YAW, roll=0.0))
        obstacle2 = carla.Transform(carla.Location(x=OBS2_X, y=OBS2_Y, z=0.3), carla.Rotation(pitch=0.0, yaw=OBS2_YAW, roll=0.0))
        obs = [obstacle1, obstacle2]
        light_state = vls.NONE
        if True:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam
        for ob in obs:
            obstacles.append(SpawnActor(vehicle_bp, ob)
            .then(SetAutopilot(FutureActor, True, trafficManager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))

    vehicles_list = []
    for response in client.apply_batch_sync(obstacles, SYNCHRONOUS_MODE):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)



    # Setup parameters
    reached_goal = False

    time.sleep(1)

    # Setup Behavior Planner
    local_planner = LocalPlanner(world)
    behavior_planner = BehaviorPlanner(world, Ego, start_point, end_point)
    controller = MPC(Ego, world, behavior_planner.mission_planner.refrence_path_global.x, behavior_planner.mission_planner.refrence_path_global.y, behavior_planner.mission_planner.refrence_path_local.x, behavior_planner.mission_planner.refrence_path_local.y)

    
    junction_points = [obj for obj in behavior_planner.mission_planner.refrence_path_local.waypoints if obj.is_junction == True]
    # print("X: ",junction_points[0].transform.location.x, "Y: ", junction_points[0].transform.location.y)
    # draw_trajectory([junction_points[0].transform.location.x], [junction_points[0].transform.location.y], world, 0.5, 1, 0, "point")

    if SYNCHRONOUS_MODE:
        world.tick()
    
    spectator = world.get_spectator()
    spectator.set_transform(Ego.get_transform())
    prev_x = START_X
    prev_y = START_Y


    obstacles = world.get_actors().filter("*vehicle*")
    traffic_lights = world.get_actors().filter("*traffic.traffic_light*")
    traffic_lights = TrafficLights(traffic_lights)
        

    while  not behavior_planner.goal_reached:
        timestamp = world.get_snapshot()
        print("-------------------------------------------------")
        # print("CARLA TIME STAMP: ", round(timestamp.elapsed_seconds,4))
        # print("     Synchronous Mode: ", SYNCHRONOUS_MODE)
        # print("     Carla Delta Time: ",round(timestamp.delta_seconds,4))

        s = time.time()
        current_state = get_current_states(Ego, behavior_planner.mission_planner.refrence_path_local)
        curr_x = current_state["x"]
        curr_y = current_state["y"]

        draw_trajectory([prev_x, curr_x], [prev_y, curr_y], world, current_state["z"] + 0.5, 1, 0, "line", carla.Color(0,255,255))

        prev_x = curr_x
        prev_y = curr_y

        e1 = time.time()        

        behavior, target_vel = behavior_planner.get_next_behavior(current_state, behavior_planner.mission_planner.refrence_path_local, obstacles, traffic_lights)
        e2 = time.time()
        # behavior = "FOLLOW_LANE"
        # target_s = None
        # target_d = 0
        # target_vel = current_state["target_speed"]

        if behavior == "EMERGENCY_BRAKE":
            print("Applying Emergency Brake")
            Ego.apply_control((carla.VehicleControl(throttle=0, brake=1)))
            continue

        x, y, yaw, v, final_speed = local_planner.run_step(behavior_planner.mission_planner.refrence_path_local, current_state, target_vel) 
        e3 = time.time()
        controller.update_waypoints(x, y, yaw, v, current_state, final_speed, behavior_planner.mission_planner.refrence_path_local.x, behavior_planner.mission_planner.refrence_path_local.y)
        controller.run_step()
        e4 = time.time()

        

        if CONTROL_HORIZON == 1:
            pass
            # print("     Time to run one code loop: ", round(e4-s,4))
        # print("     Behaviour: ", behavior)
        # print("     Get State time: ", round(e1-s,3))
        # print("     Behavior Planner Time: ", round(e2-e1,3))
        # print("     Local Planner Time: ", round(e3-e2,3))
        # print("     Controller Time: ", round(e4-e3,3))
        # print("     Desired Speed: ", round(ms_to_mph(final_speed),2))
        # print("     Current speed of vehicle: ", round(ms_to_mph(current_state["speed"]),2))


    control = carla.VehicleControl(0, 0, 1)
    Ego.apply_control(control)