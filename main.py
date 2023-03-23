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
from scipy.optimize import curve_fit

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
        "z": round(Ego.get_transform().location.z,2),
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

    test = False

    # Setup CARLA Simualtor
    client = carla.Client('localhost',2000)
    client.set_timeout(10.0)
    world = client.load_world('Town05')
    actors = []

    # Get start and end points
    spawn_points = world.get_map().get_spawn_points()

    if test == True:
        start_point_car = carla.Transform(carla.Location(x=625, y=-16.0, z=0.300000), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
        start_point_route = carla.Transform(carla.Location(x=625, y=-16.0, z=0.300000), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
    else:
        start_point_car = carla.Transform(carla.Location(x=189.740814, y=-100.026948, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        start_point_route = carla.Transform(carla.Location(x=189.740814, y=-102.026948, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))

    obs = carla.Transform(carla.Location(x=189.740814, y=-30.026948, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))  
    end_point = carla.Transform(carla.Location(x=-99.3, y=189, z=0.300000))

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

    spawn_obstacles = True

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
    
    time.sleep(1)

    if test == True:
        flag = 1
        speed_list_all = []
        acc_list_all = []
        time_list_all = []

        speed_list_acc = []
        acc_list = []
        time_list_acc = []
        
        speed_list_dec = []
        dec_list = []
        time_list_dec = []
        
        total_acc_list = []
        
        prev_speed = 0
        t = 0

        throttle = 0.05
        brake = 0
        steering = 0
        path = "./data/"+ str(throttle)
        if not os.path.exists(path):
            os.makedirs(path)

        s1 = s2 = time.time()
    
    # obs_veh.set_target_velocity(target_velocity)
    while  not reached_goal:
        s = time.time()
        current_state = get_current_states(Ego, local_planner)
        e1 = time.time()
        
        if test == True:
            Ego.apply_control(carla.VehicleControl(throttle, steering, brake))
            e1 = time.time()
            speed = current_state["speed"]
            acc = current_state["long_acc"]
            # total_acc = current_state["total_acc"]

            if (speed > 38 or (abs(prev_speed - speed) < 0.01 and t>10)) and flag:
                print("Decelerating Now")
                flag = 0
                throttle = 0.0
                brake = 0.0
                steering = 0.0
                Ego.apply_control(carla.VehicleControl(throttle, steering, brake))
                s2 = time.time()
                e1 = time.time()
            
            t = round(e1 - s2, 2)
            
            if flag == 1:
                speed_list_acc.append(speed)
                acc_list.append(acc)
                # total_acc_list.append(total_acc)
                time_list_acc.append(t)
            else:
                speed_list_dec.append(speed)
                dec_list.append(acc)
                time_list_dec.append(t)
            
            speed_list_all.append(speed)
            acc_list_all.append(acc)
            time_list_all.append(e1-s1)

            prev_speed = current_state["speed"]
            
            if speed <0.01 and flag == 0:
                break
        else:
            state, target_s, target_d = behavior_planner.get_next_behavior(current_state, lookahead_path= 10 + current_state["speed"]*3, vehicles=vehicles)
            # print("EGo Speed: ", current_state["speed"])
            # print("Current State", state)
            if state == "SAFETY_STOP":
                print("Applying Emergency Brake")
                Ego.apply_control((carla.VehicleControl(throttle=0, brake=1)))
                continue

            x, y, yaw, v, dt, planning_time = local_planner.run_step(current_state, target_s, target_d) 
            e2 = time.time()
            draw_trajectory(world, x,y, current_state["z"]+0.5)
            e3 = time.time()
            controller.update_waypoints(x, y, yaw, v, current_state, dt, planning_time)
            print("------")
            # print("Current Velocity for Local planner", v[0])
            print("Current Velocity of vehicle", current_state["speed"])
            print("CUrrent Acceleration of Ego: ", current_state["long_acc"])
            controller.run_step()
            e4 = time.time()
            
            # print("TOTAL TIME: ", e4-s)
            # print("Get State time: ", e1-s)
            # print("Local Planner Time: ", e2-e1)
            # print("Drawing Time: ", e3-e2)
            # print("Controller Time: ", e4-e3)
            # print("---------------------------")

if test == True:

    np.savetxt(path + "/speed_list_acc.csv", np.array(speed_list_acc))
    np.savetxt(path + "/acc_list.csv", np.array(acc_list))
    np.savetxt(path + "/speed_list_dec.csv", np.array(speed_list_dec))
    np.savetxt(path + "/dec_list.csv", np.array(dec_list))

    # Full Profile
    plt.figure()
    plt.plot(time_list_all, speed_list_all)
    plt.title("Time vs Velcoity full test")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")

    plt.figure()
    plt.plot(time_list_all, acc_list_all)
    plt.title("Time vs Acceleration full test")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s2)")

    # Accelerate
    # plt.figure()
    # plt.plot(time_list_acc, speed_list_acc)
    # plt.title("Time vs Velcoity when Accelerating")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity (m/s)")
    # plt.figure()
    # plt.plot(time_list_acc, acc_list)
    # plt.title("Time vs Acceleration when Accelerating")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (m/s2)")
    plt.figure()
    plt.plot(speed_list_acc, acc_list)
    plt.title("Velocity vs Acceleration when Accelerating")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Acceleration (m/s2)")

    # Decelerate
    # plt.figure()
    # plt.plot(time_list_dec, speed_list_dec)
    # plt.title("Time vs Velcoity when Decelerating")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity (m/s)")
    # plt.figure()
    # plt.plot(time_list_dec, dec_list)
    # plt.title("Time vs Acceleration when Decelerating")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (m/s2)")

    plt.figure()
    plt.plot(speed_list_dec, dec_list)
    plt.title("Velocity vs Acceleration when Decelerating")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Acceleration (m/s2)")

    # plt.figure()
    # plt.plot(time_list_acc, total_acc_list)
    # plt.plot(time_list_acc, acc_list)
    # plt.title("Total Acc vs Long Acc")

    plt.show()

#TODO: Calculate Planning duration based on S 

    # s = time.time()a
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
