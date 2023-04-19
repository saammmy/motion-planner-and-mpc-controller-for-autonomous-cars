from math import sqrt
import time
import Local_Planner
from misc import get_speed
import numpy as np
from  utils import *
from params import *
from Mission_Planner import MissionPlanner
import random
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
from agents.navigation.local_planner import RoadOption



def future_s(speed, acc, t):
    return speed*t + 0.5*max(min(acc,MAX_ACC),-5)*t**2

def time_to_collision(relative_speed, delta_s):
    return delta_s/relative_speed

def calculate_distance(current_state,vehicle):
    return sqrt((current_state['x'] - vehicle.get_location().x)**2 + (current_state['y'] - vehicle.get_location().y)**2)


class BehaviorPlanner:
    def __init__(self,world, Ego, start_point, end_point):
        self.current_behavior = "FOLLOW_LANE"
        self.world = world
        self.Ego = Ego
        self.lookahead_time = 6
        self.react_to_collision = 8
        self.overtake_lookahead_time = 2.6
        self.speed = 0
        self.was_tailgate = False
        self.start_point = start_point
        self.end_point = end_point
        self.mission_planner = MissionPlanner(world, Ego, start_point, end_point)
        self.end_x, self.end_y = self.mission_planner.refrence_path_global.frenet_to_cartesian(self.mission_planner.refrence_path_global.s[-1],0)

        self.goal_reached = False
        self.idle_time = 0
        self.prev_time = time.time()
    
    def is_reached_goal(self, current_s_local):
        curremt_s_global = current_s_local + self.mission_planner.s_last_wrt_global
        if  self.mission_planner.refrence_path_global.s[-1] - curremt_s_global < GOAL_REACH_THRESHOLD:
            return True
        return False
    
    def get_waypoint_from_frenet(self, s,d):
        x, y = self.mission_planner.refrence_path_local.frenet_to_cartesian(s, d)
        argmin = get_closest_waypoint(self.mission_planner.refrence_path_local.waypoints_x, self.mission_planner.refrence_path_local.waypoints_y, x, y)
        waypoint = self.mission_planner.refrence_path_local.waypoints[argmin]

        return waypoint, argmin

    def normalize_d(self,d, lane_width = LANE_WIDTH):
        if d < lane_width/2 and d > -lane_width/2:
            return 0
        elif d > lane_width/2 and d < lane_width*3/2:
            return LANE_WIDTH
        elif d > -lane_width*3/2 and d < -lane_width/2:
            return -LANE_WIDTH
        else:
            # print("ERROR: Car is outside reference path range with d= ",d)
            return None
        
    def get_next_behavior(self,current_state, local_path, obstacles, traffic_lights):
        self.speed = current_state["speed"]
        ego_angle = self.Ego.get_transform().rotation.yaw
        waypoint, idx = self.get_waypoint_from_frenet(current_state["s"], current_state["d"])
        if idx+1 == len(self.mission_planner.refrence_path_local.waypoints):
            next_waypoint = waypoint
        else:
            next_waypoint = self.mission_planner.refrence_path_local.waypoints[idx+2] 
        if next_waypoint.is_junction:
            draw_trajectory([current_state["x"]], [current_state["y"]], self.world, current_state["z"] + 1.4, 1, 0, "point", carla.Color(255,0,255))
        direction = self.mission_planner.refrence_path_local.direction[idx]

        self.goal_reached = self.is_reached_goal(current_state["s"])

        is_reroute = self.mission_planner.is_reroute(current_state)
        if is_reroute:
            print("     New Local Global Path Recieved")

        self.speed = current_state["target_speed"]
        target_s = None
        self.current_behavior = "FOLLOW LANE"

        left_lane_obstacles= []
        right_lane_obstacles = []
        current_lane_obstacles = []
        other_lane_obstacles = []

        left_lane_hazard_obstacles = []
        right_lane_hazard_obstacles = []
        current_lane_hazard_obstacles = []
        other_lane_hazard_obstacles = []

        ego_norm_d = self.normalize_d(current_state["d"])
        # print("Current d", current_state["d"])
        lookahead = min(max(MIN_OBS_RAD,future_s(current_state["speed"], current_state["long_acc"], self.lookahead_time)),MAX_OBS_RAD)

        # Collect waypoints and s when turn happens
        s_future_lookahead = min(current_state["s"] + lookahead, self.mission_planner.refrence_path_local.s[-1])
        # print("S future",s_future_lookahead)
        waypoint_future,idx_future = self.get_waypoint_from_frenet(s_future_lookahead,0)
        incoming_lane_change = None
        s_lane_change = None
        # print("Direction: ",self.mission_planner.refrence_path_local.direction[idx: idx_future])
        for i in range(idx, idx_future):
            if self.mission_planner.refrence_path_local.direction[i] == RoadOption.CHANGELANERIGHT:
                incoming_lane_change = "Right"
                s_lane_change,_ = self.mission_planner.refrence_path_local.cartesian_to_frenet(self.mission_planner.refrence_path_local.waypoints_x[i], self.mission_planner.refrence_path_local.waypoints_y[i])
                break
            elif self.mission_planner.refrence_path_local.direction[i] == RoadOption.CHANGELANELEFT:
                incoming_lane_change = "Left"
                s_lane_change,_ = self.mission_planner.refrence_path_local.cartesian_to_frenet(self.mission_planner.refrence_path_local.waypoints_x[i], self.mission_planner.refrence_path_local.waypoints_y[i])
                break
        
        incoming = "STRAIGHT"
        for i in range(idx, idx_future):
            if self.mission_planner.refrence_path_local.direction[i] == RoadOption.RIGHT:
                incoming = "RIGHT_TURN"
                s_incoming,_=self.mission_planner.refrence_path_local.cartesian_to_frenet(self.mission_planner.refrence_path_local.waypoints_x[i], self.mission_planner.refrence_path_local.waypoints_y[i])
                print("RIGHT INCOMING IN DISTANCE: ", s_incoming - current_state["s"])
                break
            elif self.mission_planner.refrence_path_local.direction[i] == RoadOption.RIGHT:
                incoming = "LEFT_TURN"
                s_incoming,_=self.mission_planner.refrence_path_local.cartesian_to_frenet(self.mission_planner.refrence_path_local.waypoints_x[i], self.mission_planner.refrence_path_local.waypoints_y[i])
                break
            
        for obstacle in obstacles:
            if(obstacle.id == self.Ego.id):
                continue
            # get the location of the obstacle
            obstacle_location = obstacle.get_location()

            # calculate the vector from the ego vehicle to the obstacle
            to_obstacle_vector = obstacle_location - current_state["location_vector"]

            # calculate the distance between the ego vehicle and the obstacle
            distance_to_obstacle = self.Ego.get_location().distance(obstacle_location)

            if distance_to_obstacle < lookahead:
                dot_product = current_state["forward_vector"].x * to_obstacle_vector.x + current_state["forward_vector"].y * to_obstacle_vector.y + current_state["forward_vector"].z * to_obstacle_vector.z
                magnitude_a = math.sqrt(current_state["forward_vector"].x ** 2 + current_state["forward_vector"].y ** 2 + current_state["forward_vector"].z ** 2)
                magnitude_b = math.sqrt(to_obstacle_vector.x ** 2 + to_obstacle_vector.y ** 2 + to_obstacle_vector.z ** 2)

                # calculate angle in radians
                cos_theta = dot_product / (magnitude_a * magnitude_b)
                theta = math.acos(cos_theta)

                # convert to degrees
                angle = math.degrees(theta)

                obstacle_angle = obstacle.get_transform().rotation.yaw

                # print(f"angle_to_obstacle: y:{obstacle.get_location().y}, angle:{angle}")
                # check if the obstacle is directly ahead of the ego vehicle
                try:
                    if angle < 90.0 and (-45 <wrap_angle_pi_to_pi(np.deg2rad(ego_angle - obstacle_angle))< 45):    
                        obs_s, obs_d = local_path.cartesian_to_frenet(obstacle.get_location().x, obstacle.get_location().y)
                        d_offset = 0
                        if incoming_lane_change:
                            if obs_s > s_lane_change:
                                if incoming_lane_change == "Left":
                                    d_offset = -LANE_WIDTH
                                else:
                                    d_offset = LANE_WIDTH
                        obs_d_norm = self.normalize_d(obs_d + d_offset)
                        # if obs_d_norm!=None:
                        #     obs_d_norm += d_offset

                        # obs_d_norm = self.normalize_d(obs_d)
                        delta_s = obs_s - current_state["s"]
                        # print("obs d", obs_d, " for id: ", obstacle.id)
                        if obs_d_norm == None:
                            other_lane_obstacles.append(Obstacle(obstacle, None, obs_s, obs_d, delta_s))
                        elif obs_d_norm == ego_norm_d:
                            current_lane_obstacles.append(Obstacle(obstacle, 0, obs_s, obs_d, delta_s))
                        elif obs_d_norm > ego_norm_d:
                            right_lane_obstacles.append(Obstacle(obstacle, -1, obs_s, obs_d, delta_s))
                        elif obs_d_norm < ego_norm_d:
                            left_lane_obstacles.append(Obstacle(obstacle, 1, obs_s, obs_d, delta_s))
                    elif distance_to_obstacle < HAZARD_LOOKAHEAD_RAD and 90 < angle < 180 and (-45 <wrap_angle_pi_to_pi(np.deg2rad(ego_angle - obstacle_angle))< 45):
                        obs_s, obs_d = local_path.cartesian_to_frenet(obstacle.get_location().x, obstacle.get_location().y)
                        obs_d_norm = self.normalize_d(obs_d)
                        delta_s = obs_s - current_state["s"]
                        if obs_d_norm == None:
                            other_lane_hazard_obstacles.append(Obstacle(obstacle, None, obs_s, obs_d, delta_s))
                        elif obs_d_norm == ego_norm_d:
                            current_lane_hazard_obstacles.append(Obstacle(obstacle, 0, obs_s, obs_d, delta_s))
                        elif obs_d_norm > ego_norm_d:
                            right_lane_hazard_obstacles.append(Obstacle(obstacle, -1, obs_s, obs_d, delta_s))
                        elif obs_d_norm < ego_norm_d:
                            left_lane_hazard_obstacles.append(Obstacle(obstacle, 1, obs_s, obs_d, delta_s))
                except Exception as e:
                    print("ERROR in getting obstacle")
                    # Check if ahead of s future

        # print(len(left_lane_obstacles))
        # print(len(right_lane_obstacles))
        # print(len(current_lane_obstacles))
        
        
        closest_traffic_light_idx = get_closest_waypoint(traffic_lights.x_list, traffic_lights.y_list, current_state["x"], current_state["y"])
        closest_traffic_light = traffic_lights.lights[closest_traffic_light_idx]

        my_traffic_light = None
        if self.Ego.get_location().distance(closest_traffic_light.get_location()) < 50:
            group_traffic_light = closest_traffic_light.get_group_traffic_lights()
            # print("     No of Incoming Lights: ",len(group_traffic_light))
            for traffic_light in group_traffic_light:
                #Now check if traffic light is ahead of us or behind
                traffic_light_location = traffic_light.get_location()
                to_traffic_light_vector = traffic_light_location - current_state["location_vector"]
                dot_product = current_state["forward_vector"].x * to_traffic_light_vector.x + current_state["forward_vector"].y * to_traffic_light_vector.y + current_state["forward_vector"].z * to_traffic_light_vector.z
                magnitude_a = math.sqrt(current_state["forward_vector"].x ** 2 + current_state["forward_vector"].y ** 2 + current_state["forward_vector"].z ** 2)
                magnitude_b = math.sqrt(to_traffic_light_vector.x ** 2 + to_traffic_light_vector.y ** 2 + to_traffic_light_vector.z ** 2)
                cos_theta = dot_product / (magnitude_a * magnitude_b)
                theta = math.acos(cos_theta)
                angle = math.degrees(theta)

                traffic_light_angle = traffic_light.get_transform().rotation.yaw
                traffic_light_state = traffic_light.get_state() 
                if (angle<90) and (-135 <wrap_angle_pi_to_pi(np.deg2rad(ego_angle - traffic_light_angle))< -45):
                    if traffic_light_state == carla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0)
                    elif traffic_light_state == carla.TrafficLightState.Yellow:
                        color = carla.Color(255, 255, 0)
                    else:
                        color = carla.Color(0,255,0)
                    draw_trajectory([traffic_light.get_location().x, traffic_light.get_location().x+2], [traffic_light.get_location().y, traffic_light.get_location().y+2], self.world, 0.5, 1, PLOT_TIME, "line", color)
                    if traffic_light_state != carla.TrafficLightState.Green:
                        my_traffic_light = traffic_light
                        # print("     Red Traffic Light Encountered = Traffic Id: ",my_traffic_light.id)
                        self.current_behavior = "TRAFFIC_LIGHT_STOP"
                        if(self.Ego.is_at_traffic_light() and next_waypoint.is_junction ):
                            draw_trajectory([current_state["x"]], [current_state["y"]], self.world, current_state["z"] + 0.7, 1, 0, "point", carla.Color(255,0,255))

                            self.speed = 0.0
                        else:
                            self.speed = 3.0
                        break
            
        current_lane_obstacles = sorted(current_lane_obstacles, key=lambda Obstacle: Obstacle.delta_s)
        left_lane_obstacles = sorted(left_lane_obstacles, key=lambda Obstacle: Obstacle.delta_s)
        right_lane_obstacles = sorted(right_lane_obstacles, key=lambda Obstacle: Obstacle.delta_s)
        
        overtake_lookahead = min(max(MIN_OVERTAKE_RANGE, future_s(current_state["speed"], current_state["long_acc"], self.overtake_lookahead_time)), MAX_OVERTAKE_RANGE)

        if current_lane_obstacles:
            overtake_delta_s = current_lane_obstacles[0].s - current_state["s"]
            if overtake_delta_s<6 and ego_norm_d == 0:
                self.current_behavior = "EMERGENCY_BRAKE"
            elif 0 < overtake_delta_s < overtake_lookahead and ego_norm_d == 0:
                if not left_lane_obstacles or (left_lane_obstacles and left_lane_obstacles[0].s > current_lane_obstacles[0].s + OVERTAKE_THRESHOLD):
                    waypoint_left=   waypoint.get_left_lane()
                    if waypoint_left!=None and waypoint_left.lane_type == carla.LaneType.Driving and waypoint_left.lane_id * waypoint.lane_id > 0 and incoming == "RIGHT_TURN" and (s_incoming-current_state["s"])>50:
                        self.current_behavior = "OVERTAKE"
                        s, d = self.mission_planner.refrence_path_local.cartesian_to_frenet(waypoint_left.transform.location.x, waypoint_left.transform.location.y)
                        self.mission_planner.re_route(max(s - S_REROUTE_THRESHOLD,0), ego_norm_d + d)
                    else:
                        self.current_behavior = "TAILGATE"
                        self.tailgate_obs = current_lane_obstacles[0]
                        self.was_tailgate = True
                else:
                    self.current_behavior = "TAILGATE"
                    self.tailgate_obs = current_lane_obstacles[0]
                    self.was_tailgate = True
        
        if current_state["speed"] < 0.5:
            # print("     Car is IDLE for:", self.idle_time)
            self.idle_time += time.time() - self.prev_time
        else:
            self.idle_time = 0
        self.prev_time = time.time()
        
        check_idle = 0
        if my_traffic_light:
            if my_traffic_light.get_state() == carla.TrafficLightState.Green:
                check_idle = 1
            else:
                self.idle_time = 0
        else:
            check_idle = 1

        if self.idle_time > 5 and check_idle:
            waypoint_right =   self.get_waypoint_from_frenet(current_state["s"], current_state["d"])[0].get_right_lane()
            if not right_lane_obstacles or (right_lane_obstacles and right_lane_obstacles[0].s > current_lane_obstacles[0].s + OVERTAKE_THRESHOLD):
                if waypoint_right!=None and waypoint_right.lane_type == carla.LaneType.Driving:
                    self.current_behavior = "IDLE_FOR_LONG_TIME"
                    self.speed = current_state["target_speed"]
                    s, d = self.mission_planner.refrence_path_local.cartesian_to_frenet(waypoint_right.transform.location.x, waypoint_right.transform.location.y)
                    self.mission_planner.re_route(max(s - S_REROUTE_THRESHOLD,0), ego_norm_d + d)
                    self.idle_time = 0

        # print(direction)
        if direction == RoadOption.CHANGELANERIGHT:
            if (len(right_lane_hazard_obstacles) !=0):
                self.current_behavior = "HAZARD_ON_PATH"
                s, d = self.mission_planner.refrence_path_local.cartesian_to_frenet(current_state["x"], current_state["y"])
                self.mission_planner.re_route(max(s,0), ego_norm_d)

        elif direction == RoadOption.CHANGELANELEFT:
            if (len(left_lane_hazard_obstacles) !=0):
                self.current_behavior = "HAZARD_ON_PATH"
                s, d = self.mission_planner.refrence_path_local.cartesian_to_frenet(current_state["x"], current_state["y"])
                self.mission_planner.re_route(max(s,0), ego_norm_d)


        if (not current_lane_obstacles or self.current_behavior == "IDLE_FOR_LONG_TIME" or self.current_behavior=="OVERTAKE") and (self.current_behavior!="TRAFFIC_LIGHT_STOP"):
            self.was_tailgate = False
            self.speed = current_state["target_speed"]

        if self.was_tailgate == True and current_lane_obstacles:
            self.speed = self.tailgate_obs.vel
            overtake_delta_s = current_lane_obstacles[0].s - current_state["s"]
            if self.speed < 0.2 and overtake_delta_s > 12:
                self.speed = 0.6

        return self.current_behavior, self.speed
