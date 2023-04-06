from math import sqrt
import Local_Planner
from misc import get_speed
import numpy as np
from  utils import *
from params import *
from Mission_Planner import MissionPlanner



def future_s(speed, acc, t):
    return speed*t + 0.5*max(min(acc,MAX_ACC),-5)*t**2

def time_to_collision(relative_speed, delta_s):
    return delta_s/relative_speed

def calculate_distance(current_state,vehicle):
    return sqrt((current_state['x'] - vehicle.get_location().x)**2 + (current_state['y'] - vehicle.get_location().y)**2)

class Obstacle:
    def __init__(self, obs, lane=None, s=None, d=None, delta_s=None):
        
        self.obstacle = obs
        self.id = obs.id
        self.lane = lane
        self.s = s
        self.d = d
        self.delta_s = delta_s
        self.vel = obs.get_velocity()
        self.acc = obs.get_acceleration()
        self.vel = sqrt(self.vel.x**2 + self.vel.y**2)
        self.acc = sqrt(self.acc.x**2 + self.acc.y**2)
        # transform = obs.get_transform()
        # transform_matrix = np.linalg.inv(transform.get_matrix())
        # self.vel = transform_matrix @ np.array([self.vel.x, self.vel.y, self.vel.z, 0.0])
        # self.acc = transform_matrix @ np.array([self.acc.x, self.acc.y, self.acc.z, 0.0])
        self.intent = None



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
    
    def is_reached_goal(self, current_s_local):
        curremt_s_global = current_s_local + self.mission_planner.s_last_wrt_global
        if  self.mission_planner.refrence_path_global.s[-1] - curremt_s_global < GOAL_REACH_THRESHOLD:
            return True
        return False
    
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

    def assignLane(self, d, lane_width=LANE_WIDTH):

        if d < lane_width/2 and d > -lane_width/2:
            return 0
        elif d > lane_width/2 and d < lane_width*3/2:
            return -1
        elif d > -lane_width*3/2 and d < -lane_width/2:
            return 1
        else:
            return None
        
    def get_next_behavior(self,current_state, local_path,obstacles):
        self.speed = current_state["speed"]

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

        ego_lane = self.assignLane(current_state["d"])
        ego_norm_d = self.normalize_d(current_state["d"])
        # print("Current d", current_state["d"])
        lookahead = min(max(MIN_OBS_RAD,future_s(current_state["speed"], current_state["long_acc"], self.lookahead_time)),MAX_OBS_RAD)
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

                # print(f"angle_to_obstacle: y:{obstacle.get_location().y}, angle:{angle}")
                # check if the obstacle is directly ahead of the ego vehicle
                if angle < 90.0:
                    obs_s, obs_d = local_path.cartesian_to_frenet(obstacle.get_location().x, obstacle.get_location().y)
                    obs_d_norm = self.normalize_d(obs_d)
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
                        
            
        current_lane_obstacles = sorted(current_lane_obstacles, key=lambda Obstacle: Obstacle.delta_s)
        left_lane_obstacles = sorted(left_lane_obstacles, key=lambda Obstacle: Obstacle.delta_s)
        right_lane_obstacles = sorted(right_lane_obstacles, key=lambda Obstacle: Obstacle.delta_s)

        # print("CURR ",current_lane_obstacles)
        # print("RIGHT ",right_lane_obstacles)
        # print("LEFT ",left_lane_obstacles)
        
        overtake_lookahead = min(max(MIN_OVERTAKE_RANGE, future_s(current_state["speed"], current_state["long_acc"], self.overtake_lookahead_time)), MAX_OVERTAKE_RANGE)

        if current_lane_obstacles:
            overtake_delta_s = current_lane_obstacles[0].s - current_state["s"]
            if overtake_delta_s<10:
                self.current_behavior = "EMERGENCY_BRAKE"
            elif 0 < overtake_delta_s < overtake_lookahead:
                if not left_lane_obstacles or (left_lane_obstacles and left_lane_obstacles[0].s > current_lane_obstacles[0].s + OVERTAKE_THRESHOLD):
                    self.current_behavior = "OVERTAKE"
                    x, y = self.mission_planner.refrence_path_local.frenet_to_cartesian(current_state["s"], current_state["d"])
                    argmin = get_closest_waypoint(self.mission_planner.refrence_path_local.waypoints_x, self.mission_planner.refrence_path_local.waypoints_y, x, y)
                    waypoint_left = self.mission_planner.refrence_path_local.waypoints[argmin].get_left_lane()
                    draw_trajectory([waypoint_left.transform.location.x], [waypoint_left.transform.location.y], self.world, 0.5, 1, 0, "point")
                    print(waypoint_left.lane_type)
                    if waypoint_left!=None and waypoint_left.lane_type == "Driving":
                        s, d = self.mission_planner.refrence_path_local.cartesian_to_frenet(waypoint_left.transform.location.x, waypoint_left.transform.location.y)
                        self.mission_planner.re_route(s, ego_norm_d + d)
                        # self.mission_planner.re_route(current_state["s"], ego_norm_d - LANE_WIDTH)
                    else:
                        self.current_behavior = "TAILGATE"
                        self.tailgate_obs = current_lane_obstacles[0]
                        self.was_tailgate = True
                else:
                    self.current_behavior = "TAILGATE"
                    self.tailgate_obs = current_lane_obstacles[0]
                    self.was_tailgate = True
        
        if not current_lane_obstacles or self.current_behavior=="OVERTAKE":
            self.was_tailgate = False
            self.speed = current_state["target_speed"]

        if self.was_tailgate == True:
            self.speed = self.tailgate_obs.vel
        return self.current_behavior, self.speed
