from math import sqrt
import Local_Planner
from misc import get_speed
import numpy as np
from  utils import *

def future_s(speed, acc, t):
    return speed*t + 0.5*acc*t**2

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
    def __init__(self,local_planner, Ego):
        self.current_behavior = "FOLLOW_LANE"
        self.local_planner = local_planner
        self.Ego = Ego
        self.lookahead_time = 5
        self.react_to_collision = 8
        self.overtake_lookahead = 3
        self.current_d = 0

    def assignLane(self, d, lane_width=3.5):

        if d < 1.75 and d > -1.75:
            return 0
        elif d > 1.75 and d < 5.25:
            return -1
        elif d > -5.25 and d < -1.75:
            return 1
        else:
            return None
        
    def get_next_behavior(self,current_state,obstacles):
        target_speed = current_state["target_speed"]
        target_s = None

        left_lane_vehicles= []
        right_lane_vehicles = []
        current_lane_vehicles = []

        active_obstacles = []
        
        lane_width = 3.5
        semi_lane_width = 1.75
        ego_lane = self.assignLane(current_state["d"])
        
        lookahead = max(30,future_s(current_state["speed"], current_state["long_acc"], self.lookahead_time))

        for obstacle in obstacles:
            if(obstacle.id == self.Ego.id):
                continue
            s, d = self.local_planner.cartesian_to_frenet(obstacle.get_location().x, obstacle.get_location().y)
            delta_s = s - current_state["s"]
            if 0< delta_s < lookahead:
                lane = self.assignLane(d)
                # print("     OBSTACLE DETECTED: ", obstacle.id)
                active_obstacles.append(Obstacle(obstacle, lane, s, d, delta_s))

        self.current_behavior = "FOLLOW LANE"
        for obstacle in active_obstacles:
            if obstacle.lane == ego_lane:
                relative_vel = current_state["speed"] - obstacle.vel
                # collision_time = time_to_collision(relative_vel, obstacle.delta_s)
                overtake_lookahead = max(current_state["s"]+15,current_state["s"] + future_s(current_state["speed"], current_state["long_acc"], self.overtake_lookahead))
                if  overtake_lookahead > obstacle.s:
                    # Entering Overtake and Tailgate Check
                    self.current_behavior = "OVERTAKE"
                    self.current_d = -3.5
                    if ego_lane != 1:
                        for obs in active_obstacles:
                            if obs.id != obstacle.id:
                                if obs.lane == 1:
                                    self.current_behavior = "TAILGATE"
                                    target_speed = obstacle.vel
                                    target_s = obstacle.s - 8.0
                                    self.current_d = 0
                                    break
                        
        return self.current_behavior, target_s, self.current_d, target_speed
            
















        # elif closest_distance != float('inf') and current_state["speed"] < current_state["target_speed"] - 2:

        #     if closest_distance < 10:
        #         self.current_behavior = "SAFETY_STOP"
        #         return self.current_behavior, None, None, None
            
        #     if get_speed(closest_vehicle) == 0:
        #         self.current_behavior = "STOP"
        #         return self.current_behavior, closest_vehicle_s - 8 , 0 , 0

        #     else :
        #         self.current_behavior = "FOLLOW_LEAD"
        #         obs_vel = closest_vehicle.get_velocity()
        #         obs_acc = closest_vehicle.get_acceleration()
        #         obs_transform = closest_vehicle.get_transform()
        #         transform_matrix = np.linalg.inv(obs_transform.get_matrix())
        #         obs_vel = transform_matrix @ np.array([obs_vel.x, obs_vel.y, obs_vel.z, 0.0])
        #         obs_acc = transform_matrix @ np.array([obs_acc.x, obs_acc.y, obs_acc.z, 0.0])

        #         return self.current_behavior, closest_vehicle_s-8, round(obs_vel[0],2), round(obs_acc[0],2)
        
        # elif closest_distance != float('inf') and current_state["speed"] > current_state["target_speed"] - 2:
        #     self.current_behavior = "lane_change"
        #     return None, None

