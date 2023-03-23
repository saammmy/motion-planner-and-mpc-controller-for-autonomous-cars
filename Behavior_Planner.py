from math import sqrt
import Local_Planner
from misc import get_speed
import numpy as np
def calculate_distance(current_state,vehicle):
    return sqrt((current_state['x'] - vehicle.get_location().x)**2 + (current_state['y'] - vehicle.get_location().y)**2)


class BehaviorPlanner:
    def __init__(self,local_planner):
        self.current_behavior = "FOLLOW_LANE"
        self.local_planner = local_planner

    
    def get_next_behavior(self,current_state, lookahead_path, vehicles):


        closest_distance = float('inf')
        closest_vehicle = None
        closest_vehicle_s = None
        closest_vehicle_d = None

        if len(vehicles)!=0:
            for vehicle in vehicles:
                s, d = self.local_planner.cartesian_to_frenet(vehicle.get_location().x, vehicle.get_location().y)
                if 0< s - current_state["s"] < lookahead_path:
                    dist = calculate_distance(current_state, vehicle)
                    if(dist<=closest_distance):
                        closest_distance = dist
                        closest_vehicle = vehicle
                        closest_vehicle_s = s
                        closest_vehicle_d = d


        if closest_distance == float('inf'):
            self.current_behavior = "follow_lane"
            return self.current_behavior, None, 0.0
        
        else:

            self.current_behavior = "lane_change"
            return self.current_behavior, None, -3.5
            
















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

