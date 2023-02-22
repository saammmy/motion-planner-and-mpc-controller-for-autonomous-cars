from math import sqrt


def calculate_distance(current_state,vehicle):
    return sqrt((current_state['x'] - vehicle['x'])**2 + (current_state['y'] - vehicle['y'])**2)


class BehaviorPlanner:
    def __init__(self,distance_threshold=20):
        self.current_behavior = "FOLLOW_LANE"
        self.distance_threshold = distance_threshold
    
    def get_next_behavior(self,current_state, lookahead_path, vehicles):

        current_lane = current_state['lane']
        
        closest_distance = float('inf')
        closest_vehicle = []
        for vehicle in vehicles:
            if  vehicle['lane'] == current_lane:
                dist = calculate_distance(current_state, vehicle) < closest_distance
                if(dist<=closest_distance):
                    closest_distance = dist
                    closest_vehicle = vehicle
        
        if closest_distance <= self.distance_threshold:
            return "lane_change"
        else:
            return "lane_follow"

