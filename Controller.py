import math
import sys
import glob
import os
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

class PIDController:
    def __init__(self, vehicle):
        # Set up the PID controller gains
        self.kp_lat = 1.0
        self.ki_lat = 0.0
        self.kd_lat = 0.0
        self.kp_long = 1.0
        self.ki_long = 0.0
        self.kd_long = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        
        # Store the vehicle and waypoint list
        self.vehicle = vehicle
        self.current_waypoint_index = 0
        self.current_state = []


    def update_trajectory(self, x, y,yaw,velocity):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = velocity
        self.current_waypoint_index = 1


    def update(self,current_state):
        self.current_state = current_state
        # target state: x,y,yaw,velocity
        target_state = (self.x[self.current_waypoint_index], self.y[self.current_waypoint_index], 
                            self.yaw[self.current_waypoint_index], self.v[self.current_waypoint_index])

        lateral_error = (self.current_state["y"] - target_state[1]) * math.cos(self.current_state["yaw"]) - (self.current_state["x"] - target_state[0]) * math.sin(self.current_state["yaw"])
        longitudinal_error = target_state[3] - self.current_state["speed"]
        distance_to_waypoint = math.sqrt((self.current_state["x"] - target_state[0])**2 + (self.current_state["y"] - target_state[1])**2)
        if distance_to_waypoint < 0.5:
            self.current_waypoint_index += 1
        
        steering_angle = (self.kp_lat * lateral_error) + (self.ki_lat * lateral_error) + (self.kd_lat * lateral_error)

        throttle = (self.kp_long * longitudinal_error) + (self.ki_long * longitudinal_error) + (self.kd_long * longitudinal_error)

        control = carla.VehicleControl(throttle, steering_angle)
        self.vehicle.apply_control(control)

class state:
    def __init__(self, X=0, Y=0, YAW=0, V=0, CTE=0, EYAW=0):
        self.x = X
        self.y = Y
        self.yaw = YAW
        self.v = V
        self.cte = CTE
        self.eyaw = EYAW

class inputs:
    def __init__(self, str_angle=0, throttle=0, brake=0):
        self.str_angle = str_angle
        self.throttle = throttle
        self.brake = brake

class MPC:
    def __init__(self, vehicle, dt = 0.2, prediction_horizon = 10, control_horizon = 1):
        self.w_cte = 100.0
        self.w_eyaw = 100.0
        self.w_dthr = 1.0
        self.w_dstr = 1.0
        self.w_thr = 1.0
        self.w_str = 1.0
        self.w_vel = 50.0

        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        # Control Input Bounds
        self.thr_bounds = (0.0, 1.0)
        self.str_bounds = (-1.22, 1.22)
        self.bounds = (self.thr_bounds, self.str_bounds)*self.prediction_horizon

        # Control Inputs
        self.curr_input = inputs()

        self.vehicle_state = state()
        self.length = 3
        self.coff = []
        self.waypoints = []
        self.waypoints_wrt_vehicle = []
        
        # Store the vehicle and waypoint list
        self.vehicle = vehicle
    
    def update_vehicle_state(self, x=0, y=0, yaw=0, v=0, cte=0, eyaw=0):
        self.vehicle_state.x = x
        self.vehicle_state.y = y
        self.vehicle_state.yaw = yaw
        self.vehicle_state.cte = cte
        self.vehicle_state.eyaw = eyaw
    
    def update_waypoints(self, x, y, yaw, v):
        self.waypoints = np.zeros(shape=(4, np.shape(x)[0]))
        self.update_vehicle_state(x[0], y[0], yaw[0], v[0])
        self.waypoints[0,:] = x
        self.waypoints[1,:] = y
        self.waypoints[2,:] = yaw
        self.waypoints[3,:] = v        
    
    def global_to_vehicle(self, waypoints):
        n_waypoints = np.shape(waypoints)[1]

        waypoints_wrt_vehicle = np.zeros(shape=(4, n_waypoints))
        waypoints_wrt_vehicle[0,:] = np.cos(-self.vehicle_state.yaw)*(waypoints[0,:]-self.vehicle_state.x) - np.sin(-self.vehicle_state.yaw)*(waypoints[1,:]-self.vehicle_state.y)
        waypoints_wrt_vehicle[1,:] = np.sin(-self.vehicle_state.yaw)*(waypoints[0,:]-self.vehicle_state.x) + np.cos(-self.vehicle_state.yaw)*(waypoints[1,:]-self.vehicle_state.y)
        waypoints_wrt_vehicle[2,:] = waypoints[2,:] - self.vehicle_state.yaw
        waypoints_wrt_vehicle[3,:] = waypoints[3,:] # Velocity is not relative to the current of the vehicle

        return waypoints_wrt_vehicle

    def model(self, control_input, curr_state):

        next_state = state()
        next_state.x = curr_state.x + curr_state.v*np.cos(curr_state.yaw)*self.dt
        next_state.y = curr_state.y + curr_state.v*np.sin(curr_state.yaw)*self.dt
        next_state.yaw = curr_state.yaw + curr_state.v/self.length * control_input.str_angle*self.dt
        next_state.v = curr_state.v + control_input.throttle*self.dt

        yaw_desired = np.arctan(self.coff[2]+2*self.coff[1]*curr_state.x + 3*self.coff[0]*curr_state.x**2)
        next_state.cte = np.polyval(self.coff,curr_state.x) - curr_state.y + (curr_state.v*np.sin(curr_state.eyaw)*self.dt)
        next_state.eyaw = curr_state.yaw - yaw_desired + (curr_state.v/self.length*control_input.str_angle*self.dt)

        return next_state

    def cost_function(self,control_inputs):
        curr_state = self.vehicle_state
        control_input = inputs()
        cost = 0
        for itr in range(0,(self.prediction_horizon)*2,2):
            control_input.throttle = control_inputs[itr]
            control_input.str_angle = control_inputs[itr+1]

            next_state = self.model(control_input, curr_state)
            cost += self.w_cte*(next_state.cte)**2 + self.w_eyaw*(next_state.eyaw)**2 + self.w_vel * (next_state.v - self.waypoints_wrt_vehicle[3,int(itr/2)+1])**2 + \
                    self.w_thr*(control_input.throttle)**2 + self.w_dthr*(control_input.throttle - control_inputs[itr-2])**2 + \
                    self.w_str*(control_input.str_angle)**2 + self.w_dstr*(control_input.str_angle - control_inputs[itr-1])**2
        return cost

    def run_step(self):
        # Transfer the coordinates from global coordinates to vehicle coordinates
        self.waypoints_wrt_vehicle = self.global_to_vehicle(self.waypoints)

        # Fit a cubic polynomial to the waypoints
        self.coff = np.polyfit(self.waypoints_wrt_vehicle[0,:], self.waypoints_wrt_vehicle[1,:], 3)

        # Run the minimization
        control_inputs = [self.curr_input.throttle, self.curr_input.str_angle] * self.prediction_horizon
        control_inputs = minimize(self.cost_function, control_inputs, method='SLSQP', bounds = self.bounds)
        control_inputs = control_inputs.x

        for i in range(self.control_horizon):
            if control_inputs[i*2] > 0:
                throttle = control_inputs[i*2]
                brake = 0
            else:
                throttle = 0
                brake = control_inputs[i*2]
            steering_angle = control_inputs[i*2 + 1]
            # print("Waypoints: ", self.waypoints)
            # print("Waypoints wrt vehicle ", self.waypoints_wrt_vehicle)
            print("Throttle: ", throttle)
            print("Steering: ", steering_angle)
            print("Brake: ", brake)

            control = carla.VehicleControl(throttle, steering_angle, brake)
            self.vehicle.apply_control(control)



        


        
            