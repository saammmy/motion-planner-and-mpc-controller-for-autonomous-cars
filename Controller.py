import math
import sys
import time
import glob
import os
import shutil
import numpy as np
from scipy.optimize import minimize, differential_evolution
from matplotlib import pyplot as plt
from collections import deque

from torch.utils.tensorboard import SummaryWriter

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

# # Set up the plot
# plt.ion()  # Turn on interactive mode
# fig_1, plot_traj = plt.subplots()
# x_mpc, y_mpc, x_planner, y_planner = [], [], [], []
# xy_mpc, = plot_traj.plot([], [], 'r-', label="MPC")
# xy_planner, = plot_traj.plot([], [], 'b-', label="Motion Planner Trajectory")
# xy_global, = plot_traj.plot([], [], 'k--', label="Global Path")
# plot_traj.legend([xy_mpc, xy_planner, xy_global], [xy_mpc.get_label(), xy_planner.get_label(), xy_global.get_label()], loc=0)
# plot_traj.set_xlabel("X")
# plot_traj.set_ylabel("Y")

# fig_2, plot_vel = plt.subplots()
# v_mpc, v_planner, time_data = [], [], []
# vt_mpc, = plot_vel.plot([], [], 'r-', label="MPC Velocity")
# vt_planner, = plot_vel.plot([], [], 'b-', label="Motion Planner Velocity")
# plot_vel.legend([vt_mpc, vt_planner], [vt_mpc.get_label(), vt_planner.get_label()], loc=0)
# plot_vel.set_xlabel("time")
# plot_vel.set_ylabel("velocity")

# fig_3, plot_str = plt.subplots()
# str_mpc = []
# strt_mpc, = plot_str.plot([], [], 'g-')
# plot_str.set_ylim(-1.22,1.22)
# plot_str.set_xlabel("time")
# plot_str.set_ylabel("Steering Angle")

# fig_4, plot_acc = plt.subplots()
# acc_mpc, realacc_carla = [], []
# acct_mpc, = plot_acc.plot([], [], 'g-', label="MPC")
# realacct_carla, = plot_acc.plot([], [], 'b-', label="Carla Acceleration")
# plot_acc.legend([acct_mpc, realacct_carla], [acct_mpc.get_label(), realacct_carla.get_label()], loc=0)
# plot_acc.set_ylim(-12,12)
# plot_acc.set_xlabel("time")
# plot_acc.set_ylabel("Acceleration")




class PIDController:
    def __init__(self, vehicle):
        # Set up the PID controller gains
        self.kp_lat = 0.1
        self.ki_lat = 0.0
        self.kd_lat = 0.0
        self.kp_long = 4
        self.ki_long = 0.0
        self.kd_long = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        
        # Store the vehicle and waypoint list
        self.vehicle = vehicle
        self.current_waypoint_index = 1
        self.current_state = []
        self.lon_controller = PIDLongitudnalController()
        self.lat_controller = PIDLateralController()
        self.target_state = None

    def update_trajectory(self, x, y,yaw,velocity):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = velocity
        self.current_waypoint_index = 1
        self.target_state = (self.x[self.current_waypoint_index], self.y[self.current_waypoint_index], self.yaw[self.current_waypoint_index], self.v[self.current_waypoint_index])


    def update(self,current_state):
        self.current_state = current_state

        distance_to_waypoint = math.sqrt((self.current_state["x"] - self.target_state[0])**2 + (self.current_state["y"] - self.target_state[1])**2)
        if distance_to_waypoint < 0.8:
            self.current_waypoint_index += 1

        # target state: x,y,yaw,velocity
        self.target_state = (self.x[self.current_waypoint_index], self.y[self.current_waypoint_index], 
                            self.yaw[self.current_waypoint_index], self.v[self.current_waypoint_index])


        throttle = self.lon_controller.pid_control(self.target_state[3], self.current_state["speed"])
        steer = self.lat_controller.pid_control(self.current_state, [self.target_state[0], self.target_state[1]])
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        self.vehicle.apply_control(control)
        
class PIDLongitudnalController():
    def __init__(self, dt=0.2):
        self._k_p = 1
        self._k_d = 0.7
        self._k_i = 0.7
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def pid_control(self, target_speed, current_speed):
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), 0, 1.0)

class PIDLateralController():
    def __init__(self, dt=0.2):
        self._k_p = 1
        self._k_d = 0
        self._k_i = 0
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def cross2(self,a:np.ndarray,b:np.ndarray)->np.ndarray:
        return np.cross(a,b)
        
    def pid_control(self, current_state, next_point):

        v_begin = [current_state["x"], current_state["y"]]
        v_end = [ v_begin[0] + math.cos(current_state["yaw"]), v_begin[1] + math.sin(current_state["yaw"])]

        v_vec = np.array([v_end[0] - v_begin[0], v_end[1] - v_begin[1], 0])
        w_vec = np.array([ next_point[0] - v_begin[0], next_point[1] - v_begin[1] , 0 ])

        _dot = math.acos(np.clip(np.dot(w_vec,v_vec) /
                                    (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        
        _cross = self.cross2(v_vec,w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class state:
    def __init__(self, X=0, Y=0, YAW=0, V=0, Z=0, STR=0, ACC=0, BETA=0, CTE=0, EYAW=0):
        self.x = X
        self.y = Y
        self.z = Z
        self.yaw = YAW
        self.beta = BETA
        
        self.v = V
        self.str = STR
        self.a = ACC
        self.cte = CTE
        self.eyaw = EYAW

class inputs:
    def __init__(self, str_angle=0, throttle=0, brake=0):
        self.str_angle = str_angle
        self.throttle = throttle
        self.brake = brake

class MPC:
    def __init__(self, vehicle, world, global_x, global_y, dt = 0.2, prediction_horizon = 8, control_horizon = 1):
        self.w_cte = 0.05 #0.05
        self.w_eyaw = 60.0 #60.0
        self.w_dthr = 2500 #2500.0
        self.w_dstr = 1000 #1000.0
        self.w_thr = 1.0 #1.0
        self.w_str = 1.0 #1.0
        self.w_vel = 5.0 #5.0  
        self.w_dvel = 200.0 #200.0

        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.planning_time = 0
        self.method = "graph_tracking"
        if self.method =="throttle_offset":
            self.thr_offset = 0.75
        elif self.method == "pid_control":
            self._k_p = 0.27
            self._k_d = 0.01
            self._k_i = 0.15
        elif self.method == "pure_pid":
            self._k_p = 1
            self._k_d = 0.7
            self._k_i = 0.7
        elif self.method == "graph_tracking":
            self.poly_dict = np.load("./data/poly3_dict.npy", allow_pickle='TRUE').item()

        
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

        # Control Input Bounds
        self.thr_bounds = (-5.0, 5.0)
        self.str_bounds = (-1.22, 1.22)
        self.bounds = None
        self.throttle = 0
        self.steering_angle = 0
        self.brake = 0
        
        # Control Inputs
        self.curr_input = inputs()

        self.vehicle_state = state()
        self.length = 2.89
        self.lr = 1.445
        self.lf = 1.445
        self.coff = []
        global_x = [ -x for x in global_x]
        self.global_x = global_x
        self.global_y = global_y
        self.waypoints = []
        self.waypoints_wrt_vehicle = []
        
        # Store the vehicle and waypoint list
        self.vehicle = vehicle
        self.world = world

        self.mpc_plot = True
        self.exp_name = "Experiment_1"
        self.log_graph_path = "./LogGraphs/" + self.exp_name
        if not os.path.exists(self.log_graph_path):
            os.makedirs(self.log_graph_path)
        else:
            shutil.rmtree(self.log_graph_path + "/", ignore_errors=True)
        self.graph_creator = SummaryWriter(self.log_graph_path)
        self.start_time = round(time.time(),2)
    
    def convert_angle(self, angle):
        angle = np.asarray(angle)
        
        return np.where(angle<0, 2*np.pi+angle, angle)
    
    def normalize_angle(self, angle):
        return angle % (2 * np.pi)

    def angle_diff(self, a, b):
        diff = abs(a-b) % (2 * np.pi)
        return 2 * np.pi - diff if diff > np.pi else diff

    def plot_traj_carla(self, control_inputs, throttle_ip, steer_ip, brake_ip):
        curr_state = self.vehicle_state
        realacc = curr_state.a
        future_v = []
        for itr in range(self.prediction_horizon):
            des_next_state = state(self.waypoints[0,itr],self.waypoints[1,itr],self.waypoints[2,itr],self.waypoints[3,itr])

            next_state = self.get_next_state(curr_state, control_inputs[itr], control_inputs[itr+self.prediction_horizon], des_next_state)
            
            curr_point = carla.Location(x=curr_state.x,y=curr_state.y,z=self.vehicle_state.z+0.2)
            next_point = carla.Location(x=next_state.x,y=next_state.y,z=self.vehicle_state.z+0.2)
            
            self.world.debug.draw_arrow(curr_point, next_point, thickness=0.1, arrow_size=0.1, color=carla.Color(255, 0, 0), life_time=0.3)
            curr_state = next_state
            if itr == 0:
                x = -curr_state.x  # Update the x value
                y = curr_state.y  # Generate a random y value
                v = curr_state.v

                desx = -des_next_state.x
                desy = des_next_state.y
                desv = des_next_state.v
                                
                t = round(time.time(),2) - self.start_time
                str = control_inputs[itr+self.prediction_horizon]
                acc = control_inputs[itr]
            future_v.append(next_state.v)
        self.graph_creator.add_scalar("Steering Angle (Degree)",str*180/np.pi, global_step=t)
        self.graph_creator.add_scalars("Velocity (mph)",{"Velocity Predicted by MPC":v*2.24, "Velocity Proposed by Local Planner":desv*2.24, "Current Velocity in CARLA":self.vehicle_state.v*2.24},global_step=t)
        self.graph_creator.add_scalars("Acceleration",{"Acceleration Predicted by MPC":acc, "Carla Acceleration":realacc}, global_step=t)
        self.graph_creator.add_scalar("Steering Input to Carla",steer_ip, global_step=t)
        self.graph_creator.add_scalar("Throttle Input to Carla",throttle_ip, global_step=t)
        self.graph_creator.add_scalar("Brake Input to Carla", brake_ip, global_step=t)
        

        # print("MPC Future Velocity: ", future_v)
                
        # x_mpc.append(x)
        # y_mpc.append(y)
        # x_planner.append(desx)
        # y_planner.append(desy)

        # xy_mpc.set_data(x_mpc, y_mpc)  # Update the line data
        # xy_planner.set_data(x_planner, y_planner)  # Update the line data
        # xy_global.set_data(self.global_x, self.global_y)
        # plot_traj.relim()  # Update the axes limits
        # plot_traj.autoscale_view()  # Autoscale the view
        # plot_traj.set_aspect('equal')
        # fig_1.canvas.draw()  # Redraw the figure
        # fig_1.canvas.flush_events()  # Flush the GUI events

        # time_data.append(t)

        # v_mpc.append(v)
        # v_planner.append(desv)
        # vt_mpc.set_data(time_data, v_mpc)  # Update the line data
        # vt_planner.set_data(time_data, v_planner)  # Update the line data
        # plot_vel.relim()  # Update the axes limits
        # plot_vel.autoscale_view()  # Autoscale the view
        # fig_2.canvas.draw()  # Redraw the figure
        # fig_2.canvas.flush_events()  # Flush the GUI events

        # str_mpc.append(str)
        # strt_mpc.set_data(time_data, str_mpc)
        # plot_str.relim()  # Update the axes limits
        # plot_str.autoscale_view()  # Autoscale the view
        # fig_3.canvas.draw()  # Redraw the figure
        # fig_3.canvas.flush_events()  # Flush the GUI events

        # acc_mpc.append(acc)
        # realacc_carla.append(realacc)
        # acct_mpc.set_data(time_data, acc_mpc)
        # realacct_carla.set_data(time_data, realacc_carla)
        # plot_acc.relim()  # Update the axes limits
        # plot_acc.autoscale_view()  # Autoscale the view
        # fig_4.canvas.draw()  # Redraw the figure
        # fig_4.canvas.flush_events()  # Flush the GUI events

    def convert_input_to_carla(self, str_angle, target_acc):
        self.steering_angle = str_angle / self.str_bounds[1]
        acc_maped = target_acc / self.thr_bounds[1]

        if self.method == "throttle_offset":
            if acc_maped >= 0:
                self.throttle = min(acc_maped + self.thr_offset,1)
                self.brake = 0
            elif acc_maped < 0 and acc_maped > -self.thr_offset:
                self.throttle = 0
                self.brake = 0
            else:
                self.throttle = 0
                self.brake = min(abs(acc_maped),1)

        elif self.method == "pid_control":
            print("Target Acc", target_acc)
            print("Curr Acceleration", self.vehicle_state.a)
            error = target_acc - min(self.vehicle_state.a,10)
            self._error_buffer.append(error)

            if len(self._error_buffer) >= 2:
                _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
                _ie = sum(self._error_buffer) * self._dt
            else:
                _de = 0.0
                _ie = 0.0

            self.throttle = np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), 0, 1.0)
            print("Throttle: ", throttle)
        
        elif self.method == "pure_pid":
            error = 30 - self.vehicle_state.v
            self._error_buffer.append(error)

            if len(self._error_buffer) >= 2:
                _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
                _ie = sum(self._error_buffer) * self._dt
            else:
                _de = 0.0
                _ie = 0.0

            self.throttle = np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), 0, 1.0)
            print("Throttle: ", throttle)
        
        elif self.method == "graph_tracking":
            delta_min = float('inf')
            throttle = 0.0
            target_vel = self.vehicle_state.v + target_acc * self.dt
            
            if self.waypoints[3,-1] < 0.5:
                # print("I am here")
                # self.throttle -= 0.05
                # if self.throttle <= 0:
                self.brake += 0.1
                self.brake = min(self.brake,1)
                self.throttle = 0
                self.steering_angle = 0
            else:
                for key in self.poly_dict:
                    poly = self.poly_dict[key]
                    acc = np.polyval(poly, target_vel)
                    delta = np.abs(acc - target_acc)
                    if delta<delta_min:
                        delta_min = delta
                        throttle = key
                self.throttle = min(self.throttle+0.1, throttle)
                self.brake = 0

    def update_vehicle_state(self, x=0, y=0, yaw=0, v=0, a=0, z=0, cte=0, eyaw=0):
        self.vehicle_state.x = x
        self.vehicle_state.y = y
        self.vehicle_state.yaw = self.convert_angle(yaw)
        self.vehicle_state.v = v
        self.vehicle_state.a = a
        self.vehicle_state.z = z
        self.vehicle_state.cte = cte
        self.vehicle_state.eyaw = eyaw
    
    def update_waypoints(self, x, y, yaw, v, vehicle_state, dt, planning_time):
        self.update_vehicle_state(vehicle_state["x"], vehicle_state["y"], vehicle_state["yaw"], vehicle_state["speed"], vehicle_state["long_acc"], vehicle_state["z"])
        self.dt = dt
        self.planning_time = planning_time
        self.prediction_horizon = min(int(planning_time/dt),8)
        self.bounds = (self.thr_bounds,)*self.prediction_horizon + (self.str_bounds,)*self.prediction_horizon

        self.waypoints = np.zeros(shape=(4, np.shape(x)[0]-1))
        self.waypoints[0,:] = x[1:]
        self.waypoints[1,:] = y[1:]
        self.waypoints[2,:] = self.convert_angle(yaw[1:])
        self.waypoints[3,:] = v[1:]        
    
    def global_to_vehicle(self, waypoints):
        n_waypoints = np.shape(waypoints)[1]

        waypoints_wrt_vehicle = np.zeros(shape=(4, n_waypoints))
        waypoints_wrt_vehicle[0,:] = np.cos(-self.vehicle_state.yaw)*(waypoints[0,:]-self.vehicle_state.x) - np.sin(-self.vehicle_state.yaw)*(waypoints[1,:]-self.vehicle_state.y)
        waypoints_wrt_vehicle[1,:] = np.sin(-self.vehicle_state.yaw)*(waypoints[0,:]-self.vehicle_state.x) + np.cos(-self.vehicle_state.yaw)*(waypoints[1,:]-self.vehicle_state.y)
        waypoints_wrt_vehicle[2,:] = waypoints[2,:] - self.vehicle_state.yaw
        waypoints_wrt_vehicle[3,:] = waypoints[3,:] # Velocity is not relative to the current of the vehicle

        return waypoints_wrt_vehicle

    def model_rear(self, control_input, curr_state, des_next_state):

        next_state = state()
        next_state.x = curr_state.x + curr_state.v*np.cos(curr_state.yaw)*self.dt
        next_state.y = curr_state.y + curr_state.v*np.sin(curr_state.yaw)*self.dt
        next_state.yaw = curr_state.yaw + curr_state.v/self.length * np.tan(control_input.str_angle)*self.dt
        next_state.v = curr_state.v + control_input.throttle*self.dt

        next_state.cte = (next_state.x - des_next_state.x)**2 + (next_state.y - des_next_state.y)**2
        next_state.eyaw = (next_state.yaw - des_next_state.yaw)**2

        return next_state

    def model_cg(self, control_input, curr_state, des_next_state):

        next_state = state()
        next_state.x = curr_state.x + curr_state.v*np.cos(curr_state.yaw + curr_state.beta)*self.dt
        next_state.y = curr_state.y + curr_state.v*np.sin(curr_state.yaw + curr_state.beta)*self.dt
        next_state.beta = self.normalize_angle(np.arctan((self.lr/self.length)*np.tan(control_input.str_angle)))
        next_state.yaw = self.normalize_angle(curr_state.yaw + curr_state.v/self.lr * np.sin(next_state.beta)*self.dt)
        next_state.v = curr_state.v + control_input.throttle*self.dt
        
        next_state.cte = (next_state.x - des_next_state.x)**2 + (next_state.y - des_next_state.y)**2 
        next_state.eyaw = (self.angle_diff(next_state.yaw, des_next_state.yaw))**2

        return next_state

    def get_next_state(self, curr_state, throttle, str_angle, des_next_state):
        control_input = inputs()
        control_input.throttle = throttle
        control_input.str_angle = str_angle
        
        next_state = self.model_cg(control_input, curr_state, des_next_state)

        return next_state
    
    def get_costs(self,curr_state, next_state, control_inputs, itr):
        cost_cte = self.w_cte*(next_state.cte)
        cost_eyaw = self.w_eyaw*(next_state.eyaw)
        cost_vel = self.w_vel * (next_state.v - self.waypoints[3,-1])**2 #Checking with final value
        cost_thr = self.w_thr*(control_inputs[itr])**2
        cost_str = self.w_str*(control_inputs[itr+self.prediction_horizon])**2
        cost_dvel = self.w_dvel*(next_state.v - curr_state.v)**2
        if itr!=0:
            cost_dthr = self.w_dthr*(control_inputs[itr] - control_inputs[itr-1])**2
            cost_dstr = self.w_dstr*(control_inputs[itr+self.prediction_horizon] - control_inputs[itr+self.prediction_horizon-1])**2
        else:
            cost_dthr = cost_dstr = 0
            
        cost = cost_cte + cost_eyaw + cost_vel + cost_thr + cost_str + cost_dvel + cost_dthr + cost_dstr

        # print("---------------------------")
        # print("Cost cte: ",cost_cte)
        # print("Cost cost_eyaw: ",cost_eyaw)
        # print("Cost cost_vel: ",cost_vel)
        # print("Cost cost_thr: ",cost_thr)
        # print("Cost cost_str: ",cost_str)
        # print("Cost cost_dvel: ",cost_dvel)
        # print("Cost cost_dthr: ",cost_dthr)
        # print("Cost cost_dstr: ",cost_dstr)

        return cost

    def cost_function(self, control_inputs):
        curr_state = self.vehicle_state
        cost = 0

        for itr in range(self.prediction_horizon):
            des_next_state = state(self.waypoints[0,itr],self.waypoints[1,itr],self.waypoints[2,itr])
            next_state = self.get_next_state(curr_state, control_inputs[itr], control_inputs[itr+self.prediction_horizon], des_next_state)
            cost += self.get_costs(curr_state, next_state, control_inputs, itr)
            curr_state = next_state
    
        return cost

    def run_step(self):
        # Transfer the coordinates from global coordinates to vehicle coordinates
        self.waypoints_wrt_vehicle = self.global_to_vehicle(self.waypoints)
        # print(self.waypoints[3,-1])
        
        # Run the minimization
        # print("Sample Time: ", self.dt, "Prediction Horizon: ", self.prediction_horizon, "Local Planning Time:", self.planning_time)
        
        control_inputs = [0,0] * self.prediction_horizon
        control_inputs = minimize(self.cost_function, control_inputs, method='SLSQP' , bounds = self.bounds) #L-BFGS-B
        control_inputs = control_inputs.x
        
       

        for i in range(self.control_horizon):
            s = time.time()
            self.convert_input_to_carla(control_inputs[i+self.prediction_horizon], control_inputs[i])
            if self.mpc_plot:
                self.plot_traj_carla(control_inputs, self.throttle, self.steering_angle, self.brake)
                # print("Local Planner Future Velocities: ", self.waypoints[3,:])
                plt.show()
            # print("Waypoints: ", self.waypoints)
            # print("Waypoints wrt vehicle ", self.waypoints_wrt_vehicle)
            # print("Control Inputs ", control_inputs)
            # print("Throttle: ", throttle)
            # print("Steering: ", steering_angle)
            # print("Brake: ", brake)

            control = carla.VehicleControl(self.throttle, self.steering_angle, self.brake)
            self.vehicle.apply_control(control)
        
            