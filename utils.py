import numpy as np
from params import *
import math
import sys
import glob
import os
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
        self.vel = math.sqrt(self.vel.x**2 + self.vel.y**2)
        self.acc = math.sqrt(self.acc.x**2 + self.acc.y**2)
        # transform = obs.get_transform()
        # transform_matrix = np.linalg.inv(transform.get_matrix())
        # self.vel = transform_matrix @ np.array([self.vel.x, self.vel.y, self.vel.z, 0.0])
        # self.acc = transform_matrix @ np.array([self.acc.x, self.acc.y, self.acc.z, 0.0])
        self.intent = None

class TrafficLights:
    def __init__(self, traffic_lights):
        self.lights = traffic_lights
        self.x_list = []
        self.y_list = []
        
        for traffic_light in traffic_lights:
            self.x_list.append(traffic_light.get_location().x)
            self.y_list.append(traffic_light.get_location().y)

def ms_to_mph(speed):
        return speed * 2.24
    
def mph_to_ms(speed):
    return speed/2.24

def kmph_to_ms(speed):
    return speed*0.277778

def m_to_feet(d):
    return d*3.28

def convert_angle(angle):
    angle = np.asarray(angle)
    return np.where(angle<0, 2*np.pi+angle, angle)

def wrap_angle_pi_to_pi(angle):
    angle = angle + math.pi  # shift angle range to [0, 2*pi)
    angle = angle % (2*math.pi)  # wrap angle in range [0, 2*pi)
    if angle < 0:
        angle = angle + 2*math.pi  # convert negative angle to positive equivalent
    angle = angle - math.pi  # shift angle range back to [-pi, pi)
    return np.rad2deg(angle)

def normalize_angle(angle):
    return angle % (2 * np.pi)

def angle_diff(a, b):
    diff = abs(a-b) % (2 * np.pi)
    return 2 * np.pi - diff if diff > np.pi else diff

def future_s(speed, acc, t):
    return speed*t + 0.5*max(min(acc,MAX_ACC),-5)*t**2

def distance(waypoint1, waypoint2):

    x = waypoint1.transform.location.x - waypoint2.transform.location.x
    y = waypoint1.transform.location.y - waypoint2.transform.location.y

    return math.sqrt(x * x + y * y)

def dist_xy(x1,y1,x2,y2):
    dx = x1 - x2
    dy = y1 - y2

    return math.sqrt(dx**2 + dy**2)

def draw_trajectory(x, y, world, z, length, time, type="point", color = carla.Color(255, 0, 0)):
        for i in range(length):
            start_point = carla.Location(x=x[i],y=y[i],z= z)
            if type == "point":
                world.debug.draw_point(start_point,size=0.075, color=color, life_time=time)
            else:
                next_point = carla.Location(x=x[i+1], y=y[i+1], z= z)
                if type == "arrow":
                    world.debug.draw_arrow(start_point, next_point, thickness=0.1, arrow_size=0.1, color=color, life_time=time)
                elif type == "line":
                    world.debug.draw_line(start_point, next_point, thickness=0.1, color=color, life_time=time)

def get_closest_waypoint(waypoints_x, waypoints_y, x, y):
    dx = np.array(waypoints_x)-x
    dy = np.array(waypoints_y)-y
    
    return int(np.argmin(np.hypot(dx,dy)))


