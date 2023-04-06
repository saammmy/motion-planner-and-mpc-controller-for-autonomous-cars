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
    
    return np.argmin(np.hypot(dx,dy))