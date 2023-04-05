import numpy as np
from params import *
import math

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