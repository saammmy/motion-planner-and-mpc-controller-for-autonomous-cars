import numpy as np

def ms_to_mph(speed):
        return speed * 2.24
    
def mph_to_ms(speed):
    return speed/2.24

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
