import math
import numpy as np
def ramp_profile(curr_vel, target_vel, ds):

    acc = (target_vel**2 - curr_vel**2)/(2*ds[-1])

    v = np.sqrt(curr_vel**2 + 2*acc*ds)

    return v


class RampGenerator:
    def __init__(self, max_accel= 2.0, min_accel=-3):
        self.min_accel = min_accel
        self.max_accel = max_accel

    def plan(self, curr_vel, target_vel, ds):
        req_accel = (target_vel**2 - curr_vel**2)/(2*ds[-1])
        if curr_vel < target_vel:
            if req_accel > self.max_accel:
                print("Breaching maximum acceleation limits")
                req_accel = self.max_accel
                target_vel = math.sqrt(curr_vel**2 + 2*req_accel*ds[-1])
        else:
            if req_accel < self.min_accel:
                # print("Breaching minimum acceleation limits")
                req_accel = self.min_accel
                target_vel = math.sqrt(curr_vel**2 + 2*req_accel*ds[-1])

        velocity = np.sqrt(curr_vel**2 + 2.0*req_accel*ds)

        return velocity, req_accel

