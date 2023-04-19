import Local_Planner
import numpy as np
a = Local_Planner.QuinticPolynomial(10,10, 0,11,10,0,0.1)

s = [a.calc_pos(t) for t in np.arange(0,0.2,0.1)]
print(s)
