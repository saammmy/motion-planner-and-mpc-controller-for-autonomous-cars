CLIENT = 2000
TOWN = 'Town06'

RANDOM_POINT = False
SPAWN_OBSTACLE = False

if TOWN == 'Town06':
    START_X = 600.0
    START_Y = -17.5
    START_YAW = 180.0
    END_X = 0.0
    END_Y = 600
    OBS1_X = 550
    OBS1_Y = -17.75
    OBS1_YAW = 180.0
    OBS2_X = 550
    OBS2_Y =  -21.5 #-14.75 #
    OBS2_YAW = 180.0
elif TOWN == 'Town05':
    START_X = 189.74
    START_Y = -10.03
    START_YAW = 90.0
    # END_X = -99.3
    # END_Y = 189
    END_X = -17.5
    END_Y = 600
    OBS1_X = 100.0
    OBS1_Y = 188.0
    OBS1_YAW = 180.0

MAX_ACC = 5
MIN_ACC = -5

COMPUTATION_TIME = 0.06 # CALCULATE COMPUTATION TIME FOR YOUR PC
SAMPLE_TIME = 0.2 # MAKE SURE IT IS > COMPUTATION_TIME
CONTROL_HORIZON = 1
PREDICTION_HORIZON = 8
PLANNING_DURATION = SAMPLE_TIME * PREDICTION_HORIZON
TICK_TIME = 0.01
SYNCHRONOUS_MODE = False

LOCAL_GLOBAL_PLAN_MIN = 200
LOCAL_GLOBAL_PLAN_MAX = 400

if SYNCHRONOUS_MODE:
    PLOT_TIME = TICK_TIME
elif CONTROL_HORIZON>1:
    PLOT_TIME = SAMPLE_TIME # DEfine based on code speed
else:
    PLOT_TIME = COMPUTATION_TIME
