# motion-planner-and-mpc-controller-for-autonomous-cars

Note: This repository is under development and will be used for code sharing purposes
The main goal of the project include
1. To develop a motion planner and an MPC controller to follow the given trajectory.
2. Compare the different types of algorithms for Motion Planning
3. Compare the performance of MPC based control when compared to PID control

## Run Instructions

The is based on CARLA 0.9.10 version which can be quick installled as follows:

`````
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
sudo apt-get update
sudo apt-get install carla-simulator=0.9.10-1
cd /opt/carla-simulator
`````

The Repository should be cloned in the PythonAPI folder in carla-simulator:

`````
cd /opt/carla-simulator/PythonAPI
git clone https://github.com/saammmy/motion-planner-and-mpc-controller-for-autonomous-cars.git
`````


To run the main code execute the below comands:

`````
python main.py --no-of-walker 0 --no-of-vehicles 0
`````
