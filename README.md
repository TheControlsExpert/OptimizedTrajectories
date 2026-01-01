# OptimizedTrajectories

## Simulator.py


In the main() method, the request final position and initial position are turned into joint-angles using inverse-kinematics, where the solution that makes the CG as high as possible (to not hit the gearbox underneath) is chosen. Then, it is fed into Ipopt and Casadi which gives an optimal trajectory.

When simulating, at each time-step (dt = 0.001) the dynamics of the arms are solved using RKDP integration, while using parameters of the arm from class ArmConstantsReal and the request current in the step() method. The optimized trajectory is sampled at each control time-step (dt = 0.005), and setpoint acceleration, velocity, and position is found using finite differences. Through the feed_forward() method, these setpoints are turned into a torque and afterwards a current applied - since the motors are using TorqueCurrentFOC control. Additionally, a P-controller is used to help the arm keep up with the position setpoints of the trajectory.

Results are vizualized using Matplotlib's FuncAnimation() 

## Optimizer.py

In __init__, the problem is defined: between the final angles and initial angles (which we previously found using inv. kinematics), create n=25 interior points with constant dt between them. Using finite differences, apply acceleration and jerk constraints at each interior point. Similarly, using finite differences, find the acceleration, position, and velocity requested at each interior point. Using feed_forward_casadi() in Simulator.py, turn those values into a requested current and voltage (using DCMotorConstants) and constrain those values as well. Finally, convert each interior point's angles into x,y positions using forward_kinematics_casadi() in Simulator.py and apply spatial constraints. Minimize total time.

In solve(), an initial guess is made for the interior angles using linear interpolation between the initial and final angle of the trajectory. Then, Ipopt solves!




