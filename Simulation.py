from cmath import rect
import sys

import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.signal import StateSpace
import matplotlib.animation as animation
import json
import keyboard
import frccontrol as frc
import math
from collections import deque
from Optimizer import Optimizer
import casadi as ca




class ArmSim:

    def __init__(self, dt, xyparameters = [0,0]):
        self.distance_to_reef = 0.1
        self.dt = dt
        self.transfom3d_bottomshaft = [0.249, 0.125]
        self.constants = ArmConstantsReal()
        self.constants_optimization = ArmConstantsCAD()
        self.motor_constants = DCMotorConstants()
        start_state = np.concatenate((self.inv_kinematics(np.array([xyparameters[0], xyparameters[1]])), np.array([0,0]))).reshape((-1,1))
        self.state = start_state
        self.target = np.array([xyparameters[0], xyparameters[1]]).reshape((-1, 1))
        self.lastError = np.array([0,0]).reshape((-1, 1))
        self.ErrorDerivative = np.array([0.0,0.0]).reshape((-1, 1))
        

        # state_labels = [("Angle Bottom", "rad"), ("Angle Top", "rad"), ("Angular velocity Bottom", "rad/s"), ("Angular velocity Top", "rad/s")]
        # u_labels = [("Current Bottom", "A"), ("Current Top", "A")]
        # self.set_plot_labels(state_labels, u_labels)
       

        self.t = 0.0
        M,C,G = self.get_dynamics_matrices(start_state)

        self.u_k = np.zeros((2,1))
        self.u_ff = np.zeros((2,1))
        self.kP_bottom = 950
        self.kP_top = 400
        

        self.kP2 = 120
        self.kD = 0

        self.log = frc.Trajectory(np.array([self.t]), np.concatenate((start_state, np.zeros((2,1)))).reshape(-1,1))
                                                                     
        #represents current added from feedback
       

    def step(self):
        
        self.state = frc.rkdp(self.dynamics_true, self.state, self.u_ff + self.u_k, self.dt)
        self.t+= self.dt
    
    def feed_forward(self, states, accels = np.zeros((2,1))):
        (M, C, G) = self.get_dynamics_matrices_CAD(states)
        omegas = states[2:4]
        torque = M @ accels + C @ omegas + G

      

        return np.array([torque[0] / DCMotorConstants.kT_bottomArm, torque[1] / DCMotorConstants.kT_topArm]).reshape((-1,1))
        
    def feed_forward_casadi(self, states, accels):
        (M, C, G) = self.get_dynamics_matrices_casadi(states)

        M_times_acceleration = (
            M[0,0] * accels[0] + M[0,1] * accels[1],
            M[1,0] * accels[0] + M[1,1] * accels[1],
        )
        C_times_velocity = (
            C[0,0] * states[2] + C[0,1] * states[3],
            C[1,0] * states[2] + C[1,1] * states[3],
        )
        current = ca.vertcat(
            (1/DCMotorConstants.kT_bottomArm)*(M_times_acceleration[0] + C_times_velocity[0] + G[0]),
            (1/DCMotorConstants.kT_topArm)*(M_times_acceleration[1] + C_times_velocity[1] + G[1]))
        
       
        return current

    
   
    
    def dynamics_true(self, state, u):
        #tanh avoids the discontinuity of signum function
        #tanh is used to add frictional torque opposing motion

        #previous friction model: -3 * (np.sign(state[2:4])) -> discontinuous
        torque = np.array([u[0]*DCMotorConstants.kT_bottomArm, u[1]*DCMotorConstants.kT_topArm]).reshape((-1,1))  
        -3 * (np.tanh(100 * state[2:4]))

       
        #my solution
        #torque = u*self.constants.MotorKt + np.array(Math.sign(state[2] * (-0.5)), Math.sign(state[3] * (-0.5)))
        (M, C, G) = self.get_dynamics_matrices(state)
        omega_vec = state[2:4]
        accel_vector = np.linalg.inv(M) @ (torque - C @ omega_vec - G)
        state_derivative_vector = np.concatenate((omega_vec, accel_vector))
        error = self.target - self.state[0:2]
        self.ErrorDerivative = (error - self.lastError) / self.dt
       #print(self.ErrorDerivative)
        self.lastError = error
       
        self.log.insert(self.t, np.concatenate((self.state, error)).reshape(-1,1))

     
       # if (abs(accel_vector[0]) < 0.5 and abs(accel_vector[1]) < -0.5):
       #    print(G)
       
        return state_derivative_vector
    
    def inv_kinematics(self, state, invert = False):
        x, y = state[:2]
        x_adjusted = x - self.transfom3d_bottomshaft[0]
        y_adjusted = y - self.transfom3d_bottomshaft[1]

      
        theta2 = np.arccos(
            (x_adjusted * x_adjusted + y_adjusted * y_adjusted - (self.constants.length_bottom * self.constants.length_bottom + self.constants.length_top * self.constants.length_top))
            / (2 * self.constants.length_bottom * self.constants.length_top)
        )

    #always pick the solution where the arm is "above" the gearbox 
        theta2 = theta2 * -np.sign(x_adjusted)

        if invert:
            theta2 = -theta2

        theta1 = np.arctan2(y_adjusted, x_adjusted) - np.arctan2(
            self.constants.length_top * np.sin(theta2), self.constants.length_bottom + self.constants.length_top * np.cos(theta2)
        )

        return theta1, theta2
    
    def forward_kinematics_casadi(self, state):
        theta1, theta2 = state[0], state[1]
        #print(theta1, theta2)
        joint2 = ca.vertcat(self.constants.length_bottom*ca.cos(theta1)+self.transfom3d_bottomshaft[0], self.constants.length_bottom*ca.sin(theta1)+self.transfom3d_bottomshaft[1])
        end_eff = joint2 + ca.vertcat(self.constants.length_top*ca.cos(theta1 + theta2), self.constants.length_top*ca.sin(theta1 + theta2))
        #print(joint2, end_eff)
        return (joint2, end_eff)


    def forward_kinematics(self, state):
        [theta1, theta2] = state[:2].flatten()
        #print(theta1, theta2)
        joint2 = np.array([self.constants.length_bottom*np.cos(theta1)+self.transfom3d_bottomshaft[0], self.constants.length_bottom*np.sin(theta1)+self.transfom3d_bottomshaft[1]])
        end_eff = joint2 + np.array([self.constants.length_top*np.cos(theta1 + theta2), self.constants.length_top*np.sin(theta1 + theta2)])
        #print(joint2, end_eff)
        return (joint2, end_eff)
    
    def get_dynamics_matrices_casadi(self, states):
      
        theta1, theta2, omega1, omega2 = states[0], states[1], states[2], states[3]

        const = self.constants
        l1 = const.length_bottom
        r1 = const.distance_pivot_COM_bottom
        r2 = const.distance_pivot_COM_top
        m1 = const.mass_bottom
        m2 = const.mass_top
        I1 = const.moiCOM_bottom
        I2 = const.moiCOM_top
        g = 9.81
        c2 = ca.cos(theta2)
        hM = l1 * r2 * c2
        hC = -m2 * l1 * r2 * ca.sin(theta2)

       
        M = ca.vertcat(
            ca.horzcat(m1*r1**2 + m2*(l1**2 + r2**2 + 2*hM) + I1 + I2, m2*(r2**2 + hM) + I2),
            ca.horzcat(m2*(r2**2 + hM) + I2, m2*r2**2 + I2)
        )

       
        C = ca.vertcat(
            ca.horzcat(hC*omega2, hC*(omega1 + omega2)),
            ca.horzcat(-hC*omega1, 0)
        )

       
        G = ca.vertcat(
            g*ca.cos(theta1)*(m1*r1 + m2*l1) + g*ca.cos(theta1 + theta2)*m2*r2,
            g*ca.cos(theta1 + theta2)*m2*r2
        )

        return M, C, G


    def get_dynamics_matrices(self, states):
      
        [theta1, theta2, omega1, omega2] = states[:4].flatten()
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 + theta2)

        const = self.constants

        l1 = const.length_bottom
        r1 = const.distance_pivot_COM_bottom
        r2 = const.distance_pivot_COM_top
        m1 = const.mass_bottom
        m2 = const.mass_top
        I1 = const.moiCOM_bottom
        I2 = const.moiCOM_top
        g = 9.81

        hM = l1*r2*c2
        M = m1*np.array([[r1*r1, 0], [0, 0]]) + m2*np.array([[l1*l1 + r2*r2 + 2*hM, r2*r2 + hM], [r2*r2 + hM, r2*r2]]) + \
            I1*np.array([[1, 0], [0, 0]]) + I2*np.array([[1, 1], [1, 1]])

        hC = -m2*l1*r2*np.sin(theta2)
        C = np.array([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        G = g*np.cos(theta1) * np.array([[m1*r1 + m2*l1, 0]]).T + \
            g*np.cos(theta1+theta2) * np.array([[m2*r2, m2*r2]]).T
        
        return (M, C, G)   
    
    def get_dynamics_matrices_CAD(self, states):
      
        [theta1, theta2, omega1, omega2] = states[:4].flatten()
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        c12 = np.cos(theta1 + theta2)

        const = self.constants_optimization

        l1 = const.length_bottom
        r1 = const.distance_pivot_COM_bottom
        r2 = const.distance_pivot_COM_top
        m1 = const.mass_bottom
        m2 = const.mass_top
        I1 = const.moiCOM_bottom
        I2 = const.moiCOM_top
        g = 9.81

        hM = l1*r2*c2
        M = m1*np.array([[r1*r1, 0], [0, 0]]) + m2*np.array([[l1*l1 + r2*r2 + 2*hM, r2*r2 + hM], [r2*r2 + hM, r2*r2]]) + \
            I1*np.array([[1, 0], [0, 0]]) + I2*np.array([[1, 1], [1, 1]])

        hC = -m2*l1*r2*np.sin(theta2)
        C = np.array([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        G = g*np.cos(theta1) * np.array([[m1*r1 + m2*l1, 0]]).T + \
            g*np.cos(theta1+theta2) * np.array([[m2*r2, m2*r2]]).T
        
        return (M, C, G)   
    
    

        

class DCMotorConstants:
    MAX_VOLTAGE = 10.0
    #although current limit is tech 40 A, fuses are temperature-based, and can remain at higher current for a short period of time
    MAX_CURRENT = 80.0
    
    nominalVoltageVolts = 12.0

    stall_torque_bottomArm = 9.37 * 100 * 2 #gear ratio of 100, 2 KrakenX60 motors
    stall_current_bottomArm = 483 * 2
    free_current_bottomArm = 2 * 2
    free_speed_bottomArm = 5800 * 2 * math.pi * (1/60)
    rOhm_bottomArm = nominalVoltageVolts / stall_current_bottomArm
    kV_bottomArm = free_speed_bottomArm / (nominalVoltageVolts - rOhm_bottomArm * free_current_bottomArm)
    kT_bottomArm = stall_torque_bottomArm / stall_current_bottomArm  

    stall_torque_topArm= 9.37 * 100 * 1 #gear ratio of 100, 1 KrakenX60 motor
    stall_current_topArm = 483 * 1
    free_current_topArm = 2 * 1
    free_speed_topArm = 5800 * 2 * math.pi * (1/60)
    rOhm_topArm = nominalVoltageVolts / stall_current_topArm
    kV_topArm = free_speed_topArm / (nominalVoltageVolts - rOhm_topArm * free_current_topArm)
    kT_topArm = stall_torque_topArm / stall_current_topArm 

       
    @staticmethod
    def current_to_voltage_bottom(current, speed):
        return DCMotorConstants.rOhm_bottomArm * current + speed / DCMotorConstants.kV_bottomArm
    @staticmethod
    def current_to_voltage_top(current, speed):
        return DCMotorConstants.rOhm_topArm * current + speed / DCMotorConstants.kV_topArm
    


class ArmConstantsReal: 
    def __init__(self):
        # Arm physical properties
        self.mass_top = 2.3+0.4  # kg
        self.moiCOM_top = 0.88+0.1 # kg*m^2
        self.distance_pivot_COM_top = 0.29  # m
        self.length_top = 0.85 # m

        self.mass_bottom = 4.49+0.4
        self.moiCOM_bottom = 0.70+0.1  # kg*m^2
        self.distance_pivot_COM_bottom = 0.52  # m
        self.length_bottom = 1 # m

      
       

        # Friction coefficients
         
         

#use to test system's resilience against model mismatch between CAD and real world
class ArmConstantsCAD: 
    def __init__(self):
        # Arm physical properties
        self.mass_top = 2.3  # kg
        self.moiCOM_top = 0.88 # kg*m^2
        self.distance_pivot_COM_top = 0.29  # m
        self.length_top = 0.85 # m

        self.mass_bottom = 4.49
        self.moiCOM_bottom = 0.70  # kg*m^2
        self.distance_pivot_COM_bottom = 0.52  # m
        self.length_bottom = 1 # m

       
       

        # Friction coefficients
         
def main():
   
    dt_sim = 0.001
    dt_control = 0.005

    #parameters_xy = [[0.391+0.1 - 0.03, 0.773], [0.391+0.1 - 0.03, 1.85]]
   # parameters_xy = [[0.391+0.1+0.289/2, 0.5], [0.391/2, -0.1]]
    parameters_xy = [[-0.5, 1], [0.391+0.1 - 0.03, 1.85]]


    sim = ArmSim(dt_sim, parameters_xy[0])
    traj = Optimizer(sim)

    theta0 = sim.inv_kinematics(np.array([parameters_xy[0][0], parameters_xy[0][1]]))
    thetaf = sim.inv_kinematics(np.array([parameters_xy[1][0], parameters_xy[1][1]]))

    parameters_theta = [theta0, thetaf]
    print("handed over parameters")
    result = traj.solve(parameters_theta)

    num_steps = math.ceil((result[0]) / dt_sim)
   

    t_since_last_control_update = dt_control
    for i in range(num_steps):
        if t_since_last_control_update >= dt_control:
            t_since_last_control_update -= dt_control
            sample = sampleTrajectory(result, traj, sim.t)
            sim.target = np.array([[sample[0][0]], [sample[0][1]]]).reshape((-1,1))
            current = sim.feed_forward(np.array([[sample[0][0]], [sample[0][1]], [sample[1][0]], [sample[1][1]]]), 
                                       np.array([[sample[2][0]], [sample[2][1]]]))
            sim.u_ff = current
            sim.u_k = np.array([sim.kP_bottom * (sample[0][0] - sim.state[0]), sim.kP_top * (sample[0][1] - sim.state[1])])
       
        t_since_last_control_update += dt_sim
        sim.step()


    #for i in range(math.ceil(5 / dt_sim)):
    #   if t_since_last_control_update >= dt_control:
    #      t_since_last_control_update -= dt_control
    #     sim.target = np.array(thetaf).reshape((-1,1))
    #        current = sim.feed_forward(np.array([[thetaf[0]], [thetaf[1]], [0.0], [0.0]]), np.array([[0.0], [0.0]]))
    #       sim.u_ff = current
    #      sim.u_k = sim.kP2 * (sim.target - sim.state[0:2])
       
    #   t_since_last_control_update += dt_sim
    #  sim.step()

   

    #animate_traj(result, traj, sim)
    #for i in range(n_steps):
    #    sim.step(np.array([[0],[0]]), np.array([[0],[0]]))
    print("Simulation complete, animating")
    animate_arm(sim)
def sampleTrajectory(result, traj: Optimizer, t):
    dt = result[0] / (traj.n+1)
    prevIndex =  math.floor(t / dt)
    nextIndex =  math.ceil(t / dt)
    if nextIndex == prevIndex: 
        nextIndex += 1
    secondPrevIndex = prevIndex - 1
    secondNextIndex = nextIndex + 1    

    if secondPrevIndex < 0:
        secondPrevIndex = 0
    if secondNextIndex > traj.n + 1:  
        secondNextIndex = traj.n + 1
    position_0 = position_0 = np.interp(
    t,
    [prevIndex * dt, nextIndex * dt],
    [result[1][prevIndex], result[1][nextIndex]]
)  
    position_1 = np.interp(
    t,
    [prevIndex * dt, nextIndex * dt],
    [result[2][prevIndex], result[2][nextIndex]]
)

    velocity_0 = (result[1][nextIndex] - result[1][prevIndex]) / dt
    velocity_1 = (result[2][nextIndex] - result[2][prevIndex]) / dt

    acceleration_0 = 0
    acceleration_1 = 0

    if (t % dt) / dt < 0.5:
        prevVelocity_0 = (result[1][prevIndex] - result[1][secondPrevIndex]) / dt
        prevVelocity_1 = (result[2][prevIndex] - result[2][secondPrevIndex]) / dt
        acceleration_0 = (velocity_0 - prevVelocity_0) / dt
        acceleration_1 = (velocity_1 - prevVelocity_1) / dt

    else:
        nextVelocity_0 = (result[1][secondNextIndex] - result[1][nextIndex]) / dt
        nextVelocity_1 = (result[2][secondNextIndex] - result[2][nextIndex]) / dt
        acceleration_0 = (nextVelocity_0 - velocity_0) / dt
        acceleration_1 = (nextVelocity_1 - velocity_1) / dt
    return [[position_0, position_1], [velocity_0, velocity_1], [acceleration_0, acceleration_1]]  
    


def animate_traj(result, traj: Optimizer, arm: ArmSim):
    print(result[0])
    #dt = result[0] / (traj.n+1)
    dt = 1
    
    
    fps = 1/dt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.grid(True)
    total_len = arm.constants.length_bottom + arm.constants.length_top
    ax.set_xlim(-total_len+arm.transfom3d_bottomshaft[0], total_len+arm.transfom3d_bottomshaft[0])
    ax.set_ylim(-total_len+arm.transfom3d_bottomshaft[1], total_len+arm.transfom3d_bottomshaft[1])

    rect1 = patches.Rectangle((0, 0), 0.391, 0.175, linewidth=2, edgecolor='red', facecolor='red')
    #L1 scoring position
    rect2 = patches.Rectangle((0.391+arm.distance_to_reef, 0), 0.289, 0.454, linewidth=2, edgecolor='blue', facecolor='blue')
    #reef pole
    rect3 = patches.Rectangle((0.391+arm.distance_to_reef + 0.289, 0), 0.079, 1.566, linewidth=2, edgecolor='blue', facecolor='blue')
    #L4 horizontal bar
    rect4 = patches.Rectangle((0.391+arm.distance_to_reef + 0.03, 0.454+1.07), 0.259, 0.042, linewidth=2, edgecolor='blue', facecolor='blue')
    #L4 vertical bar
    rect5 = patches.Rectangle((0.391+arm.distance_to_reef + 0.03, 0.454+1.07), 0.042, 0.301, linewidth=2, edgecolor='blue', facecolor='blue')
    #L2 scoring position
    rect6 = patches.Rectangle((0.391+arm.distance_to_reef +0.04, 0.773), 0.303, 0.042, linewidth=2, edgecolor='blue', facecolor='blue', angle = 360 - 35)
    #L3 scoring position
    rect7 = patches.Rectangle((0.391+arm.distance_to_reef +0.04, 0.773+0.403), 0.303, 0.042, linewidth=2, edgecolor='blue', facecolor='blue', angle = 360 - 35)

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.add_patch(rect5)
    ax.add_patch(rect6)
    ax.add_patch(rect7)


    def get_arm_joints(state):
        """Get the xy positions of all three robot joints (?) - base joint (at 0,0), elbow, end effector"""
        (joint_pos, eff_pos) = arm.forward_kinematics(state)
        x = np.array([arm.transfom3d_bottomshaft[0], joint_pos[0], eff_pos[0]])
        y = np.array([arm.transfom3d_bottomshaft[1], joint_pos[1], eff_pos[1]])
        return (x,y)

    theta_init_bottom, theta_init_top = result[1][0], result[2][0]
    (x_init, y_init) = get_arm_joints(np.array([theta_init_bottom, theta_init_top, 0, 0])) 
    current_line, = ax.plot(x_init, y_init, 'g-o')


    def init(): 
        current_line.set_data(x_init, y_init)
        return current_line


    def animate(i):
        theta_bottom = result[1][i]
        theta_top = result[2][i]
        (x_vector, y_vector) = get_arm_joints(np.array([theta_bottom, theta_top, 0, 0]))
        current_line.set_data(x_vector, y_vector)
        return current_line
    nframes = traj.n + 2
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=True)
    plt.show()


        








def animate_arm(arm: ArmSim, fps = 20):

    def plot_data(t, hist, indices, ax: plt.Axes = None, lines = None):
        if ax is None and lines is None:
            raise Exception("Either ax or lines must be given.")
        data = hist[indices, :]
        if lines is None:
            for idx in indices:
                ax.plot(t, hist[idx,:])
            lines = ax.get_lines()
        else:
            for i, idx in enumerate(indices):
                lines[i].set_data(t, hist[idx,:])
        return lines
    

    def get_arm_joints(state):
        """Get the xy positions of all three robot joints (?) - base joint (at 0,0), elbow, end effector"""
        (joint_pos, eff_pos) = arm.forward_kinematics(state)
        x = np.array([arm.transfom3d_bottomshaft[0], joint_pos[0], eff_pos[0]])
        y = np.array([arm.transfom3d_bottomshaft[1], joint_pos[1], eff_pos[1]])
        return (x,y)
    
    dt = 1.0/fps
   
    tvec = np.arange(0, arm.t + dt, dt)

    fig = plt.figure()
    ax = fig.add_subplot(4,4,(1,15))
    ax.axis('square')
    ax.grid(True)
    #gearbox
    rect1 = patches.Rectangle((0, 0), 0.391, 0.175, linewidth=2, edgecolor='red', facecolor='red')
    #L1 scoring position
    rect2 = patches.Rectangle((0.391+arm.distance_to_reef, 0), 0.289, 0.454, linewidth=2, edgecolor='blue', facecolor='blue')
    #reef pole
    rect3 = patches.Rectangle((0.391+arm.distance_to_reef + 0.289, 0), 0.079, 1.566, linewidth=2, edgecolor='blue', facecolor='blue')
    #L4 horizontal bar
    rect4 = patches.Rectangle((0.391+arm.distance_to_reef + 0.03, 0.454+1.07), 0.259, 0.042, linewidth=2, edgecolor='blue', facecolor='blue')
    #L4 vertical bar
    rect5 = patches.Rectangle((0.391+arm.distance_to_reef + 0.03, 0.454+1.07), 0.042, 0.301, linewidth=2, edgecolor='blue', facecolor='blue')
    #L2 scoring position
    rect6 = patches.Rectangle((0.391+arm.distance_to_reef +0.04, 0.773), 0.303, 0.042, linewidth=2, edgecolor='blue', facecolor='blue', angle = 360 - 35)
    #L3 scoring position
    rect7 = patches.Rectangle((0.391+arm.distance_to_reef +0.04, 0.773+0.403), 0.303, 0.042, linewidth=2, edgecolor='blue', facecolor='blue', angle = 360 - 35)

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.add_patch(rect5)
    ax.add_patch(rect6)
    ax.add_patch(rect7)

    total_len = arm.constants.length_bottom + arm.constants.length_top
    ax.set_xlim(-total_len+0.391, total_len+0.391)
    ax.set_ylim(0, 1.9)
    initial_state = arm.log.sample(0)
    (x_init, y_init) = get_arm_joints(initial_state)
    current_line, = ax.plot(x_init, y_init, 'g-o')

    ax1 = fig.add_subplot(4,4,16)
    ax1.set_aspect('auto')
    ax1.grid(True)
    ax1.set_xlim(0, fps)
    ax1_line, = plot_data(0, initial_state, [5], ax = ax1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Error Top Arm (degrees)", fontsize=8)
    ax1.yaxis.set_label_position("right")
    ax1.set_xlim(0, arm.log.end_time )
    ax1.set_ylim(-5, 5)

    ax2 = fig.add_subplot(4,4,8)
    ax2.set_aspect('auto')
    ax2.grid(True)
    ax2.set_xlim(0, fps)
    ax2_line, = plot_data(0, initial_state, [4], ax = ax2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error Bottom Arm (degrees)", fontsize=8)
    ax2.yaxis.set_label_position("right")
    ax2.set_xlim(0, arm.log.end_time)
    ax2.set_ylim(-5, 5)


    error_history_ax1 = []
    time_history_ax1 = []

    error_history_ax2 = []
    time_history_ax2 = []
    #time_plotting = deque(maxlen= math.ceil(1/(2*dt)))
    
    def init():

        current_line.set_data(x_init, y_init)

        error_history_ax1.clear()
        time_history_ax1.clear()

        error_history_ax2.clear()
        time_history_ax2.clear()

        return current_line
    def animate(i):
        t = tvec[i]
        state = arm.log.sample(t)
        error_history_ax1.append(state[5].item() * 180/math.pi)
        time_history_ax1.append(t)
        
        error_history_ax2.append(state[4].item() * 180/math.pi)
        time_history_ax2.append(t)
      
        ax1_line.set_data(time_history_ax1, error_history_ax1)
        ax2_line.set_data(time_history_ax2, error_history_ax2)

        (x_vector, y_vector) = get_arm_joints(state)
        current_line.set_data(x_vector, y_vector)
        

        return current_line
    nframes = len(tvec)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()  
     

    
