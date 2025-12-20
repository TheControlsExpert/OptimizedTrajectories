import sys

import matplotlib as mpl
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

    def __init__(self, dt, start_state = np.zeros((4,1))):
        self.dt = dt
        self.state = start_state
        self.transfom3d_bottomshaft = [0.249, 0.125]

        # state_labels = [("Angle Bottom", "rad"), ("Angle Top", "rad"), ("Angular velocity Bottom", "rad/s"), ("Angular velocity Top", "rad/s")]
        # u_labels = [("Current Bottom", "A"), ("Current Top", "A")]
        # self.set_plot_labels(state_labels, u_labels)
        self.constants = ArmConstantsReal()
        self.motor_constants = DCMotorConstants()

        self.t = 0.0
        M,C,G = self.get_dynamics_matrices(start_state)

        self.log = frc.Trajectory(np.array([self.t]), np.concatenate((start_state, G)).reshape(-1,1))
                                                                     
        #represents current added from feedback
        self.u_k = np.zeros((2,1))
        self.u_ff = np.zeros((2,1))

    def step(self, u_ff, u_k):
        self.state = frc.rkdp(self.dynamics_true, self.state, self.u_ff + self.u_k, self.dt)
        self.t+= self.dt
    
    def feed_forward(self, states, accels = np.zeros((2,1))):
        (M, C, G) = self.get_dynamics_matrices(states)
        omegas = states[2:4]
        torque = M @ accels + C @ omegas + G
        return torque / self.constants.MotorKt
        
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
            (1/self.constants.MotorKt)*(M_times_acceleration[0] + C_times_velocity[0] + G[0]),
            (1/self.constants.MotorKt)*(M_times_acceleration[1] + C_times_velocity[1] + G[1]))
        
       
        return current

    
   
    
    def dynamics_true(self, state, u):
        #tanh avoids the discontinuity of signum function
        #tanh is used to add frictional torque opposing motion

        torque = u*self.constants.MotorKt  -3.5 * (np.tanh(state[2:4]))
       
        #my solution
        #torque = u*self.constants.MotorKt + np.array(Math.sign(state[2] * (-0.5)), Math.sign(state[3] * (-0.5)))
        (M, C, G) = self.get_dynamics_matrices(state)
        omega_vec = state[2:4]
        accel_vector = np.linalg.inv(M) @ (torque - C @ omega_vec - G)
        state_derivative_vector = np.concatenate((omega_vec, accel_vector))
        self.log.insert(self.t, np.concatenate((self.state, torque)).reshape(-1,1))
        
        
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
    kT_bottomArm = stall_torque_bottomArm / stall_current_bottomArm * 100 

    stall_torque_topArm= 9.37 * 100 * 1 #gear ratio of 100, 1 KrakenX60 motor
    stall_current_topArm = 483 * 1
    free_current_topArm = 2 * 1
    free_speed_topArm = 5800 * 2 * math.pi * (1/60)
    rOhm_topArm = nominalVoltageVolts / stall_current_topArm
    kV_topArm = free_speed_topArm / (nominalVoltageVolts - rOhm_topArm * free_current_topArm)
    kT_topArm = stall_torque_topArm / stall_current_topArm * 100 

       
    @staticmethod
    def current_to_voltage_bottom(current, speed):
        return DCMotorConstants.rOhm_bottomArm * current + speed / DCMotorConstants.kV_bottomArm
    @staticmethod
    def current_to_voltage_top(current, speed):
        return DCMotorConstants.rOhm_topArm * current + speed / DCMotorConstants.kV_topArm
    


class ArmConstantsCAD: 
    def __init__(self):
        # Arm physical properties
        self.mass_top = 2.24  # kg
        self.moiCOM_top = 0.0862 # kg*m^2
        self.distance_pivot_COM_top = 0.265  # m
        self.length_top = 0.900 # m

        self.mass_bottom = 4.40
        self.moiCOM_bottom = 0.413  # kg*m^2
        self.distance_pivot_COM_bottom = 0.367  # m
        self.length_bottom = 0.750 # m

        self.MotorKt = 0.4
       

        # Friction coefficients
         

#use to test system's resilience against model mismatch between CAD and real world
class ArmConstantsReal:
    def __init__(self):
        # Arm physical properties
        self.mass_top = 2.24  # kg
        self.moiCOM_top = 0.0862 # kg*m^2
        self.distance_pivot_COM_top = 0.265  # m
        self.length_top = 0.900 # m

        self.mass_bottom = 4.40
        self.moiCOM_bottom = 0.413  # kg*m^2
        self.distance_pivot_COM_bottom = 0.367  # m
        self.length_bottom = 0.750 # m

        self.MotorKt = 0.4
def main():
    
   

    dt = 0.005
    state_initial = np.array([math.pi/4,-3/4*math.pi,0,0]).reshape((4,1))
    sim = ArmSim(dt, start_state = state_initial)
    n_steps = math.ceil(20/dt)

    traj = Optimizer(sim)
    parameters_xy = [[0, 1], [0, 1.5]]
    theta0 = sim.inv_kinematics(np.array([parameters_xy[0][0], parameters_xy[0][1]]))
    thetaf = sim.inv_kinematics(np.array([parameters_xy[1][0], parameters_xy[1][1]]))
    parameters_theta = [theta0, thetaf]
    print("handed over parameters")
    result = traj.solve(parameters_theta)
    animate_traj(result, traj, sim)
    #for i in range(n_steps):
    #    sim.step(np.array([[0],[0]]), np.array([[0],[0]]))
    #animate_arm(sim)
    
def animate_traj(result, traj: Optimizer, arm: ArmSim):
    dt = result[0] / (traj.n+1)
    #dt = 0.25
    fps = 1/dt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.grid(True)
    total_len = arm.constants.length_bottom + arm.constants.length_top
    ax.set_xlim(-total_len+arm.transfom3d_bottomshaft[0], total_len+arm.transfom3d_bottomshaft[0])
    ax.set_ylim(-total_len+arm.transfom3d_bottomshaft[1], total_len+arm.transfom3d_bottomshaft[1])


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


        








def animate_arm(arm: ArmSim, fps = 40):

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
        x = np.array([0, joint_pos[0], eff_pos[0]])
        y = np.array([0, joint_pos[1], eff_pos[1]])
        return (x,y)
    
    dt = 1.0/fps
    #replace 15 with total trajectory time 
    tvec = np.arange(0, 20 + dt, dt)

    fig = plt.figure()
    ax = fig.add_subplot(4,4,(1,15))
    ax.axis('square')
    ax.grid(True)
    total_len = arm.constants.length_bottom + arm.constants.length_top
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)
    initial_state = arm.log.sample(0)
    (x_init, y_init) = get_arm_joints(initial_state)
    current_line, = ax.plot(x_init, y_init, 'g-o')

    ax1 = fig.add_subplot(4,4,16)
    ax1.set_aspect('auto')
    ax1.grid(True)
    ax1.set_xlim(0, fps)
    ax1_line, = plot_data(0, initial_state, [4], ax = ax1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("G1 (Nm)")
    ax1.yaxis.set_label_position("right")

    queue_plotting = deque(maxlen= math.ceil((1/(2*dt))))
    time_plotting = deque(maxlen= math.ceil(1/(2*dt)))
    
    def init():

        current_line.set_data(x_init, y_init)
        return current_line
    def animate(i):
        t = tvec[i]
        state = arm.log.sample(t)
        queue_plotting.append(state[4])
        time_plotting.append(i)
        ax1.set_xlim(time_plotting[0], time_plotting[0] + fps)
        ax1.set_ylim(-1, 1)
        ax1_line.set_data(time_plotting, queue_plotting)

        (x_vector, y_vector) = get_arm_joints(state)
        current_line.set_data(x_vector, y_vector)
        

        return current_line
    nframes = len(tvec)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()  
     

    
