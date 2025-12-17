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



class ArmSim:
    def __init__(self, dt, start_state = np.zeros((4,1))):
        self.dt = dt
        self.state = start_state

        # state_labels = [("Angle Bottom", "rad"), ("Angle Top", "rad"), ("Angular velocity Bottom", "rad/s"), ("Angular velocity Top", "rad/s")]
        # u_labels = [("Current Bottom", "A"), ("Current Top", "A")]
        # self.set_plot_labels(state_labels, u_labels)
        self.constants = ArmConstantsReal()

        self.t = 0.0
        self.log = frc.Trajectory(np.array([self.t]), start_state.reshape(-1,1))
        self.loop_time = 0.020

        #represents current added from feedback
        self.u_k = np.zeros((2,1))
        self.u_ff = np.zeros((2,1))

    def step(self, u_ff, u_k):
        self.state = frc.rkdp(self.dynamics_true, self.state, self.u_ff + self.u_k, self.dt)
        self.t+= self.dt
        self.log.insert(self.t, self.state)
    
    def dynamics_true(self, state, u):
        torque = u*self.constants.MotorKt  -1 * np.tanh(20*state[2:4]) - 0.01*state[2:4]
        #my solution
        #torque = u*self.constants.MotorKt + np.array(Math.sign(state[2] * (-0.5)), Math.sign(state[3] * (-0.5)))
        (M, C, G) = self.get_dynamics_matrices(state)
        omega_vec = state[2:4]
        accel_vector = np.linalg.inv(M) @ (torque - C @ omega_vec - G)
        state_derivative_vector = np.concatenate((omega_vec, accel_vector))
        return state_derivative_vector

    def forward_kinematics(self, state):
        [theta1, theta2] = state[:2].flatten()
        print(theta1, theta2)
        joint2 = np.array([self.constants.length_bottom*np.cos(theta1), self.constants.length_bottom*np.sin(theta1)])
        end_eff = joint2 + np.array([self.constants.length_top*np.cos(theta1 + theta2), self.constants.length_top*np.sin(theta1 + theta2)])
        print(joint2, end_eff)
        return (joint2, end_eff)

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

   
    for i in range(n_steps):
        sim.step(np.array([[0],[0]]), np.array([[0],[0]]))
    animate_arm(sim)

def animate_arm(arm: ArmSim, fps = 20):
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
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.grid(True)
    total_len = arm.constants.length_bottom + arm.constants.length_top
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)
    initial_state = arm.log.sample(0)
    (x_init, y_init) = get_arm_joints(initial_state)
    current_line, = ax.plot(x_init, y_init, 'g-o')
    print(x_init)
    print(y_init)
   

    def init():

        current_line.set_data(x_init, y_init)
        return current_line
    def animate(i):
        t = tvec[i]
        state = arm.log.sample(t)
        (x_vector, y_vector) = get_arm_joints(state)
        current_line.set_data(x_vector, y_vector)
        
        return current_line
    nframes = len(tvec)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()  
     

    
