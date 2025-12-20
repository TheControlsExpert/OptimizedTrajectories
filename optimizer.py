import math
import casadi as ca
import numpy as np


class Optimizer:
    def __init__(self, sim):
        self.sim = sim


        #self.max_accelBottom = 10
        #self.max_accelTop = 100

        #self.max_jerkBottom = 30
        #self.max_jerkTop = 30


        self.opti = ca.Opti()
        self.opti.solver("ipopt")
        self.n = 25

        self._total_time = self.opti.variable()
        dt = self._total_time / (self.n + 1)
        self.opti.subject_to(self._total_time > 0)
        self.opti.subject_to(self._total_time < 10)
        self.opti.minimize(self._total_time)

        theta_points = []
        self.theta_points = theta_points
        theta_points.append([self.opti.parameter(), self.opti.parameter()])
        for _ in range(self.n):
            theta_0 = self.opti.variable() #bottom arm angle
            theta_1 = self.opti.variable() #top arm angle
            self.opti.subject_to(
                self.opti.bounded(
                    32 * ca.pi / 180,
                    theta_0,
                    (180-19) * ca.pi / 180,
                )
            )

            joint, end_effector = self.sim.forward_kinematics_casadi(ca.vertcat(theta_0, theta_1))
            #keep the arm out of the gearbox area
            normX = 2 * (end_effector[0] - (0.391)/2) / (0.391)
            normY = 2 * (end_effector[1] - (0.175/2)) / (0.175)

            self.opti.subject_to(      
                    ca.fmax(ca.fabs(normX), ca.fabs(normY)) > 1          
            )

            #keep the end effector above the ground
            self.opti.subject_to(
                    end_effector[1] >= 0.05
            )
            

            theta_points.append([theta_0, theta_1])
        theta_points.append([self.opti.parameter(), self.opti.parameter()])
 
        # Apply point constraints
        for i in range(self.n + 2):
            # Get surrounding points
            last_theta_2 = theta_points[i if i <= 1 else i - 2]
            last_theta = theta_points[i if i == 0 else i - 1]
            current_theta = theta_points[i]
            next_theta = theta_points[i if i == self.n + 1 else i + 1]

            # Apply constraints to joints
            last_velocity_2 = (
                (last_theta[0] - last_theta_2[0]) / dt,
                (last_theta[1] - last_theta_2[1]) / dt,
            )
            last_velocity = [
                (current_theta[0] - last_theta[0]) / dt,
                (current_theta[1] - last_theta[1]) / dt,
            ]
            next_velocity = (
                (next_theta[0] - current_theta[0]) / dt,
                (next_theta[1] - current_theta[1]) / dt,
            )
            last_acceleration = (
                (last_velocity[0] - last_velocity_2[0]) / dt,
                (last_velocity[1] - last_velocity_2[1]) / dt,
            )
            acceleration = ca.vertcat(
                (next_velocity[0] - last_velocity[0]) / dt,
                (next_velocity[1] - last_velocity[1]) / dt,
            )
            jerk = (
                (acceleration[0] - last_acceleration[0]) / (2 * dt),
                (acceleration[1] - last_acceleration[1]) / (2 * dt),
            )
            states = ca.vertcat(current_theta[0], current_theta[1], last_velocity[0], last_velocity[1])
            current = self.sim.feed_forward_casadi(states, acceleration)
            voltage_bottom, voltage_top = self.sim.motor_constants.current_to_voltage_bottom(current[0], last_velocity[0]), self.sim.motor_constants.current_to_voltage_top(current[1], last_velocity[1])



            self.opti.subject_to(
                self.opti.bounded(-self.sim.motor_constants.MAX_VOLTAGE, voltage_bottom, self.sim.motor_constants.MAX_VOLTAGE)
            )
            self.opti.subject_to(
                self.opti.bounded(-self.sim.motor_constants.MAX_VOLTAGE, voltage_top, self.sim.motor_constants.MAX_VOLTAGE)
            )

            self.opti.subject_to(
                self.opti.bounded(
                    -self.sim.motor_constants.MAX_CURRENT, current[0], self.sim.motor_constants.MAX_CURRENT
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    -self.sim.motor_constants.MAX_CURRENT, current[1], self.sim.motor_constants.MAX_CURRENT
                )
            )

    def solve(self, parameters):
        #parameters is a 2d array containing the start and end angles
        self.opti.set_value(
            self.theta_points[0][0],
            parameters[0][0])


        self.opti.set_value(
            self.theta_points[0][1],
            parameters[0][1],
        )
        self.opti.set_value(
            self.theta_points[len(self.theta_points) - 1][0],
            parameters[1][0],
        )
        
        self.opti.set_value(
            self.theta_points[len(self.theta_points) - 1][1],
            parameters[1][1],
        )

        self.opti.set_initial(self._total_time, 1)

        

        for i in range(1, self.n + 1):
            self.opti.set_initial(
                self.theta_points[i][0],
                (parameters[1][0] - parameters[0][0]) * (i / (self.n + 2))
                + parameters[0][0],
            )
            
            self.opti.set_initial(
                    self.theta_points[i][1],
                    (parameters[1][1] - parameters[0][1]) * (i / (self.n + 2))
                    + parameters[0][1],
            )
            
        try:
            print("Solving...")
            sol = self.opti.solve()
        except RuntimeError as e:
            print("Solver failed!")
            print("Solver message:", e)
            # Don't call show_infeasibilities() here
            return None
        # Get results
        result = (sol.value(self._total_time), [], [])
        for theta in self.theta_points:
            result[1].append(sol.value(theta[0]))
            result[2].append(sol.value(theta[1]))
        return result    









        
        







        

  