# Robot.py Class
# Class containing basic functionality for sensing, moving, and improving with each iteration

import numpy as np
import os
import pybullet
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import constants as c
from motor import MOTOR
from sensor import SENSOR
from pyrosim.neuralNetwork import NEURAL_NETWORK

class ROBOT:

    def __init__(self, id: int, m: int = 0, s: int = 0, p_id: int = 0):
        self.Prepare_To_Sense(m, s)
        self.Prepare_To_Act()
        self.id = id
        self.nn = NEURAL_NETWORK(f"brain{p_id}.nndf")
        self.in_air = True
        self.jump_counter = 0
        self.touch_matrix = np.ndarray((c.iterations, c.numSensorNeurons), dtype=int)

    def Prepare_To_Sense(self, num_motors: int = 0, num_sensors: int = 0):
        self.sensors = {}

        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = SENSOR(linkName)

    def Prepare_To_Act(self):
        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = MOTOR(jointName)

    def Sense(self, iteration: int):
        # Loop through the different sensors to determine which are touching the ground
        idx_sensor = 0
        curr_air_status = True
        for sensor_name in self.sensors.keys():
            if self.sensors[sensor_name].Get_Value(iteration) != -1:
                curr_air_status = False
            self.touch_matrix[iteration, idx_sensor] = self.sensors[sensor_name].Get_Value(iteration)
            idx_sensor += 1

        if iteration > 200:
            # Check whether the robot is currently fully in the air or not
            if curr_air_status != self.in_air:
                # If they land on the ground after being fully in the air (not the other way around), then increment the
                # jump_counter variable of the robot. This is designed to allow the robot to freeze after they hit the
                # ground in the end of their third jump.
                if self.in_air:
                    pass
                else:
                    self.jump_counter += 1

                # Finally, change the value of the current air status for the robot since it changed.
                self.in_air = curr_air_status

    def Think(self):
        self.nn.Update()

    # Function controlling the action forces of the motors in the robot
    def Act(self, t: int):
        for neuronName in self.nn.Get_Neuron_Names():
            if self.nn.Is_Motor_Neuron(neuronName):
                # Current location of the body unit at [0][0,1,2]. The Z characteristic is for current elevation of the
                # body, but this doesn't account for legs touching the ground.
                body_pos = pybullet.getLinkState(self.id, 0)

                # Name of the given joint to apply force to
                jointName = self.nn.Get_Motor_Neurons_Joint(neuronName).encode('utf-8')
                joint_name_not_encoded = self.nn.Get_Motor_Neurons_Joint(neuronName)

                # Cycle of energy surges (pushing off to jump) occurs during the 300th timestep.
                time_step = (t % 300) # Jumping goes through a cycle every 333 time steps, starting at t = 0.

                # Legs neurons are given input signal to move
                if 'Foot' in joint_name_not_encoded:
                    desiredPosition = self.nn.Get_Value_Of(neuronName)
                    # Restrict movements of prismatic joints (Forwards and backwards, act as pistons for feet) to remain connected to each leg
                    if desiredPosition > 0.6:
                        desiredPosition = 0.6
                    elif desiredPosition < -0.6:
                        desiredPosition = -0.6
                elif 'Body' in joint_name_not_encoded:
                    desiredPosition = self.nn.Get_Value_Of(neuronName)
                elif 'Calf' in joint_name_not_encoded:
                    desiredPosition = self.nn.Get_Value_Of(neuronName)
                # If no position is given, assume this standard position.
                else:
                    desiredPosition = 0.5

                # Apply forces to given robot's jump
                if time_step > 200 and time_step < 300:
                    self.motors[jointName].Act(desiredPosition, np.pi / 4, 100)
                else:
                    self.motors[jointName].Act(desiredPosition, np.pi / 4, 40)


    # GetFitness Function records all of the fitness values of the current robot, determining the final (X,Y,Z) position
    # of the robot, along with the overall fitness score for whether the robot was in the air for a long time and whether they moves in the positive X-direction.
    # Params:

    def GetFitness(self, p_id):
        basePositionAndOrientation = pybullet.getBasePositionAndOrientation(self.id)

        basePosition = basePositionAndOrientation[0]
        xCoord = basePosition[0]
        yCoord = basePosition[1]
        zCoord = basePosition[2]
        total_airtime = 0

        for iteration in range(len(self.touch_matrix)):
            # Assume the robot is in the air until proven otherwise
            in_air = True
            if self.touch_matrix[iteration].sum() < -13:
                in_air = False

            # If in_air is still true, then increment the total_airtime variable.
            if in_air:
                total_airtime += 1

        # Reduction for initial time in the sky (do not let total airtime go negative
        total_airtime = total_airtime - 100
        if total_airtime < 0:
            total_airtime = 0

        # Problems:
        # Current robot tries to jump in the wrong direction. We can solve this by penalizing moving in the y-direction.
        # Robot doesn't jump three times. We should penalize the robot for each jump that it makes above the three jumps mark.

        # Record all fitness scores as given above, then return
        f = open(f"tmp{p_id}.txt", "w")
        # Rewarding furthest distance away from the user, and lightly awarding longest airtime.
        # However, we care about the robot taking the quickest route, which may mean penalizing unnecessary movements.
        # Team A will be focused on keeping the robot to be within three jump, while Team B will be focused on keeping
        # the robot within the most direct Positive X direction. This means minimizing the deviation from Y=0 at the end
        # of the iteration. We will try to penalize these evenly, with each extra jump reducing the total fitness by
        # 0.5, and each unit in either y-direction reducing the fitness by 1.

        # Current function is a failure
        # if p_id % 10 < 5:
        #     if self.jump_counter > 0 and self.jump_counter < 6:
        #         f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str((xCoord * 1.5) + (total_airtime * 0.01) + ((3 - abs(3 - self.jump_counter)) * 0.5)) + ", " + str(total_airtime))
        #     else:
        #         f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str((xCoord * 1.5) + (total_airtime * 0.01) + ((6 - self.jump_counter) * 0.5)) + ", " + str(total_airtime))
        # else:
        #     f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str((xCoord * 1.5) + (total_airtime * 0.01) - (abs(yCoord))) + ", " + str(total_airtime))
        if self.jump_counter == 3 or self.jump_counter == 4:
            f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str((xCoord * 3.0) + (total_airtime * 0.02) - (abs(yCoord) * 2) + 20) + ", " + str(total_airtime))
        else:
            f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str((xCoord * 3.0) + (total_airtime * 0.02) - (abs(yCoord) * 2) - (self.jump_counter * 0.3)) + ", " + str(total_airtime))
        os.rename(f"tmp{p_id}.txt", f"fitness{p_id}.txt")
        f.close()

        return
