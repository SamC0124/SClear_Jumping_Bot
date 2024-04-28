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
        self.max_height = 0
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
                if body_pos[0][2] > self.max_height:
                    self.max_height = body_pos[0][2] + 0.5

                # Name of the given joint to apply force to
                jointName = self.nn.Get_Motor_Neurons_Joint(neuronName).encode('utf-8')
                joint_name_not_encoded = self.nn.Get_Motor_Neurons_Joint(neuronName)

                # Current Timestep of the jumping cycle
                time_step = (t % 333) # Jumping goes through a cycle every 333 time steps, starting at t = 0.

                # Problems:
                # Current robot tries to jump in the wrong direction each time it jumps, evolution increases the

                # Stage 1: Feet stay still, legs slightly bend to brace for next jump
                if time_step < 200:
                    if 'Foot' in joint_name_not_encoded:
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                        if desiredPosition > 0.7:
                            desiredPosition = 0.7
                        elif desiredPosition < -0.7:
                            desiredPosition = -0.7
                    elif 'Calf' in joint_name_not_encoded and ('Back' in joint_name_not_encoded or 'Left' in joint_name_not_encoded):
                        desiredPosition = (self.nn.Get_Value_Of(neuronName))
                    elif 'Body' in joint_name_not_encoded and ('Front' in joint_name_not_encoded or 'Right' in joint_name_not_encoded):
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                    else:
                        desiredPosition = np.pi / 4

                # Stage 2: Legs fully extend to create a large lift force for the body, feet impact the ground to give extra lift
                elif time_step < 300:
                    if 'Foot' in joint_name_not_encoded:
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                        if desiredPosition > 0.7:
                            desiredPosition = 0.7
                        elif desiredPosition < -0.7:
                            desiredPosition = -0.7
                    elif 'Body' in joint_name_not_encoded and ('Back' in joint_name_not_encoded or 'Left' in joint_name_not_encoded):
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                    elif 'Calf' in joint_name_not_encoded and ('Front' in joint_name_not_encoded or 'Right' in joint_name_not_encoded):
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                    else:
                        desiredPosition = np.pi / 4

                # Stage 3: Body moves back into static position to securely land robot on the ground.
                else:
                    if 'Foot' in joint_name_not_encoded:
                        desiredPosition = -self.nn.Get_Value_Of(neuronName)
                        if desiredPosition > 0.7:
                            desiredPosition = 0.7
                        elif desiredPosition < -0.7:
                            desiredPosition = -0.7
                    elif 'Calf' in joint_name_not_encoded and ('Back' in joint_name_not_encoded or 'Left' in joint_name_not_encoded):
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                    elif 'Body' in joint_name_not_encoded and ('Front' in joint_name_not_encoded or 'Right' in joint_name_not_encoded):
                        desiredPosition = self.nn.Get_Value_Of(neuronName)
                    else:
                        desiredPosition = np.pi / 4

                # Only apply forces to the motors if the robot hasn't landed their third jump (ignoring the initial
                # time of robot landing on the ground).
                # Apply forces to given robot's jump
                if time_step > 200 or time_step < 300:
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
            if self.touch_matrix[iteration].sum() == -13:
                in_air = False

            # If in_air is still true, then increment the total_airtime variable.
            if in_air:
                total_airtime += 1

        # Reduction for initial time in the sky
        total_airtime = total_airtime - 100

        # Record all fitness scores as given above, then return
        f = open(f"tmp{p_id}.txt", "w")
        # Rewarding furthest distance away from the user, and lightly awarding longest airtime.
        # Update: Increasing the reward from airtime, because robot tends to stay only on the ground.
        f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str(xCoord + (total_airtime * 0.05)))
        os.rename(f"tmp{p_id}.txt", f"fitness{p_id}.txt")
        f.close()

        return
