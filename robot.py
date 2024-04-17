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
        self.touch_matrix = np.ndarray((c.iterations, c.numSensorNeurons), dtype=int)

    def Prepare_To_Sense(self, num_motors: int = 0, num_sensors: int = 0):
        self.sensors = {}

        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = SENSOR(linkName)

    def Prepare_To_Act(self):
        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            if jointName == b'Body_BackLeg':
                self.motors[jointName] = MOTOR(jointName)
            else:
                self.motors[jointName] = MOTOR(jointName)

    def Sense(self, iteration: int):
        # looping through the different sensor names, because sensors dictionary cannot be references with int index values.
        idx_sensor = 0
        for sensor_name in self.sensors.keys():
            self.touch_matrix[iteration, idx_sensor] = self.sensors[sensor_name].Get_Value(iteration)
            idx_sensor += 1

    def Think(self):
        self.nn.Update()

    def Act(self, t: int):
        for neuronName in self.nn.Get_Neuron_Names():
            if self.nn.Is_Motor_Neuron(neuronName):
                # Current location of the body unit at [0][0,1,2]. The Z characteristic is for current elevation of the
                # body, but this doesn't account for legs touching the ground.
                body_pos = pybullet.getLinkState(self.id, 0)
                if body_pos[0][2] > self.max_height:
                    self.max_height = body_pos[0][2]
                position_to_air = 0 # 0 indicates that the robot is touching the ground, 1 indicates that the robot is almost touching the ground, 2 indicates that the robot is in the air.

                jointName = self.nn.Get_Motor_Neurons_Joint(neuronName).encode('utf-8')
                joint_name_not_encoded = self.nn.Get_Motor_Neurons_Joint(neuronName)
                time_step = (t % 250) # Jumping goes through a cycle every 500 time steps, starting at t = 0.

                # Stage 1: Feet stay still, legs slightly bend to brace for next jump
                if time_step < 200:
                    if 'Foot' in joint_name_not_encoded:
                        desiredAngle = 1 - self.nn.Get_Value_Of(neuronName) * 0.25
                    elif 'Calf' in joint_name_not_encoded and ('Back' in joint_name_not_encoded or 'Left' in joint_name_not_encoded):
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) + np.sin((time_step * np.pi) / 400) * 0.5
                    elif 'Body' in joint_name_not_encoded and ('Front' in joint_name_not_encoded or 'Right' in joint_name_not_encoded):
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) + np.sin((time_step * np.pi) / 400) * 0.5
                    else:
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) - np.sin((time_step * np.pi) / 400) * 0.5

                # Stage 2: Legs fully extend to create a large lift force for the body, feet impact the ground to give extra lift
                elif time_step < 230:
                    if 'Foot' in joint_name_not_encoded:
                        desiredAngle = -((self.nn.Get_Value_Of(neuronName) * 0.5) + 1)
                    elif 'Body' in joint_name_not_encoded and ('Back' in joint_name_not_encoded or 'Left' in joint_name_not_encoded):
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) - np.cos(((time_step - 200) * np.pi) / 25) * 2.5
                    elif 'Calf' in joint_name_not_encoded and ('Front' in joint_name_not_encoded or 'Right' in joint_name_not_encoded):
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) - np.cos(((time_step - 200) * np.pi) / 25) * 2.5
                    else:
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) + np.cos(((time_step - 200) * np.pi) / 25) * 2.5

                # Stage 4: Body moves back into static position to securely land robot on the ground.
                else:
                    if 'Foot' in joint_name_not_encoded:
                        desiredAngle = 1 - self.nn.Get_Value_Of(neuronName) * 0.25
                    elif 'Calf' in joint_name_not_encoded and ('Back' in joint_name_not_encoded or 'Left' in joint_name_not_encoded):
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) + np.sin((time_step * np.pi) / 100) * 0.5
                    elif 'Body' in joint_name_not_encoded and ('Front' in joint_name_not_encoded or 'Right' in joint_name_not_encoded):
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) + np.sin((time_step * np.pi) / 100) * 0.5
                    else:
                        desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.7) - np.sin((time_step * np.pi) / 100) * 0.5

                # Apply forces to given robot's jump
                if time_step > 200 or time_step < 300:
                    self.motors[jointName].Act(desiredAngle, np.pi / 4, 300)
                else:
                    self.motors[jointName].Act(desiredAngle, np.pi / 4, 40)


    # We can modify this fitness function to record whether the link is touching the ground or not, or record from
    # another file the total number of iterations that all the links weren't touching the ground.
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

            # Check each sensor, to see if any sensor is not equal to -1.
            for idx_sensor in range(len(self.touch_matrix[iteration])):
                if self.touch_matrix[iteration][idx_sensor] != -1:
                    in_air = False

            # If in_air is still true, then increment the total_airtime variable.
            if in_air:
                total_airtime += 1
        for x in range(len(self.touch_matrix)):
            if 1 in self.touch_matrix[x]:
                print(self.touch_matrix[x])
        f = open(f"tmp{p_id}.txt", "w")
        f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str((xCoord * 6.0) + (total_airtime * 0.04))) # Rewarding furthest distance away from the user, and lightly awarding longest airtime.
        os.rename(f"tmp{p_id}.txt", f"fitness{p_id}.txt")
        f.close()

        return
