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

    def Sense(self, index: int):
        pass

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
                # Center of the body is at 0,0,1
                if body_pos[0][2] > 1.1:
                    body_pos = 2
                    print("In the Air!")
                elif body_pos[0][2] < 1.1:
                    body_pos = 0
                    print("Approaching or On the Ground!")

                jointName = self.nn.Get_Motor_Neurons_Joint(neuronName).encode('utf-8')
                time_step = (t % 500) # Jumping goes through a cycle every 500 time steps, starting at t = 0.
                if time_step < 100:
                    desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.1) - np.sin((time_step * np.pi) / 400)
                elif time_step < 300:
                    desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.1) - np.cos((time_step * np.pi) / 100)
                else:
                    desiredAngle = (self.nn.Get_Value_Of(neuronName) * 0.1) + np.cos((time_step * np.pi) / 400)
                self.motors[jointName].Set_Value(desiredAngle)
                self.motors[jointName].Act(desiredAngle)


    # We can modify this fitness function to record whether the link is touching the ground or not, or record from
    # another file the total number of iterations that all the links weren't touching the ground.
    def GetFitness(self, p_id):
        basePositionAndOrientation = pybullet.getBasePositionAndOrientation(self.id)

        basePosition = basePositionAndOrientation[0]
        xCoord = basePosition[0]
        yCoord = basePosition[1]
        zCoord = basePosition[2]

        f = open(f"tmp{p_id}.txt", "w")
        f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + ", " + str(self.max_height))
        os.rename(f"tmp{p_id}.txt", f"fitness{p_id}.txt")
        f.close()

        return
