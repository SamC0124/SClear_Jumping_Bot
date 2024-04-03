import numpy as np
import os
import pybullet
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import CONSTANTS as c
from motor import MOTOR
from sensor import SENSOR
from pyrosim.neuralNetwork import NEURAL_NETWORK

class ROBOT:

    def __init__(self, id: int, m: int = 0, s: int = 0, p_id: int = 0):
        self.Prepare_To_Sense(m, s)
        self.Prepare_To_Act()
        self.id = id
        self.nn = NEURAL_NETWORK(f"brain{p_id}.nndf")

    def Prepare_To_Sense(self, num_motors: int = 0, num_sensors: int = 0):
        self.sensors = {}

        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = SENSOR(linkName)

    def Prepare_To_Act(self):

        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            if jointName == b'Body_BackLeg':
                self.motors[jointName] = MOTOR(jointName, frequency=c.frequency/2)
            else:
                self.motors[jointName] = MOTOR(jointName)

    def Sense(self, index: int):
        for sensor in self.sensors.values():
            pass

    def Think(self):
        self.nn.Update()

    def Act(self):

        for neuronName in self.nn.Get_Neuron_Names():
            if self.nn.Is_Motor_Neuron(neuronName):
                jointName = self.nn.Get_Motor_Neurons_Joint(neuronName).encode('utf-8')
                desiredAngle = self.nn.Get_Value_Of(neuronName) * c.motorAngleRange
                self.motors[jointName].Set_Value(desiredAngle)
                self.motors[jointName].Act(desiredAngle)


    def GetFitness(self, p_id):
        stateZero = pybullet.getLinkState(self.id, 0)

        positionOfStateZero = stateZero[0]
        xCoord = positionOfStateZero[0]
        yCoord = positionOfStateZero[1]
        zCoord = positionOfStateZero[2]

        f = open(f"tmp{p_id}.txt", "w")
        f.write(str(xCoord) + ", " + str(yCoord) + ", " + str(zCoord) + "\n")
        os.rename(f"tmp{p_id}.txt", f"fitness{p_id}.txt")
        f.close()

        return
