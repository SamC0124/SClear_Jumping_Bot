# Motor.py class
# Class containing the basic motor functionality for Pyrosim robots

import numpy as np
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import constants as c

class MOTOR():

    def __init__(self, jointName: bytes, amplitude: int = c.amplitude, offset: int = c.phaseOffset, frequency: int = c.frequency, force: int = c.standardMotorForce):
        self.movementMapping = np.zeros(c.iterations)
        self.jointOfOrigin = jointName
        self.amplitude = amplitude
        self.frequency = frequency
        self.phaseOffset = offset
        self.motorForce = force

    # This function prepares the robot to move with a sinusoidal motion. However, our robot won't be initialized with
    # that in mind, instead it must find a way to propel itself upwards whenever it will hit the ground.
    def Prepare_To_Act(self, p_movements: list):
        self.movementMapping = [np.sin(self.frequency * (i * (np.pi / (c.iterations / 2)) - self.phaseOffset)) * self.amplitude for i in range(c.iterations)]


    def Act(self, desired_angle, newAmplitude = -1, givenForce = -1):
        self.Set_Value(desired_angle, newAmplitude, givenForce)


    def Set_Value(self, desiredAngle, newAmplitude, givenForce):
        if givenForce == -1:
            givenForce = self.motorForce
        if newAmplitude == -1:
            newAmplitude = c.amplitude
        else:
            self.amplitude = newAmplitude
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=0,
            jointName=self.jointOfOrigin,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=desiredAngle,
            maxForce=givenForce)