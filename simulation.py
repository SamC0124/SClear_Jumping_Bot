import constants as c
import numpy as np
import os
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import time
from motor import MOTOR
from sensor import SENSOR
from robot import ROBOT
from world import WORLD

class SIMULATION:

    def __init__(self, view_type, p_id):

        self.view_mode = view_type
        self.p_id = p_id

        # Initializing physics
        if self.view_mode == "DIRECT":
            self.physicsClient = pb.connect(pb.DIRECT)
        elif self.view_mode == "GUI":
            self.physicsClient = pb.connect(pb.GUI)
        else:
            raise Exception("INVALID VIEW TYPE GIVEN")
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)
        self.robotId = pb.loadURDF(f"body{self.p_id}.urdf")
        pyrosim.Prepare_To_Simulate(self.robotId)

        # Initializing objects
        self.robot = ROBOT(id=self.robotId, p_id=self.p_id)
        self.world = WORLD(id=p_id)


    def Run(self):
        for i in range(c.iterations):
            self.robot.Sense(iteration=i)
            self.robot.Think()
            self.robot.Act(i)

            pb.stepSimulation(self.physicsClient)

            time.sleep(0.001)

        self.Get_Fitness()


    def Save_Values(self):
        pass
        # for motor in self.robot.motors.values():
        #     np.savetxt(f"sensoryData/{motor.jointOfOrigin}MotorValues.csv", motor.movementMapping, delimiter=",")
        #     np.save(f"sensoryData/{str(motor.jointOfOrigin)}MotorValuesNumpy", motor.movementMapping)
        # for sensor in self.robot.sensors.values():
        #     np.savetxt(f"sensoryData/{sensor.name}SensorValues.csv", sensor.values, delimiter=",")
        #     np.save(f"sensoryData/{sensor.name}SensorValuesNumpy", sensor.values)


    def Get_Fitness(self):
        self.robot.GetFitness(self.p_id)


    def __del__(self):
        self.Save_Values()
        pb.disconnect()
