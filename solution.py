import numpy as np
import os
import pyrosim
from pyrosim import pyrosim
import time
import CONSTANTS as c


class Solution():
    def __init__(self, weights_by_motor, id):
        self.weights = []
        self.myId = id
        for sensor_weights in weights_by_motor:
            weights_to_store = []
            for motor_weight in sensor_weights:
                weights_to_store.append(np.random.rand() * 2 - 1)
            self.weights.append(weights_to_store)
            weights_to_store = []


    def Start_Simulation(self, view_version):
        # Function calls
        self.Create_World()
        self.Generate_Body()
        self.Generate_Brain()

        if view_version != "DIRECT" and view_version != "GUI":
            raise Exception("INVALID VIEWING TYPE GIVEN")

        status = os.system(f"python3 simulate.py {view_version} {self.myId} 2&>1 &")
        print(status)


    def Wait_For_Simulation_To_End(self):
        while not os.path.exists(f"fitness{self.myId}.txt"):
            time.sleep(0.5)

        f = open(f"fitness{self.myId}.txt", "r")
        fitness_current = f.readline().split(", ")
        self.fitness = fitness_current
        time.sleep(0.5)
        os.system(f"rm fitness{self.myId}.txt")

    def Create_World(self):
        # Creating World
        pyrosim.Start_SDF(f"world{self.myId}.sdf")
        length = 1
        width = 1
        height = 1
        currHeight = 0.5
        totalBoxIdx = 0
        initXPos = -2.5
        initYPos = -2.5

        pyrosim.End()


    def Generate_Body(self):
        # Creating Robot Parts
        pyrosim.Start_URDF(f"body{self.myId}.urdf")
        length = 0.2
        width = 1
        height = 0.2
        currHeight = 0.5
        initXPos = 2.5
        initYPos = 2.5
        pyrosim.Send_Cube(name=f"Body", pos=[0, 0, 1], size=[1, 1, 1])
        pyrosim.Send_Joint(name=f"Body_FrontLeg", parent="Body", child="FrontLeg", type="revolute",
                           position=[0, 0.5, 1.0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name=f"FrontLeg", pos=[0, 0.5, 0], size=[length, width, height])
        pyrosim.Send_Joint(name=f"FrontLeg_FrontRear", parent="FrontLeg", child="FrontRear", type="revolute",
                           position=[0, 0.5, 0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name=f"FrontRear", pos=[0, 0.5, -0.5], size=[length, 0.2, 1.0])
        pyrosim.Send_Joint(name=f"Body_BackLeg", parent="Body", child="BackLeg", type="revolute",
                           position=[0, -0.5, 1.0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name=f"BackLeg", pos=[0, -0.5, 0], size=[length, width, height])
        pyrosim.Send_Joint(name=f"BackLeg_BackRear", parent="BackLeg", child="BackRear", type="revolute",
                           position=[0, -0.5, 0], jointAxis="1 0 0")
        pyrosim.Send_Cube(name=f"BackRear", pos=[0, -0.5, -0.5], size=[length, 0.2, 1])
        pyrosim.Send_Joint(name=f"Body_LeftLeg", parent="Body", child="LeftLeg", type="revolute",
                           position=[0.5, 0, 1.0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name=f"LeftLeg", pos=[0.5, 0, 0], size=[1, 0.2, height])
        pyrosim.Send_Joint(name=f"LeftLeg_LeftRear", parent="LeftLeg", child="LeftRear", type="revolute",
                           position=[0.5, 0, 0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name=f"LeftRear", pos=[0.5, 0, -0.5], size=[0.2, 0.2, 1.0])
        pyrosim.Send_Joint(name=f"Body_RightLeg", parent="Body", child="RightLeg", type="revolute",
                           position=[-0.5, 0, 1.0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name=f"RightLeg", pos=[-0.5, 0, 0], size=[1, 0.2, height])
        pyrosim.Send_Joint(name=f"RightLeg_RightRear", parent="RightLeg", child="RightRear", type="revolute",
                           position=[-0.5, 0, 0], jointAxis="0 1 0")
        pyrosim.Send_Cube(name=f"RightRear", pos=[-0.5, 0, -0.5], size=[0.2, 0.2, 1.0])

        pyrosim.End()


    def Generate_Brain(self):
        pyrosim.Start_NeuralNetwork(f"brain{self.myId}.nndf")
        pyrosim.Send_Sensor_Neuron(name=0, linkName="Body")
        link_index = 1
        # Loop through each of the known links of the robot
        for link in ["FrontLeg", "BackLeg", "RightLeg", "LeftLeg"]:
            pyrosim.Send_Sensor_Neuron(name=link_index, linkName=link)
            pyrosim.Send_Motor_Neuron(name=link_index + c.numSensorNeurons - 1, jointName=f"Body_{link}")
            link_index += 1
        for link in ["Front", "Back", "Right", "Left"]:
            pyrosim.Send_Sensor_Neuron(name=link_index, linkName=link + "Rear")
            pyrosim.Send_Motor_Neuron(name=link_index + c.numSensorNeurons - 1, jointName=f"{link}Leg_{link}Rear")
            link_index += 1
        # Random Search Functionality
        for sensor_synapse in range(0, c.numSensorNeurons):
            for motor_synapse in range(c.numSensorNeurons, c.numSensorNeurons + c.numMotorNeurons):
                pyrosim.Send_Synapse(sourceNeuronName=sensor_synapse, targetNeuronName=motor_synapse, weight=self.weights[sensor_synapse][motor_synapse - c.numSensorNeurons])
        pyrosim.End()
