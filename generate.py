import pyrosim.pyrosim as pyrosim
import random
import string


def Create_World():
    # Creating World
    pyrosim.Start_SDF("world.sdf")
    length = 1
    width = 1
    height = 1
    currHeight = 0.5
    totalBoxIdx = 0
    initXPos = -2.5
    initYPos = -2.5

    pyrosim.End()


def Generate_Body():
    # Creating Robot Parts
    pyrosim.Start_URDF("body.urdf")
    length = 1
    width = 1
    height = 1
    currHeight = 0.5
    initXPos = 2.5
    initYPos = 2.5
    pyrosim.Send_Cube(name=f"Body", pos=[1.5, 0, 1.5], size=[length, width, height])
    pyrosim.Send_Joint(name=f"Body_BackLeg", parent="Body", child="BackLeg", type="revolute", position=[1.0, 0, 1.0])
    pyrosim.Send_Cube(name=f"BackLeg", pos=[-0.5, 0, -0.5], size=[length, width, height])
    pyrosim.Send_Joint(name=f"Body_FrontLeg", parent="Body", child="FrontLeg", type="revolute", position=[2.0, 0, 1.0])
    pyrosim.Send_Cube(name=f"FrontLeg", pos=[0.5, 0, -0.5], size=[length, width, height])

    pyrosim.End()


def Generate_Brain():
    pyrosim.Start_NeuralNetwork("brain.nndf")

    length = 1
    width = 1
    height = 1
    currHeight = 0.5
    initXPos = 2.5
    initYPos = 2.5

    pyrosim.Send_Sensor_Neuron(name=0, linkName="Body")
    pyrosim.Send_Sensor_Neuron(name=1, linkName="FrontLeg")
    pyrosim.Send_Sensor_Neuron(name=2, linkName="BackLeg")
    pyrosim.Send_Motor_Neuron(name=3, jointName="Body_BackLeg")
    pyrosim.Send_Motor_Neuron(name=4, jointName="Body_FrontLeg")
    # pyrosim.Send_Synapse(sourceNeuronName=0, targetNeuronName=3, weight=0.8)
    # pyrosim.Send_Synapse(sourceNeuronName=1, targetNeuronName=3, weight=1.0)
    # pyrosim.Send_Synapse(sourceNeuronName=0, targetNeuronName=3, weight=0.8)
    # pyrosim.Send_Synapse(sourceNeuronName=1, targetNeuronName=3, weight=0.6)
    # pyrosim.Send_Synapse(sourceNeuronName=0, targetNeuronName=4, weight=-1.0)
    # pyrosim.Send_Synapse(sourceNeuronName=1, targetNeuronName=4, weight=-1.0)
    # pyrosim.Send_Synapse(sourceNeuronName=0, targetNeuronName=4, weight=-0.8)
    # pyrosim.Send_Synapse(sourceNeuronName=1, targetNeuronName=4, weight=-1.0)

    # Random Search Functionality
    for sensor_synapse in [0, 1, 2]:
        for motor_synapse in [3, 4]:
            pyrosim.Send_Synapse(sourceNeuronName=sensor_synapse, targetNeuronName=motor_synapse,
                                 weight=random.randrange(-1, 1))

    pyrosim.End()


# Function calls
Create_World()
Generate_Body()
Generate_Brain()

# Concludes file