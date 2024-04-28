import numpy as np
import os
import pyrosim
from pyrosim import pyrosim
import time
import constants as c

# Recording best weights of iteration 1 (divided into A and B teams)
a_current = [[-0.6364808265795205, 0.6622059507753046, 0.6569773302148081, -0.12608529098289045, -0.4649166024387603, -0.3098995207213371, 0.8144949906494703, 0.09492969804952289, 0.724327764729586, -0.8722745173721602, -0.4748470358873571, 0.2353168558988521],
 [-0.897246300599879, -0.6900688127960968, -0.35123992319886654, 0.7300215696015402, 0.991912022673147, 0.41527589644878415, -0.8571918010058637, -0.2951375823539655, -0.49896317629636, 0.4981132208056247, -0.012215621484961803, 0.9033454252045869],
 [-0.0539845065400586, -0.45136878388395374, -0.051096292498863605, 0.39838454666095235, 0.9796515701622754, 0.2590018177890765, -0.7083733334128008, 0.7021758815148067, 0.58586013781939, 0.16250735327657595, -0.5464487247927858, -0.7165349582464811],
 [0.0493430565801416, 0.7594913712914169, -0.6247190421907762, -0.5806525404028307, 0.32386739092180306, 0.19819720614683733, -0.5559883505396237, -0.7870853514815228, 0.6375615538318073, 0.1336775094718472, -0.905806279799761, 0.8851485734272801],
 [0.6848806777601026, 0.44285496599959906, 0.6431411717876787, 0.40924165653154754, 0.5102470400939481, 0.2363494283378733, 0.963339947246006, 0.35155527245941487, -0.09828457477831543, 0.13171407398594392, 0.9807198187361206, 0.5551018719255545],
 [-0.6328564316889203, 0.17802899069107037, -0.9023909390692353, -0.05505675388160691, 0.542359851567997, -0.5179951290981968, 0.032957958067615145, 0.8084950266919644, -0.7557911035367821, 0.9285100758315497, 0.04115343435287144, -0.5859755199860897],
 [-0.3424863205573976, -0.1818752339599945, -0.4662396573487333, 0.478031029802934, 0.7881154539677002, 0.13222507957555174, 0.44077022800100574, 0.5774722920217972, 0.43797067096821674, 0.8166887752882988, 0.1911140237679374, -0.11681700729674138],
 [0.8398529405948487, 0.5271162436996437, -0.5548237544231673, -0.37414992410169967, 0.31214105012501236, 0.3579458921794092, 0.5882625081255763, -0.8234633192387979, 0.39470461556144487, 0.24959879695732146, -0.6684640971461382, 0.9234317061702229],
 [0.9234317061702229, 0.8869877980386878, 0.49916495056658317, 0.7026774948625758, 0.915982320346614, 0.30613089689519857, -0.6438279119591939, -0.9723643023953314, 0.22241839266384011, -0.22483041350306188, -0.409544644851793, -0.9197281790175371],
 [0.852604553095625, -0.0976370875560384, 0.5519308292506213, -0.8865598613308423, -0.971733592711326, 0.049318616801290904, -0.30346624956155677, 0.09006154957428425, 0.7877675984957231, -0.9011003707189145, 0.08786136775601894, -0.1614271999443324],
 [-0.45476462519355487, 0.20150615891776957, -0.17519469727886694, -0.2901738651204435, -0.021757799352028595, -0.9232134172597317, -0.8123066003452777, -0.2798670436138342, -0.6442263032976414, -0.2856475433613934, -0.0745471453011659, 0.3248316836489482],
 [0.002980008444390947, 0.45678995240310494, -0.16754873125322622, -0.5000484077855918, 0.7129205826963836, 0.1759625253569883, 0.8170329833196188, -0.40875567663486034, 0.8663950846826687, 0.09083001923591638, -0.1474639520396559, -0.12683125590834332],
 [-0.33718231819061595, -0.437227350097783, 0.6840297092584031, -0.4354642144855543, 0.34019243579539316, -0.35177832616631366, -0.46103328700945, -0.04935969781796845, 0.670150960127446, 0.014573382549434077, 0.02841568506851888, 0.9063340742083155]]
b_current = [[0.6364808265795205, 0.6622059507753046, 0.6569773302148081, 0.12608529098289045, 0.5729299858808401, 0.3098995207213371, 0.8144949906494703, 0.09492969804952289, 0.6461641810917595, 0.8722745173721602, 0.4748470358873571, 0.14492917704634056, 0.47972123153668833],
             [0.6900688127960968, 0.016288403281151664, 0.7300215696015402, 0.7603596964795891, 0.5486819820732616, 0.8571918010058637, 0.2951375823539655, 0.31105552363311273, 0.4981132208056247, 0.5948899843473934, 0.4418063403668704, 0.0539845065400586, 0.6957064319347013],
             [0.786013597744806, 0.39838454666095235, 0.9796515701622754, 0.2590018177890765, 0.3518718778554082, 0.7021758815148067, 0.58586013781939, 0.16250735327657595, 0.3302360246288305, 0.7165349582464811, 0.0493430565801416, 0.7594913712914169, 0.32465755945523145],
             [0.34030033197989784, 0.32386739092180306, 0.19819720614683733, 0.7041297093890873, 0.13969854552659577, 0.6375615538318073, 0.1336775094718472, 0.905806279799761, 0.8851485734272801, 0.6848806777601026, 0.44285496599959906, 0.6431411717876787, 0.07128265681292234],
             [0.5102470400939481, 0.2363494283378733, 0.963339947246006, 0.49594175783286265, 0.7563874822644454, 0.13171407398594392, 0.9807198187361206, 0.5551018719255545, 0.6328564316889203, 0.17802899069107037, 0.9023909390692353, 0.05505675388160691, 0.542359851567997],
             [0.5179951290981968, 0.032957958067615145, 0.8084950266919644, 0.7557911035367821, 0.9285100758315497, 0.04115343435287144, 0.7974917423828141, 0.512601362064671, 0.1818752339599945, 0.4341889178866998, 0.478031029802934, 0.7881154539677002, 0.13222507957555174],
             [0.44077022800100574, 0.5774722920217972, 0.43797067096821674, 0.8166887752882988, 0.1911140237679374, 0.8993266671560367, 0.8398529405948487, 0.5271162436996437, 0.5548237544231673, 0.733005688791059, 0.31214105012501236, 0.3579458921794092, 0.5171332362109646],
             [0.8234633192387979, 0.39470461556144487, 0.24959879695732146, 0.0267520193778108, 0.6684640971461382, 0.9234317061702229, 0.8869877980386878, 0.49916495056658317, 0.7026774948625758, 0.915982320346614, 0.30613089689519857, 0.2894917247104045, 0.9723643023953314],
             [0.22241839266384011, 0.22483041350306188, 0.409544644851793, 0.9197281790175371, 0.852604553095625, 0.0976370875560384, 0.5519308292506213, 0.8865598613308423, 0.971733592711326, 0.049318616801290904, 0.2922843864663973, 0.014644767675313863, 0.6820682445082678],
             [0.9011003707189145, 0.08786136775601894, 0.1614271999443324, -0.45476462519355487, 0.20150615891776957, -0.4595596605474308, -0.2901738651204435, -0.021757799352028595, -0.9232134172597317, -0.3563677346732639, -0.2798670436138342, -0.6442263032976414, -0.4265724927515899],
             [-0.0745471453011659, 0.3248316836489482, 0.002980008444390947, 0.45678995240310494, 0.49822431520392674, -0.5000484077855918, 0.7129205826963836, 0.1759625253569883, -0.6466242427157216, -0.40875567663486034, 0.8663950846826687, 0.09083001923591638, -0.03292584195449244],
             [-0.12683125590834332, -0.33718231819061595, -0.437227350097783, 0.6840297092584031, 0.2924296036596201, 0.34019243579539316, 0.32673594876624246, -0.46103328700945, -0.04935969781796845, 0.670150960127446, 0.014573382549434077, 0.02841568506851888, 0.9063340742083155]]


prior_best_solutions = {"A": a_current, "B": b_current}

class Solution():
    def __init__(self, weights_by_motor, id):
        self.weights = []
        self.myId = id
        if (id % 10) < 5:
            self.weights = prior_best_solutions["A"]
        else:
            self.weights = prior_best_solutions["B"]


    def Start_Simulation(self, view_version):
        # Function calls
        self.Create_World()
        self.Generate_Body()
        self.Generate_Brain()

        if view_version != "DIRECT" and view_version != "GUI":
            raise Exception("INVALID VIEWING TYPE GIVEN")

        os.system(f"python3 simulate.py {view_version} {self.myId} 2&>1 &")


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
        width = 1.0
        height = 0.2
        currHeight = 0.5
        initXPos = 2.5
        initYPos = 2.5
        pyrosim.Send_Cube(name=f"Body", pos=[0, 0, 2], size=[1, 1, 1])
        pyrosim.Send_Joint(name=f"Body_FrontLeg", parent="Body", child="FrontLeg", type="revolute",
                           position=[0, 0.5, 2], jointAxis="1, 0, 0")
        pyrosim.Send_Cube(name=f"FrontLeg", pos=[0, 0.5, 0], size=[length, width, height])
        pyrosim.Send_Joint(name=f"FrontLeg_FrontCalf", parent="FrontLeg", child="FrontCalf", type="revolute",
                           position=[0, 1.0, 0], jointAxis="1, 0, 0")
        pyrosim.Send_Cube(name=f"FrontCalf", pos=[0, 0, -0.5], size=[length, 0.2, 1.0])
        pyrosim.Send_Joint(name=f"Body_BackLeg", parent="Body", child="BackLeg", type="revolute",
                           position=[0, -0.5, 2], jointAxis="1, 0, 0")
        pyrosim.Send_Cube(name=f"BackLeg", pos=[0, -0.5, 0], size=[length, width, height])
        pyrosim.Send_Joint(name=f"BackLeg_BackCalf", parent="BackLeg", child="BackCalf", type="revolute",
                           position=[0, -1.0, 0], jointAxis="1, 0, 0")
        pyrosim.Send_Cube(name=f"BackCalf", pos=[0, 0, -0.5], size=[length, 0.2, 1])
        pyrosim.Send_Joint(name=f"Body_LeftLeg", parent="Body", child="LeftLeg", type="revolute",
                           position=[0.5, 0, 2], jointAxis="0, 1, 0")
        pyrosim.Send_Cube(name=f"LeftLeg", pos=[0.5, 0, 0], size=[1, 0.2, height])
        pyrosim.Send_Joint(name=f"LeftLeg_LeftCalf", parent="LeftLeg", child="LeftCalf", type="revolute",
                           position=[1.0, 0, 0], jointAxis="0, 1, 0")
        pyrosim.Send_Cube(name=f"LeftCalf", pos=[0, 0, -0.5], size=[0.2, 0.2, 1.0])
        pyrosim.Send_Joint(name=f"Body_RightLeg", parent="Body", child="RightLeg", type="revolute",
                           position=[-0.5, 0, 2], jointAxis="0, 1, 0")
        pyrosim.Send_Cube(name=f"RightLeg", pos=[-0.5, 0, 0], size=[1, 0.2, height])
        pyrosim.Send_Joint(name=f"RightLeg_RightCalf", parent="RightLeg", child="RightCalf", type="revolute",
                           position=[-1.0, 0, 0], jointAxis="0, 1, 0")
        pyrosim.Send_Cube(name=f"RightCalf", pos=[0, 0, -0.5], size=[0.2, 0.2, 1.0])

        # Create each of the feet for the robot
        pyrosim.Send_Joint(name=f"RightCalf_RightFoot", parent="RightCalf", child="RightFoot", type="prismatic", position=[0, 0, -1.0], jointAxis="0 0 1")
        pyrosim.Send_Cube(name=f"RightFoot", pos=[0, 0, 0], size=[0.1, 0.1, 1.4])
        pyrosim.Send_Joint(name=f"LeftCalf_LeftFoot", parent="LeftCalf", child="LeftFoot", type="prismatic", position=[0, 0, -1.0], jointAxis="0 0 1")
        pyrosim.Send_Cube(name=f"LeftFoot", pos=[0, 0, 0], size=[0.1, 0.1, 1.4])
        pyrosim.Send_Joint(name=f"FrontCalf_FrontFoot", parent="FrontCalf", child="FrontFoot", type="prismatic", position=[0, 0, -1.0], jointAxis="0 0 1")
        pyrosim.Send_Cube(name=f"FrontFoot", pos=[0, 0, 0], size=[0.1, 0.1, 1.4])
        pyrosim.Send_Joint(name=f"BackCalf_BackFoot", parent="BackCalf", child="BackFoot", type="prismatic", position=[0, 0, -1.0], jointAxis="0 0 1")
        pyrosim.Send_Cube(name=f"BackFoot", pos=[0, 0, 0], size=[0.1, 0.1, 1.4])

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
            pyrosim.Send_Sensor_Neuron(name=link_index, linkName=link + "Calf")
            pyrosim.Send_Motor_Neuron(name=link_index + c.numSensorNeurons - 1, jointName=f"{link}Leg_{link}Calf")
            link_index += 1
        for link in ["Front", "Back", "Right", "Left"]:
            pyrosim.Send_Sensor_Neuron(name=link_index, linkName=link + "Foot")
            pyrosim.Send_Motor_Neuron(name=link_index + c.numSensorNeurons - 1, jointName=f"{link}Calf_{link}Foot")
            link_index += 1
        # Random Search Functionality
        # Implement from two previous two solutions for two teams (A and B)
        for sensor_synapse in range(0, c.numSensorNeurons):
            for motor_synapse in range(c.numSensorNeurons, c.numSensorNeurons + c.numMotorNeurons):
                pyrosim.Send_Synapse(sourceNeuronName=sensor_synapse, targetNeuronName=motor_synapse, weight=self.weights[sensor_synapse][motor_synapse - c.numSensorNeurons])
        pyrosim.End()
