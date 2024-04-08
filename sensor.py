import numpy as np
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import constants as c

class SENSOR():

    def __init__(self, name_of_link: str = ""):
        self.values = np.zeros(c.iterations)
        self.name = name_of_link

    def Get_Value(self, index: int):
        val = pyrosim.Get_Touch_Sensor_Value_For_Link(self.name)
        self.values[index] = val
        return val

    def Export(self):
        np.savetxt(f"sensoryData/{self.name}Values.csv", self.values, delimiter=",")
        np.save(f"sensoryData/{self.name}ValuesNumpy", self.values)
