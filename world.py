import numpy as np
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import CONSTANTS as c

class WORLD():

    def __init__(self, id: any = None, r: int = 0):

        self.entities = {}

        self.planeId = pb.loadURDF("plane.urdf")
        pb.loadSDF(f"world{id}.sdf")