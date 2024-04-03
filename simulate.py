import CONSTANTS as c
import math
import numpy as np
import os
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import random as rand
import sys
import time

import simulation
from simulation import SIMULATION

viewing_mode = sys.argv[1]
id = int(sys.argv[2])
simulate = SIMULATION(viewing_mode, id)

simulate.Run()

time.sleep(10)
for i in range(c.populationSize):
    os.system(f"rm body{i}.sdf")
    os.system(f"rm brain{i}.sdf")
    os.system(f"rm fitness{i}.sdf")
