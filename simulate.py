import constants as c
import math
import numpy as np
import os
import pybullet as pb
import pybullet_data
import pyrosim.pyrosim as pyrosim
import random as rand
import simulation
from simulation import SIMULATION
import sys
import time

viewing_mode = sys.argv[1]
id = int(sys.argv[2])
simulate = SIMULATION(viewing_mode, id)

simulate.Run()
