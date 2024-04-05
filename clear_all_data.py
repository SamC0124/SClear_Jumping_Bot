import CONSTANTS as c
import os


for i in range(c.populationSize):
    os.system(f"rm body{i}.urdf")
    os.system(f"rm brain{i}.nndf")
    os.system(f"rm fitness{i}.txt")
    os.system(f"rm world{i}.sdf")