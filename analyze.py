import matplotlib.pyplot as mptpy
import numpy as np
import time as t

backLegData = np.load("sensoryData/b'Body_BackLeg'MotorValuesNumpy.npy")
frontLegData = np.load("sensoryData/b'Body_FrontLeg'MotorValuesNumpy.npy")


mptpy.plot(backLegData, linewidth=1, label="Generic Sine Data for Rotating between -pi and pi")
mptpy.plot(frontLegData, linewidth=1, label="Generic Sine Data for Rotating between -pi and pi")
mptpy.legend()
mptpy.show()

t.sleep(10)
exit()

