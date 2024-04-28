# Search.py
# Main running file for evolutionary algorithm of Jumping Robot final project.

import constants as c
import os
import parallel_hill_climber as phc
import time

jumper_bot = phc.PARALLEL_HILL_CLIMBER()
jumper_bot.Evolve()

time.sleep(3)


