import solution
from solution import Solution
import copy
import random
import CONSTANTS
import operator
import os


class HILL_CLIMBER():
    def __init__(self):
        self.parent = Solution([[0, 0], [0, 0], [0, 0]], 0)
        self.child = None


    def Evolve(self):
        # for gen in range(CONSTANTS.numberOfGenerations):
        #     self.parent.Evaluate("DIRECT")
        #
        #     self.Spawn()
        #
        #     self.Mutate()
        #
        #     self.child.Evaluate("DIRECT")
        #
        #     self.Select()
        #
        # self.Show_Best()
        pass


    def Spawn(self):
        self.child = copy.deepcopy(self.parent)


    def Mutate(self):
        # Assume that total number of rows and columns is uniform for each synapse, since each neuron is connected to
        # each next neuron in the hidden layer.
        total_rows = len(self.parent.weights)
        total_cols = len(self.parent.weights[0])
        self.child.weights[random.randint(0, total_rows - 1)][random.randint(0, total_cols - 1)] = random.random() * 2 - 1


    def Select(self):
        print(f"Parent ({self.parent.fitness[0]}) vs Child ({self.child.fitness[0]})")
        if float(self.parent.fitness[0]) > float(self.child.fitness[0]):
            self.parent = self.child


    def Show_Best(self):
        os.system("python3 simulate.py GUI")
