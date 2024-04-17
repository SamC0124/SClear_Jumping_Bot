import solution
from solution import Solution
import copy
import random
import constants as c
import operator
import os


class PARALLEL_HILL_CLIMBER():
    def __init__(self):

        self.parents = {}
        self.children = {}
        self.unique_id = 0
        for idx in range(c.populationSize):
            self.parents[f'parent{idx}'] = Solution([[0 for _ in range(c.numMotorNeurons)] for _ in range(c.numSensorNeurons)], idx)


    def Evolve(self):
        self.Evaluate(self.parents)
        for gen in range(c.numberOfGenerations):
            self.Evolve_For_One_Generation()
            self.Evaluate(self.children)
            self.Print()
            self.Select(gen)
        self.Show_Best()


    def Evaluate(self, solutions):
        for key in solutions.keys():
            solutions[key].Start_Simulation("DIRECT")
            solutions[key].Wait_For_Simulation_To_End()


    def Evolve_For_One_Generation(self):
        self.Spawn()
        self.Mutate()


    def Print(self):
        parent_keys = self.parents.keys()
        parent_keys = list(parent_keys)
        child_keys = self.children.keys()
        child_keys = list(child_keys)
        for idx in range(len(parent_keys)):
            print(f"{self.parents[parent_keys[idx]].fitness[3]} Fitness for Parent {parent_keys[idx]}, {self.children[child_keys[idx]].fitness[3]} Fitness for Child #{child_keys[idx]}")
        print("\n")


    def Spawn(self):
        self.children = {}
        for parent in self.parents.keys():
            self.children[self.unique_id] = copy.deepcopy(self.parents[parent])
            self.unique_id += 1


    def Mutate(self):
        for child_key in self.children.keys():
            total_rows = c.numSensorNeurons
            total_cols = c.numMotorNeurons
            self.children[child_key].weights[random.randint(0, total_rows - 1)][random.randint(0, total_cols - 1)] = random.random() * 2 - 1


    def Select(self, generation):
        parent_keys = self.parents.keys()
        parent_keys = list(parent_keys)
        child_keys = self.children.keys()
        child_keys = list(child_keys)
        # We will perform a first selection based off the best node between the parent and the child
        for idx in range(len(parent_keys)):
            if float(self.parents[parent_keys[idx]].fitness[3]) < float(self.children[child_keys[idx]].fitness[3]):
                self.parents[parent_keys[idx]] = self.children[child_keys[idx]]

        # # Every 5th generation, we will then select the best parent between all the parent nodes, and apply their neural network to every robot.
        # if generation % 5 == 0 and generation != 0:
        #     # Find best parent neural network
        #     best_parent_index = 0
        #     for idx in range(len(parent_keys)):
        #         if float(self.parents[parent_keys[idx]].fitness[3]) > float(self.parents[parent_keys[best_parent_index]].fitness[3]):
        #             best_parent_index = self
        #
        #     # Set all parents to best parent value
        #     for idx in range(len(parent_keys)):
        #         self.parents[parent_keys[idx]] = self.parents[parent_keys[best_parent_index]]

    def Show_Best(self):
        best_parent = list(self.parents.items())[0][1]
        for parent_key in self.parents.keys():
            if float(self.parents[parent_key].fitness[3]) > float(best_parent.fitness[3]):
                self.parents[parent_key] = best_parent
        best_parent.Start_Simulation("GUI")