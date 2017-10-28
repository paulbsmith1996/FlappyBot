import itertools
import numpy as np
import psutil
import PIL.ImageGrab
import webcolors
from time import sleep
import pyautogui
import pygame
import sys
import thread
import time
import threading
from pymouse import PyMouse
from random import *
from screen_viewer import *


class Creature:

    def __init__(self, weights):
        self.id = 0
        self.weights = weights
        self.score = 0
        self.alive = True


    def isAlive(self):
        return self.alive




#c = threading.Condition()

#--------------- Neural Evolution happens below ---------------------------#
class Bot:

    def updateRandomly(self, weights):
        return None
    def makeInitialWeights(self):
        return None

    def __init__(self, screen_def, num_neurons):
        self.n_hidden_1 = 64 # Number of nodes in hidden layer 1
        self.n_hidden_2 = 10 # Number of nodes in hidden layer 2
        self.num_input = num_neurons # Size of input is the length of the positions array
        self.num_classes = 2 # Size of the ouput is hit space or not

        self.num_creatures_per_gen = 24
        self.learning_rate = 0.10
        self.num_generations = 1000
        self.num_mutations = 4


        self.creatures = []

        for i in range(self.num_creatures_per_gen):
            self.creatures.append(Creature(self.updateRandomly(self.makeInitialWeights())))

        self.sv = ScreenViewer(screen_def, num_neurons)

        print "Bot initialized"

    def makeRandomWeights(self):
        return {
            'h1': np.random.rand(self.num_input,   self.n_hidden_1),
            'h2': np.random.rand(self.n_hidden_1,  self.n_hidden_2),
            'out': np.random.rand(self.n_hidden_2, self.num_classes)
            }

    def makeInitialWeights(self):
        return {
            'h1': np.zeros([self.num_input,   self.n_hidden_1]),
            'h2': np.zeros([self.n_hidden_1,  self.n_hidden_2]),
            'out': np.zeros([self.n_hidden_2, self.num_classes])
            }

    def updateRandomly(self, weights):
    
        new_weights = weights
        
        num_muts_h1 = randint(0, self.num_mutations)
        for i in range(num_muts_h1):
            new_weights['h1'][randint(0, self.num_input - 1)][randint(0, self.n_hidden_1 - 1)] = 1

        num_muts_h2 = randint(0, self.num_mutations)
        for i in range(num_muts_h2):
            new_weights['h2'][randint(0, self.n_hidden_1 - 1)][randint(0, self.n_hidden_2 - 1)] = 1

        num_muts_out = randint(0, self.num_mutations)
        for i in range(num_muts_out):
            new_weights['out'][randint(0, self.n_hidden_2 - 1)][randint(0, self.num_classes - 1)] = 1
        
        return new_weights

    def makeNearWeights(self, weights):
        return {
            'h1': weights['h1']   + self.noise(self.num_input,    self.n_hidden_1),
            'h2': weights['h2']   + self.noise(self.n_hidden_1,   self.n_hidden_2),
            'out': weights['out'] + self.noise(self.n_hidden_2, self.num_classes)
            }

    def noise(self, x, y):
        return [ij * self.learning_rate - (self.learning_rate / 2) for ij in np.random.rand(x, y)]




    # Pre: output layer is of size 4
    def executeOutput(self, output_layer):
        argmax = np.argmax(output_layer)

        if argmax == 1:
            pyautogui.click()

    # Create model                                                                                                    
    def neural_net(self, x, creature):
        layer_1 = np.dot(x, creature.weights['h1'])
        layer_2 = np.dot(layer_1, creature.weights['h2'])
        out_layer = np.dot(layer_2, creature.weights['out'])
    
        sum_out = np.sum(out_layer)
        if sum_out > 0:
            out_layer = [output / sum_out for output in out_layer]

        return out_layer

    def rankCreatures(self):

        for x in range(self.num_creatures_per_gen):
            for y in range(self.num_creatures_per_gen - 1):
                if self.creatures[y].score < self.creatures[y + 1].score:
                    temp = self.creatures[y]
                    self.creatures[y] = self.creatures[y + 1]
                    self.creatures[y+1] = temp

    def learn(self):

        dfc = self.sv.distanceFromCenter



        for g in range(self.num_generations):

            print "Generation " + str(g)

            for creature in self.creatures:

                print "new creature"
                pyautogui.press('space')
                pyautogui.press('space')
                
                num_apps = 0

                start = int(round(time.time() * 1000))
                while(creature.isAlive()):
                    startLoop = int(round(time.time() * 1000))
                    result = self.neural_net(self.sv.bitMap, creature)

                    self.executeOutput(result)
             
                    if self.sv.yellowCount > 0:
                        num_apps = num_apps + 1

                    #if self.sv.khakiCount > ScreenViewer.num_khaki:
                    if self.sv.yellowCount > 10:
                        creature.alive = False
                        self.sv.khakiCount = 0
                        dfc = self.sv.distanceFromCenter
                        sleep(2)
                        pyautogui.press('space')
                        #break


                    self.sv.getInput()
                    sleep(1.0 / 40.0)
                    endLoop = int(round(time.time() * 1000))
                    #print "Time to loop: " + str(endLoop - startLoop)

            
                end = int(round(time.time() * 1000))
                creature.score = ((end - start) / 1000.0) + num_apps #2 * dfc
                print creature.score
                creature.alive = True


            self.rankCreatures()

            self.creatures = self.creatures[:(self.num_creatures_per_gen / 4)]

            print "Ranked scores:\n"

            for creature in self.creatures:
                print creature.score

            for i in range(self.num_creatures_per_gen / 4):
                curCreature = self.creatures[i]
                self.creatures.append(Creature(self.makeNearWeights(curCreature.weights)))
                self.creatures.append(Creature(self.makeNearWeights(curCreature.weights)))

            for i in range(self.num_creatures_per_gen / 4):
                self.creatures.append(Creature(self.makeRandomWeights()))

            sleep(2)

        best_weights = self.creatures[0].weights

        h1_text = "["
        for line in best_weights['h1']:
            h1_text = h1_text + "["
            for ele in line:
                h1_text = h1_text + str(ele) + " "
            h1_text = h1_text + "]\n"
        h1_text = h1_text + "]"

        h1_file = open('h1.txt', 'w')
        h1_file.write(h1_text)
        h1_file.close()

        h2_file = open('h2.txt', 'w')
        h2_file.write(str(best_weights['h2']))
        h2_file.close()
        
        out_file = open('out.txt', 'w')
        out_file.write(str(best_weights['out']))
        out_file.close()
        
        print best_weights['h1']
        print best_weights['h2']
        print best_weights['out']


screen_def = (620, 20, 500, 800)
b = Bot(screen_def, 1024)
b.learn()

