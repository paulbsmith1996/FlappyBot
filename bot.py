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


screenX = 600 
screenY = 135
imageSize = pyautogui.size()
#screenWidth = imageSize[0] / 4
screenWidth = 520
screenHeight = 690


offset = 100
MOUSE_UP = [screenWidth / 2, screenHeight / 2 - offset]
MOUSE_DOWN = [screenWidth / 2, screenHeight / 2 + offset]
MOUSE_RIGHT = [screenWidth / 2 + offset, screenHeight / 2]
MOUSE_LEFT = [screenWidth / 2 - offset, screenHeight / 2]

big_mod = 1000000

max_color_distance = 200
max_color_distance_khaki = 40
max_color_distance_yellow = 100
greenRGB = (0, 255, 0, 255)
khakiRGB = (240, 230, 140, 255)
yellowRGB = (255, 255, 0, 255)

class Creature:

    def __init__(self, weights):
        self.id = 0
        self.weights = weights
        self.score = 0
        self.alive = True


    def isAlive(self):
        #self.alive = red_count() < num_red)
        #return red_count() < num_red)
        return self.alive

num_khaki = 30

num_pos = 625 # Pick a square number here
sqrt_num_pos = int(num_pos ** 0.5)

pos_x = [i * ((screenWidth - 1) / (sqrt_num_pos - 1)) for i in range(sqrt_num_pos)]
pos_y = [i * ((screenHeight - 1) / (sqrt_num_pos - 1)) for i in range(sqrt_num_pos)]

positions = list(itertools.product(pos_x, pos_y))

def getColors():
    loaded_image = PIL.ImageGrab.grab(bbox=(screenX, screenY, screenX + screenWidth, screenY + screenHeight)).load()
    return [loaded_image[pos[0], pos[1]] for pos in positions]

# Credit to fraxel on 
# https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green
# for color conversion
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def khaki_count():
    
    count = 0
    for rgb in colorRGBs:
        if closest_color(rgb) == 'khaki':
            count = count + 1

    return count

#print "grey_count is " + str(grey_count()) + ", positions size frac is " + str(grey_frac * len(positions))







c = threading.Condition()

#---------------- Launch new pyGame window -------------------------------#

windowWidth = 200
windowHeight = 150
win_offset = 10

FPS = 30
winTick = 1.0 / 30.0

colorRGBs = getColors()
#colorInputHexes = [int(webcolors.rgb_to_hex(rgb)[1:], 16) % big_mod for rgb in colorRGBs]
bitMap = np.zeros(len(colorRGBs))

khakiCount = 0
yellowCount = 0

cellWidth = (windowWidth - (2 * win_offset)) / sqrt_num_pos
cellHeight = (windowHeight - (2 * win_offset)) / sqrt_num_pos

distanceFromCenter = 0

pygame.init()
screen = pygame.display.set_mode((windowWidth, windowHeight), pygame.RESIZABLE)

def run_pygame():

    global colorRGBs
    global colorInputHexes
    global khakiCount

    while(True):

        start = int(round(time.time() * 1000))
        c.acquire()
        colorRGBs = getColors()
        end = int(round(time.time() * 1000))
        #colorInputHexes = [int(webcolors.rgb_to_hex(rgb)[1:], 16) / big_mod for rgb in colorRGBs]        
        khakiCount = 0
        greenCount = 0
        c.release()

        end2 = int(round(time.time() * 1000))
        #print "Array modifications in: " + str(end2 - end) + " ms"

        # Draw in Pygame window what neurons see
        for x in range(sqrt_num_pos):
            for y in range(sqrt_num_pos):

                curPixel = colorRGBs[x * sqrt_num_pos + y]
                distance_from_green = sum([(curPixel[i] - greenRGB[i]) ** 2 for i in range(4)]) ** 0.5
                distance_from_khaki = sum([(curPixel[i] - khakiRGB[i]) ** 2 for i in range(4)]) ** 0.5


                if (int(distance_from_green) < int(max_color_distance)):
                    curPixel = (0, 0, 0, 255)                    
                    greenCount = greenCount + 1
                    c.acquire()
                    bitMap[x * sqrt_num_pos + y] = 1
                    c.release()
                else:
                    curPixel = (255, 255, 255, 255)
                    c.acquire()
                    bitMap[x * sqrt_num_pos + y] = 0
                    c.release()

                if distance_from_khaki < max_color_distance_khaki:
                    khakiCount = khakiCount + 1

                rectCoords = (win_offset + (x * cellWidth), win_offset + (y * cellHeight), cellWidth, cellHeight)
                pygame.draw.rect(screen, curPixel, rectCoords, 0)

        #print "kc: " + str(khakiCount)
        #print "gc: " + str(greenCount)
        pygame.display.update()
        end3 = int(round(time.time() * 1000))
        #print "loop runs in: " + str(end3 - start)

def getInput():

    global colorRGBs
    global khakiCount
    global yellowCount
    global distanceFromCenter

    start = int(round(time.time() * 1000))
    c.acquire()
    colorRGBs = getColors()
    end = int(round(time.time() * 1000))
    khakiCount = 0
    greenCount = 0
    yellowCount = 0
    c.release()

    end2 = int(round(time.time() * 1000))
    
    for x in range(sqrt_num_pos):
        for y in range(sqrt_num_pos):

            curPixel = colorRGBs[x * sqrt_num_pos + y]
            distance_from_green = sum([(curPixel[i] - greenRGB[i]) ** 2 for i in range(4)]) ** 0.5
            distance_from_khaki = sum([(curPixel[i] - khakiRGB[i]) ** 2 for i in range(4)]) ** 0.5
            distance_from_yellow = sum([(curPixel[i] - yellowRGB[i]) ** 2 for i in range(4)]) ** 0.5


            if (int(distance_from_green) < int(max_color_distance)):
                greenCount = greenCount + 1
                curPixel = (0, 0, 0, 255)
                c.acquire()
                bitMap[x * sqrt_num_pos + y] = 1
                c.release()
            elif (int(distance_from_yellow) < int(max_color_distance_yellow)):
                curPixel = (255, 0, 0, 255)
                yellowCount = yellowCount + 1
                c.acquire()
                bitMap[x * sqrt_num_pos + y] = 1000
                distanceFromCenter = (y - sqrt_num_pos) ** 2
                c.release()
            else:
                curPixel = (255, 255, 255, 255)
                c.acquire()
                bitMap[x * sqrt_num_pos + y] = 0
                c.release()

            if distance_from_khaki < max_color_distance_khaki:
                khakiCount = khakiCount + 1

            rectCoords = (win_offset + (x * cellWidth), win_offset + (y * cellHeight), cellWidth, cellHeight)
            pygame.draw.rect(screen, curPixel, rectCoords, 0)

        #print "kc: " + str(khakiCount)
        #print "gc: " + str(greenCount)
    pygame.display.update()
    end3 = int(round(time.time() * 1000))


    if not yellowCount < 1:
        distanceFromCenter = 50

    end3 = int(round(time.time() * 1000))







#--------------- Neural Evolution happens below ---------------------------#



n_hidden_1 = 64 # Number of nodes in hidden layer 1
n_hidden_2 = 10 # Number of nodes in hidden layer 2
num_input = len(positions) # Size of input is the length of the positions array
num_classes = 2 # Size of the ouput is hit space or not

num_creatures_per_gen = 16
learning_rate = 0.10
num_generations = 1000
num_mutations = 4


creatures = []

def makeRandomWeights():
    return {
        'h1': np.random.rand(num_input, n_hidden_1),
        'h2': np.random.rand(n_hidden_1, n_hidden_2),
        'out': np.random.rand(n_hidden_2, num_classes)
        }

def makeInitialWeights():
    return {
        'h1': np.zeros([num_input, n_hidden_1]),
        'h2': np.zeros([n_hidden_1, n_hidden_2]),
        'out': np.zeros([n_hidden_2, num_classes])
        }

def updateRandomly(weights):
    
    new_weights = weights

    num_muts_h1 = randint(0, num_mutations)
    for i in range(num_muts_h1):
        new_weights['h1'][randint(0, num_input - 1)][randint(0, n_hidden_1 - 1)] = 1

    num_muts_h2 = randint(0, num_mutations)
    for i in range(num_muts_h2):
        new_weights['h2'][randint(0, n_hidden_1 - 1)][randint(0, n_hidden_2 - 1)] = 1

    num_muts_out = randint(0, num_mutations)
    for i in range(num_muts_out):
        new_weights['out'][randint(0, n_hidden_2 - 1)][randint(0, num_classes - 1)] = 1
        
    return new_weights

def makeNearWeights(weights):
    return {
        'h1': weights['h1'] + noise(num_input, n_hidden_1),
        'h2': weights['h2'] + noise(n_hidden_1, n_hidden_2),
        'out': weights['out'] + noise(n_hidden_2, num_classes)
        }

def noise(x, y):
    return [ij * learning_rate - (learning_rate / 2) for ij in np.random.rand(x, y)]


for i in range(num_creatures_per_gen):
    #creatures.append(Creature(makeRandomWeights()))
    creatures.append(Creature(updateRandomly(makeInitialWeights())))

# Pre: output layer is of size 4
def executeOutput(output_layer):
    argmax = np.argmax(output_layer)

    if argmax == 1:
        pyautogui.click()

# Create model                                                                                                    
def neural_net(x, creature):
    layer_1 = np.dot(x, creature.weights['h1'])
    layer_2 = np.dot(layer_1, creature.weights['h2'])
    out_layer = np.dot(layer_2, creature.weights['out'])
    
    sum_out = np.sum(out_layer)
    if sum_out > 0:
        out_layer = [output / sum_out for output in out_layer]

    return out_layer

def rankCreatures():

    global creatures
    
    for x in range(num_creatures_per_gen):
        for y in range(num_creatures_per_gen - 1):
            if creatures[y].score < creatures[y + 1].score:
                temp = creatures[y]
                creatures[y] = creatures[y + 1]
                creatures[y+1] = temp

def learn():

    global creatures
    global khakiCount
    global yellowCount
    global distanceFromCenter


    dfc = distanceFromCenter

    for g in range(num_generations):

        print "Generation " + str(g)

        for creature in creatures:

            print "new creature"
            pyautogui.press('space')
            pyautogui.press('space')

            start = int(round(time.time() * 1000))
            while(creature.isAlive()):
           
                #result = neural_net(colorInputHexes, creature)
                
                #start = int(round(time.time() * 1000))
                result = neural_net(bitMap, creature)
                #end = int(round(time.time() * 1000))                
                #print result
                #print "comp time: " + str(end - start)
                executeOutput(result)
             
                #print "kc: " + str(khakiCount)
                #print "yc: " + str(yellowCount)
                if khakiCount > num_khaki:
                    creature.alive = False
                    khakiCount = 0
                    sleep(2)
                    pyautogui.press('space')
                    #pyautogui.press('space')
                
                dfc = distanceFromCenter

                sleep(1.0 / 20.0)
                getInput()
            
            end = int(round(time.time() * 1000))
            creature.score = ((end - start) / 1000.0) - 2 * dfc
            print creature.score
            creature.alive = True




        rankCreatures()

        creatures = creatures[:(num_creatures_per_gen / 4)]

        for creature in creatures:
            print creature.score

        for i in range(num_creatures_per_gen / 4):
            curCreature = creatures[i]
            #creatures.append(Creature(makeNearWeights(curCreature.weights)))
            creatures.append(Creature(updateRandomly(curCreature.weights)))

        for i in range(num_creatures_per_gen / 2):
            creatures.append(Creature(makeRandomWeights()))

        sleep(2)

    best_weights = creatures[0].weights

    h1_file = open('h1.txt', 'w')
    h1_file.write(best_weights['h1'])
    h1_file.close()

    h2_file = open('h2.txt', 'w')
    h2_file.write(best_weights['h2'])
    h2_file.close()

    out_file = open('out.txt', 'w')
    out_file.write(best_weights['out'])
    out_file.close()

    print best_weights['h1']
    print best_weights['h2']
    print best_weights['out']



#thread.start_new_thread(run_pygame, ())
#thread.start_new_thread(learn, ())
learn()
#run_pygame()

while 1:
    pass
