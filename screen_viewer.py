#
# Author: Paul Baird-Smith 2017
#
# Screen viewer for a given portion of the monitor. Reduces the screen from
# a massive space (1400 x 900 x 255^3) to a smaller subspace (~1024^3) for
# input into a neural network. For this specific application, picks up notable
# colors including green, kahki, crimson, and goldenrod (yellow). Colors are
# used to track events and positions in the classic mobile game Flappy Bird.
#
# By reducing the size of the screen we are looking at, a loop in getInput()
# usually runs in <20ms.
#
#


import itertools
import numpy as np
import PIL.ImageGrab
import webcolors
import pyautogui
import pygame
import sys
import time
import threading

import struct
import Quartz.CoreGraphics as CG


class ScreenViewer:

    
    #------------ Parameters that hold for all instances of ScreenViewer ---------------#


    # Constants for pygame window size
    WINDOW_WIDTH  = 200
    WINDOW_HEIGHT = 150
    WIN_OFFSET    = 10

    # Holds constants for maximal distances from each color. Used to determine
    # if a given pixel is close enough in color to be categorized as that color.
    MAX_COLOR_DISTANCE_GREEN   = 210
    MAX_COLOR_DISTANCE_KHAKI   = 40
    MAX_COLOR_DISTANCE_YELLOW  = 100
    MAX_COLOR_DISTANCE_CRIMSON = 100
    
    # Values for the RGB tuples (roughly) representing each color
    greenRGB   = (0, 255, 0, 255)
    khakiRGB   = (240, 230, 140, 255)
    crimsonRGB = (220, 20, 60, 255)
    yellowRGB  = (218, 165, 32, 255)

    # The number of khaki cells observed, above which we can be fairly
    # confident we are looking at the "restart" menu in Flappy Bird
    num_khaki = 30



    #-----------------------------------------------------------------------------------#


    # Instantiate ScreenViewer object
    def __init__(self, screen_def, num_neurons):

        print "Imports for ScreenViewer complete."

        self.screen_x = self.screen_y = self.screen_width = self.screen_height = 0

        # Keep track of positional variables for the pyGame screenviewer
        # Hold position and size of portion of screen to view
        self.screen_x      = screen_def[0]
        self.screen_y      = screen_def[1]
        self.screen_width  = screen_def[2]
        self.screen_height = screen_def[3]
        self.region_width = self.region_height = 0



        # Thread object that we lock when we want to change shared state between screen
        # viewer and neural network
        self.c = threading.Condition()


        # We define a rectangular lattice of neurons, spread out evenly over our screen
        # segment. This number defines the resolution of our input to the neural network
        self.num_neurons = num_neurons # Pick a square number here
        self.sqrt_num_neurons = int(num_neurons ** 0.5)

        # Dimensions for a cell in the pygame window. Each neuron corresponds to a single cell
        self.cell_width = (ScreenViewer.WINDOW_WIDTH - (2 * ScreenViewer.WIN_OFFSET)) / self.sqrt_num_neurons
        self.cell_height = (ScreenViewer.WINDOW_HEIGHT - (2 * ScreenViewer.WIN_OFFSET)) / self.sqrt_num_neurons

        # Holds the RGB tuple value for all pixels in our screen grab at any point
        self.colorRGBs = None
        self.data = None

        # bitMap is an array that represents a simplified view of our screen grab.
        # Reduced to a 1 (and changed to a black pixel) for pixels close to green.
        # Reduced to a 100 (and changed to a red pixel) if we suspect the pixel belongs
        #     to Flappy.
        # Reduced to a 0 (and changed to a white pixel) for pixels not near green.
        self.bitMap = np.zeros(self.num_neurons)

        # Hold the count for khaki and yellow pixels
        self.khakiCount  = 0
        self.yellowCount = 0

        # Holds distance from the center line when Flappy dies
        self.bird_x = self.bird_y = 0

        # Spread our neurons in a rectangular lattice over the desired space 
        pos_x = [(i * (self.screen_width - 1)) / (self.sqrt_num_neurons - 1) for i in range(self.sqrt_num_neurons)]
        pos_y = [(i * (self.screen_height - 1)) / (self.sqrt_num_neurons - 1) for i in range(self.sqrt_num_neurons)]
        self.positions = list(itertools.product(pos_x, pos_y))

        print "Neuron positions initialized."

        # Initialize the pygame window
        pygame.init()

        # Define size and mode for our pygame window
        self.screen = pygame.display.set_mode((ScreenViewer.WINDOW_WIDTH, ScreenViewer.WINDOW_HEIGHT), pygame.RESIZABLE)

        print "Pygame successfully initialized."




    # Returns the array of RGB tuples in the given portion of the screen.
    # Credit: https://stackoverflow.com/questions/12978846/python-get-screen-pixel-value-in-os-x
    def getColors(self):

        start = int(round(time.time() * 1000))
        region = CG.CGRectMake(self.screen_x, self.screen_y, self.screen_width, self.screen_height)

        # Create screenshot as CGImage
        image = CG.CGWindowListCreateImage(
            region,
            CG.kCGWindowListOptionOnScreenOnly,
            CG.kCGNullWindowID,
            CG.kCGWindowImageDefault)

        self.region_width = CG.CGImageGetWidth(image)
        self.region_height = CG.CGImageGetHeight(image)

        # Intermediate step, get pixel data as CGDataProvider
        prov = CG.CGImageGetDataProvider(image)

        # Copy data out of CGDataProvider, becomes string of bytes
        self.data = CG.CGDataProviderCopyData(prov)
        #print len(self.data)


        end = int(round(time.time() * 1000))
        #print "Time to get input: " + str(end - start)
        return [self.pixel(pos[0], pos[1]) for pos in self.positions]
        


    def pixel(self, x, y):
        """Get pixel value at given (x,y) screen coordinates

        Must call capture first.
        """

        # Pixel data is unsigned char (8bit unsigned integer),
        # and there are four (blue,green,red,alpha)
        data_format = "BBBB"


        #self.screen_width = len(self.data) / (self.sqrt_num_neurons * self.screen_height)
        #print self.screen_width

        # Calculate offset, based on
        # http://www.markj.net/iphone-uiimage-pixel-color/
        # REALLY WEIRD +10 OFFSET. NO IDEA WHY
        offset = 4 * (((len(self.data) / (self.screen_height * 4))*int(round(y))) + int(round(x)))

        # Unpack data from string into Python'y integers
        b, g, r, a = struct.unpack_from(data_format, self.data, offset=offset)

        # Return BGRA as RGBA
        return (r, g, b, a)



        #return [PIL.ImageGrab.grab(bbox=(pos[0], pos[1], pos[0] + 1, pos[1] + 1)).load()[0, 0] for pos in self.positions]




    #---------------- Run the pyGame screen viewer -------------------------------#




    # Only call getInput after initializing pygame
    def getInput(self):

        start = int(round(time.time() * 1000))
        #self.c.acquire()
        self.colorRGBs = self.getColors()
        end = int(round(time.time() * 1000))
        self.khakiCount = 0
        greenCount = 0
        self.yellowCount = 0
        #self.c.release()
        
        end2 = int(round(time.time() * 1000))
    
        for x in range(self.sqrt_num_neurons):
            for y in range(self.sqrt_num_neurons):

                curPixel = self.colorRGBs[x * self.sqrt_num_neurons + y]
                
                distance_from_green   = sum([(curPixel[i] - ScreenViewer.greenRGB[i])   ** 2 for i in range(4)]) ** 0.5
                distance_from_khaki   = sum([(curPixel[i] - ScreenViewer.khakiRGB[i])   ** 2 for i in range(4)]) ** 0.5
                distance_from_crimson = sum([(curPixel[i] - ScreenViewer.crimsonRGB[i]) ** 2 for i in range(4)]) ** 0.5
                distance_from_yellow  = sum([(curPixel[i] - ScreenViewer.yellowRGB[i])  ** 2 for i in range(4)]) ** 0.5


                if (int(distance_from_green) < int(ScreenViewer.MAX_COLOR_DISTANCE_GREEN)):
                    greenCount = greenCount + 1
                    curPixel = (0, 0, 0, 255)
                    #self.c.acquire()
                    self.bitMap[x * self.sqrt_num_neurons + y] = 1
                    #self.c.release()
                elif (int(distance_from_yellow) < int(ScreenViewer.MAX_COLOR_DISTANCE_YELLOW) or distance_from_crimson < int(ScreenViewer.MAX_COLOR_DISTANCE_CRIMSON)):
                    #print closest_color(curPixel)
                    curPixel = (255, 0, 0, 255)
                    self.yellowCount = self.yellowCount + 1

                    self.bird_x = x
                    self.bird_y = y

                    #self.c.acquire()
                    self.bitMap[x * self.sqrt_num_neurons + y] = 100
                    self.distanceFromCenter = (y - self.sqrt_num_neurons) ** 2
                    #self.c.release()
                else:
                    curPixel = (255, 255, 255, 255)
                    #self.c.acquire()
                    self.bitMap[x * self.sqrt_num_neurons + y] = 0
                    #self.c.release()


                if distance_from_khaki < ScreenViewer.MAX_COLOR_DISTANCE_KHAKI:
                    self.khakiCount = self.khakiCount + 1

                rectCoords = (ScreenViewer.WIN_OFFSET + (x * self.cell_width), ScreenViewer.WIN_OFFSET + (y * self.cell_height), self.cell_width, self.cell_height)
                pygame.draw.rect(self.screen, curPixel, rectCoords, 0)

        pygame.display.update()
        end3 = int(round(time.time() * 1000))


        if not self.yellowCount < 1:
            self.distanceFromCenter = 50

        end3 = int(round(time.time() * 1000))
        
        return self.bitMap



screen_def = (600, 135, 520, 690)
sv = ScreenViewer(screen_def, 900)

# Continuously get input. Function is slow enough to not require sleep between calls
while(True):
    sv.getInput()




