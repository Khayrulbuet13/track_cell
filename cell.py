# Daniel Karkhut
# represents cell object

import math
import scipy.signal
import statistics
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from detectors import Detectors
from tracker import Tracker
from input import Input

class Cell:

    def __init__(self, id, directory):
        self.id = id
        self.xCoordCenter = []
        self.yCoordCenter = []
        self.indexFirstLineisPassed = None #store index that the first tracking line is passed
        self.indexMiddleLineisPassed = None #store index for mid point passing
        self.directory = directory
        self.firstCrop = None
        self.midCrop = None
        self.lastCrop = None


    #Add coords to cell xCoordCenter and yCoordCenter
    def updateValues(self, xCoordCenter, yCoordCenter):
        self.xCoordCenter.append(int(xCoordCenter))
        self.yCoordCenter.append(int(yCoordCenter))

    #Create and save graph for cell
    def generateVelGraph(self):

        #stores amount of movement per frame
        diff = []

        for x in range(len(self.xCoordCenter)-1):
            #find the distance that was traveled using pythagorean theorem
            z = abs( math.sqrt( ((self.xCoordCenter[x+1] - self.xCoordCenter[x])** 2) + ((self.yCoordCenter[x+1] - self.yCoordCenter[x])** 2) ))
            #if data point is too high, replace with average value of array
            if (z > 100):
                diff.append(statistics.median(diff))
                continue
            
            #add to array
            diff.append(z)
        

        #apply smoothing to graph
        #window_length must be equal or less than len(diff) and must be odd
        window_length = len(diff)
        if window_length % 2 == 0: 
            xhat = scipy.signal.savgol_filter(diff, 7, 3) # even, window size 51, polynomial order 3
        else: 
            xhat = scipy.signal.savgol_filter(diff, 7, 3) # odd, window size, polynomial order 3

        #plotting
        plt.title("Speed vs Time") 
        plt.xlabel("Time (frames)") 
        plt.ylabel("Speed (mm)") 
        plt.plot(range(len(self.xCoordCenter)-9), xhat[4:(len(xhat)-4)]) 
        #print(xhat)
    
        path = self.directory + "/VelGraph.png"
        plt.savefig(path)

        plt.clf()

        # generate time-velocity csv file
        a_list = np.array(range(len(self.xCoordCenter)-1))
        #a_list is time. xhat is velocity
        result = []
        temp = []
        count = 0
        for i in (range(len(a_list))):
            for j in range(4):
                if j == 3:
                    temp.append(abs( math.sqrt( ((self.xCoordCenter[i+1] - self.xCoordCenter[i])** 2) + ((self.yCoordCenter[i+1] - self.yCoordCenter[i])** 2) )))
                if j == 2:
                    temp.append(self.yCoordCenter[i])
                if j == 1:
                    temp.append(self.xCoordCenter[i])
                if j == 0:
                    temp.append(a_list[i])
            result.append(temp)
            temp = []
            count += 1
        CSVpath = self.directory + "/t-v.csv"
        
        np.savetxt(CSVpath, result, delimiter=",", fmt = '%1.f')


    def deformationIndex(self, cell_length):
        deformationRatio=(cell_length-4)/(cell_length+4)
        txtPath = self.directory + "/DeformationRatio.txt"
        file = open(txtPath, "w+")
        file.write(str(deformationRatio))
        #file.close()

    def saveImage(self):
        #if the height and length of the image =/= Image Size(cellSize) //// x.shape[0] is height, x.shape[1] is width
        try:
            #save First Crop
            if (  self.firstCrop.shape[0] <  self.firstCrop.shape[1] ):
                blackImage = np.zeros(((self.firstCrop.shape[1] -  self.firstCrop.shape[0]),  self.firstCrop.shape[1], 3), float)
                self.firstCrop = np.concatenate((self.firstCrop, blackImage), axis=0) #first variable over second variable
            path = self.directory + "/Firstcrop.png"
            cv2.imwrite(path, self.firstCrop)

            #save First Crop
            if (  self.midCrop.shape[0] <  self.midCrop.shape[1] ):
                blackImage = np.zeros(((self.midCrop.shape[1] -  self.midCrop.shape[0]),  self.midCrop.shape[1], 3), float)
                self.midCrop = np.concatenate((self.midCrop, blackImage), axis=0) #first variable over second variable
            path = self.directory + "/Midcrop.png"
            cv2.imwrite(path, self.midCrop)

            #save First Crop
            if (  self.lastCrop.shape[0] <  self.lastCrop.shape[1] ):
                blackImage = np.zeros(((self.lastCrop.shape[1] -  self.lastCrop.shape[0]), self.lastCrop.shape[1], 3), float)
                self.lastCrop = np.concatenate((self.lastCrop, blackImage), axis=0) #first variable over second variable
            path = self.directory + "/Lastcrop.png"
            cv2.imwrite(path, self.lastCrop)
        except cv2.error as e:
            print("Cell not saved")
