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


    # def deformationIndex(self, cell_length):
    #     print(cell_length)
    #     deformationRatio=(cell_length-4)/(cell_length+4)
        
    #     txtPath = self.directory + "/DeformationRatio.txt"
    #     file = open(txtPath, "w+")
    #     file.write(str(deformationRatio))
    #     #file.close()


    # def deformationIndex(self, box):
    #     """
    #     Calculate deformation index either using radius or box dimensions.
        
    #     Args:
    #         box: A tuple (x, y, w, h) representing the bounding box.
    #              If provided, calculate deformation index based on box dimensions.
        
    #     Saves the deformation index to a file.
    #     """
    #     box = np.array(box).squeeze()
        
    #     dia_a = box[:, 2].max()  # Max of the third column (index 2)
    #     dia_b = box[:, 3].min()  # Max of the fourth column (index 3)
    #     if (dia_a + dia_b) != 0:
    #         deformation_index = (dia_a - dia_b) / (dia_a + dia_b)
    #     else:
    #         deformation_index = 0  # Handle the case where w+h is 0 to avoid division by zero

    #     # Save the calculated deformation index
    #     txt_path = self.directory + "/DeformationIndex.txt"
    #     with open(txt_path, "w+") as file:
    #         file.write(str(deformation_index))
    # def deformationIndex(self, boxes):
    #     """
    #     Calculate deformation index using the dimensions of bounding boxes.
        
    #     Args:
    #         boxes: List of bounding boxes, where each box is formatted as [x, y, w, h].
    #     """
    #     if not boxes:
    #         print("No boxes provided")
    #         return

    #     # Convert list of boxes to a numpy array for easier manipulation
    #     box_array = np.array(boxes)

    #     # Calculate the max width and min height across all boxes
    #     dia_a = box_array[:, 2].max()  # Max width
    #     dia_b = box_array[:, 3].min()  # Min height
        
    #     if (dia_a + dia_b) != 0:
    #         deformation_index = (dia_a - dia_b) / (dia_a + dia_b)
    #     else:
    #         deformation_index = 0  # Avoid division by zero

    #     # Save the calculated deformation index
    #     txt_path = self.directory + "/DeformationIndex.txt"
    #     with open(txt_path, "w+") as file:
    #         file.write(str(deformation_index))

    def deformationIndex(self, boxes):
        """
        Calculate deformation index using the width and height from bounding boxes.
        
        Args:
            boxes: List of lists, where each inner list contains [x, y, width, height] for each bounding box.
        """
        widths = []
        heights = []

        # Iterate through each frame's boxes
        for frame_boxes in boxes:
            for box in frame_boxes:
                if len(box) == 4:  # Ensure the box is correctly formatted
                    _, _, w, h = box
                    widths.append(w)
                    heights.append(h)

        if widths and heights:  # Ensure there are dimensions to process
            dia_a = max(widths)
            dia_b = min(heights)
            deformation_index = (dia_a - dia_b) / float(dia_a + dia_b) if (dia_a + dia_b) != 0 else 0
        else:
            deformation_index = 0  # Default to 0 if no boxes or valid dimensions are found

        # Save the calculated deformation index
        txt_path = self.directory + "/DeformationIndex.txt"
        with open(txt_path, "w+") as file:
            file.write(str(deformation_index))
            print("Deformation index saved to", txt_path)

        return deformation_index  # Optional, if you need to use the value programmatically elsewhere



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
