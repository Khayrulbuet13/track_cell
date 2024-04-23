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

    def deformationIndex(self, boxes):
        """
        Calculate deformation index using all width and height values from bounding boxes of each frame.
        
        Args:
            boxes: List of lists, where each inner list contains [x, y, width, height] for each bounding box
                per frame.
        """
        # Path to save the deformation index results
        txt_path = self.directory + "/DeformationIndex.txt"

        # Open the file once and write each frame's deformation index
        with open(txt_path, "w+") as file:
            for frame_boxes in boxes:
                widths = [box[2] for box in frame_boxes if len(box) == 4]  # Collect all widths
                heights = [box[3] for box in frame_boxes if len(box) == 4]  # Collect all heights

                if widths and heights:
                    deformation_indexes = []
                    for w, h in zip(widths, heights):
                        deformation_index = (w - h) / float(w + h) if (w + h) != 0 else 0
                        deformation_indexes.append(deformation_index)

                    # Average the deformation indexes for the frame to have a single measure
                    avg_deformation_index = sum(deformation_indexes) / len(deformation_indexes) if deformation_indexes else 0
                    file.write(f"{avg_deformation_index}\n")
                    print(f"Deformation index for frame saved to {txt_path}: {avg_deformation_index}")
                else:
                    file.write("0\n")  # Default to 0 if no boxes or valid dimensions are found





    # def deformationIndex(self, boxes):
    #     """
    #     Calculate deformation index using the width and height from bounding boxes.
        
    #     Args:
    #         boxes: List of lists, where each inner list contains [x, y, width, height] for each bounding box.
    #     """
    #     widths = []
    #     heights = []

    #     # Iterate through each frame's boxes
    #     for frame_boxes in boxes:
    #         for box in frame_boxes:
    #             if len(box) == 4:  # Ensure the box is correctly formatted
    #                 _, _, w, h = box
    #                 widths.append(w)
    #                 heights.append(h)

    #     if widths and heights:  # Ensure there are dimensions to process
    #         dia_a = max(widths)
    #         dia_b = min(heights)
    #         deformation_index = (dia_a - dia_b) / float(dia_a + dia_b) if (dia_a + dia_b) != 0 else 0
    #     else:
    #         deformation_index = 0  # Default to 0 if no boxes or valid dimensions are found

    #     # Save the calculated deformation index
    #     txt_path = self.directory + "/DeformationIndex.txt"
    #     with open(txt_path, "w+") as file:
    #         file.write(str(deformation_index))
    #         print("Deformation index saved to", txt_path)

    #     return deformation_index  # Optional, if you need to use the value programmatically elsewhere


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
