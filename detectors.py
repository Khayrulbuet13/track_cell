'''
    File name         : detectors.py
    File Description  : Detect objects in video frame
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2

# set to 1 for pipeline images
debug = 1


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self, blurFactor, dilateFactor, blob_radius_thresh=7, debug=False):
        """Initialize variables used by Detectors class
        Args:
            blurFactor: Degree of Gaussian Blur
            dilateFactor: Degree of contour dilation
            analysis: 0 to calculate deformation ratio, 1 to turn off
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.blurFactor = blurFactor
        self.dilateFactor = dilateFactor
        self.blob_radius_thresh = blob_radius_thresh
        self.debug = debug

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.debug:
            cv2.imshow('Gray Scale', gray)
            
        gray_blurred = cv2.GaussianBlur(gray, (self.blurFactor, self.blurFactor), 0)
        if self.debug:
            cv2.imshow('Blurred Edges', gray_blurred)
        
        mask = self.fgbg.apply(gray_blurred)
        if self.debug:
            cv2.imshow('backgroundmask', mask)
        

        _, thresh = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
        if self.debug:
            cv2.imshow('Threshold Image', thresh)
        
        dilated = cv2.dilate(thresh, None, iterations=self.dilateFactor)
        if self.debug:
            cv2.imshow('Dilated Image', dilated)
        
        erode = cv2.erode(dilated, np.ones((6, 6), np.uint8) , cv2.BORDER_REFLECT)  
        if self.debug:
            cv2.imshow('erode Image', erode)
            
        
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        centers = []
        contours_refined = []
        cell_boxes = []
        trash_area = np.pi * (self.blob_radius_thresh ** 2)
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if cv2.contourArea(contour) > trash_area:
                    centers.append(np.array([[cx], [cy]]))
                    contours_refined.append(contour)
                    # x_x, y_y, w_w, h_h = cv2.boundingRect(contour)
                    # cell_boxes.append([x_x, y_y, w_w, h_h])
                    x, y, w, h = cv2.boundingRect(contour)
                    cell_boxes.append([x, y, w, h])

                    if self.debug:  # Optionally display each contour
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
        
        if self.debug:
            cv2.imshow('Final Detection', frame)
            cv2.waitKey(100)  # Wait for a key press to move to the next frame


        return centers, contours_refined,  cell_boxes if cell_boxes else []


    def Radius(self, frame, traceStart, traceEnd):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)
        fgmask=cv2.GaussianBlur(fgmask, (self.blurFactor, self.blurFactor), 0)
        fgmask = cv2.dilate(fgmask, None, iterations=self.dilateFactor) # when we apply the blur, details get lost. So, the cell detail is getting lost, losing a well-defined contour of the cell so we are padding it (adding pixel value)
        fgmask=cv2.threshold(fgmask, 1, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        radius_modified = []
        for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if (int(((traceEnd-traceStart)/2)+traceStart)-40) < x < (int(((traceEnd-traceStart)/2)+traceStart)+40):
                    radius_modified.append(radius)
                else:
                    radius_modified.append(5)
                
        if radius_modified:
            radius_max=max(radius_modified)
        else:
            radius_max=1000
                    
        
        return radius_max