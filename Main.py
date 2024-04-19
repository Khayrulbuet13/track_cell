# Daniel Karkhut

import cv2
import copy
import os
from detectors import Detectors
from tracker import Tracker
from cell import Cell
from detectors import Detectors
from tracker import Tracker
from cell import Cell



# Input parameters
blur, dilate, cellSize = 5,3,100
BLOB_RADIUS_THRESH = 7
DEBUG = False
cameraFolder = "test"
resultsFolder = "out"
traceStart = 430
traceEnd = 830

detector = Detectors(blur, dilate, BLOB_RADIUS_THRESH, DEBUG)



# Create Object Tracker
tracker = Tracker(100, 2, 5000, 100)


currFrame = 0

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]
images = []
orig_images = []
cell_length = []
# Loop Through Contents of TiffFolder
for f in (sorted(os.listdir(cameraFolder))[::1]):
#starting = 5000 # 0
#for k, f in enumerate(os.listdir( cameraFolder )[starting:] ):
  if f.endswith(".tiff"):
    # Capture frame-by-frame
    frame = cv2.imread(cameraFolder + "/" + f)
    
    print(currFrame)

    if frame is None:
        currFrame = currFrame + 1
        break

    #frame = cv2.flip(frame, 1)
    #frame = cv2.flip(frame, 1)

    images.append(frame)
    
    # Make copy of original frame
    orig_frame = copy.copy(frame)
    orig_images.append(orig_frame)

    # Detect and return centeroids of the objects in the frame
    centers, contours_refined = detector.Detect(frame)
    radius = detector.Radius(frame, traceStart, traceEnd )
    
    cell_length.append(radius)

    # If centroids are detected then track them
    if (len(centers) > 0):

        # Track object using Kalman Filter
        tracker.Update(centers)

        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace)-1):
                    #Draw trace lines and beginning/end lines
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j+1][0][0]
                    y2 = tracker.tracks[i].trace[j+1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 1)
                    cv2.line(frame, (traceStart, 0), (traceStart, 1000), (255, 0, 0), 2)
                    cv2.line(frame, (traceEnd, 0), (traceEnd, 1000), (255, 0, 0), 2)

                    # Here means: tracker.tracks[which cell is trace being drawn for].trace[index of position of cell at specific frame][0 = x, 1 = y][unsure]
                    # if cell trace begins before x = 200 and ends after x = 1000, save the trace and store the cell
                    # used to save cell's movement
                    if ( (tracker.tracks[i].trace[0][0][0] < traceStart) and (tracker.tracks[i].trace[len(tracker.tracks[i].trace)-1][0][0] > traceEnd) and (tracker.tracks[i].tracked == 0) ):
                        #Make directory for new cell
                        cellDirectory = resultsFolder + "/Cell_{}".format(tracker.tracks[i].track_id) 
                        os.makedirs(cellDirectory, exist_ok=True)


                        # create intermediary cell object
                        cell = Cell(tracker.tracks[i].track_id, cellDirectory)

                        # loop to capture full movement of cell
                        for x in range(len(tracker.tracks[i].trace)):
                            cell.updateValues(tracker.tracks[i].trace[x][0][0], tracker.tracks[i].trace[x][1][0])

                            if (cell.indexFirstLineisPassed is None) and (tracker.tracks[i].trace[x][0][0] >= traceStart):
                                cell.indexFirstLineisPassed = x 

                            if (cell.indexMiddleLineisPassed is None) and (tracker.tracks[i].trace[x][0][0] >= ((traceStart + traceEnd)/2)):
                                cell.indexMiddleLineisPassed = x

                        # take last photo 
                        k = currFrame 
                        #photoFrame = cv2.imread(cameraFolder + "/" + str(k).zfill(6) + ".tiff") 
                        photoFrame = orig_images[k]
                        crop = photoFrame[(int(tracker.tracks[i].trace[-1][1][0])  - int(cellSize/2)):(int(tracker.tracks[i].trace[-1][1][0]) + int(cellSize/2)), (int(tracker.tracks[i].trace[-1][0][0]) - int(cellSize/2)):(int(tracker.tracks[i].trace[-1][0][0]) + int(cellSize/2))].copy()
                        cell.lastCrop = crop

                        # take first photo
                        k = currFrame - (len(tracker.tracks[i].trace) + tracker.tracks[i].skipped_frames) + cell.indexFirstLineisPassed       
                        #photoFrame = cv2.imread(cameraFolder + "/" + str(k).zfill(6) + ".tiff") 
                        photoFrame = orig_images[k]
                        crop = photoFrame[(int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][1][0]) - int(cellSize/2)):(int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][1][0]) + int(cellSize/2)), (int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][0][0]) - int(cellSize/2)):(int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][0][0]) + int(cellSize/2))].copy()
                        cell.firstCrop = crop

                        #take middle photo
                        k = currFrame - (len(tracker.tracks[i].trace) + tracker.tracks[i].skipped_frames) + cell.indexMiddleLineisPassed
                        #photoFrame = cv2.imread(cameraFolder + "/" + str(k).zfill(6) + ".tiff") 
                        photoFrame = orig_images[k]
                        crop = photoFrame[(0):(photoFrame.shape[0]), (int(tracker.tracks[i].trace[cell.indexMiddleLineisPassed][0][0]) - int(cellSize)):(int(tracker.tracks[i].trace[cell.indexMiddleLineisPassed][0][0]) + int(cellSize))].copy()
                        cell.midCrop = crop

                        cell.deformationIndex(max(cell_length[(currFrame - (len(tracker.tracks[i].trace) + tracker.tracks[i].skipped_frames) + cell.indexFirstLineisPassed):currFrame]))
                        # remove cell from being tracked again by setting initial position high
                        tracker.tracks[i].tracked = 1

                        cell.generateVelGraph()
                        cell.saveImage()

                        del cell
    
    DEBUG = 1
    if DEBUG:
        # Draw vertical lines at traceStart and traceEnd
        height = frame.shape[0]
        cv2.line(frame, (traceStart, 0), (traceStart, height), (255, 0, 0), 2)  # Red line at traceStart
        cv2.line(frame, (traceEnd, 0), (traceEnd, height), (255, 0, 0), 2)      # Red line at traceEnd


        for contour in contours_refined:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
            
        cv2.imshow('Detected Contours', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    currFrame = currFrame + 1