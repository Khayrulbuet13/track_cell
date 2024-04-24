from ij import IJ, ImagePlus, ImageStack
from ij.gui import GenericDialog
from ij.io import DirectoryChooser, FileSaver
from ij.plugin import ImageCalculator, FolderOpener, Duplicator
from ij.plugin.filter import ParticleAnalyzer, Analyzer
from ij.measure import ResultsTable, Measurements
from ij.process import ImageProcessor
import math
import os, shutil
import datetime
from java.text import SimpleDateFormat

# Function to get user inputs including debug mode
def getUserInputs():
    gd = GenericDialog("Settings and Mode")
    gd.addNumericField("Gaussian Blur Sigma:", 2, 1)
    gd.addNumericField("Min Radius of Cell (pixels):", 5, 1)
    gd.addNumericField("Max Radius of Cell (pixels):", 15, 1)
    gd.addNumericField("Min Threshold Value:", 12, 0)
    gd.addNumericField("Max Threshold Value:", 100, 0)
    gd.addNumericField("Batch Size (number of images per batch):", 5000, 0)
    gd.addCheckbox("Debug Mode", False)
    gd.showDialog()
    if gd.wasCanceled():
        print("User canceled the dialog!")
        return None
    else:
        blurSigma = gd.getNextNumber()
        minRadius = gd.getNextNumber()
        maxRadius = gd.getNextNumber()
        minThreshold = gd.getNextNumber()
        maxThreshold = gd.getNextNumber()
        batchSize = int(gd.getNextNumber())
        debug_mode = gd.getNextBoolean()
        return blurSigma, minRadius, maxRadius, minThreshold, maxThreshold, batchSize, debug_mode

# Get user inputs
userInputs = getUserInputs()
if userInputs is None:
    exit()

blurSigma, minRadius, maxRadius, minThreshold, maxThreshold, batchSize, debug_mode = userInputs

# Define the directories
dc = DirectoryChooser("Choose your folder")
input_folder = dc.getDirectory()

# Print where the files are being processed
print("Processing files in:", input_folder)

if input_folder is None:
    print("No input folder selected.")
    exit()

dc = DirectoryChooser("Choose an output directory")
output_folder = dc.getDirectory()
if output_folder is None:
    print("No output folder selected.")
    exit()

destroy = 1

# Constants for Particle Analysis
paOptions = ParticleAnalyzer.SHOW_NONE | ParticleAnalyzer.ADD_TO_MANAGER
paMeasurements = Measurements.AREA
minArea = math.pi * (minRadius ** 2)
maxArea = math.pi * (maxRadius ** 2)
rt = ResultsTable()
pa = ParticleAnalyzer(paOptions, paMeasurements, rt, minArea, maxArea)

file_list = [f for f in os.listdir(input_folder) if not f.startswith('.')]  # Filter out hidden files
total_files = len(file_list)
batch_ctr = 0

for start in range(0, total_files, batchSize):
    start_time = datetime.datetime.now()  # Capture the start time
    end = min(start + batchSize, total_files)
    batch_files = file_list[start:end]
    print("Processing batch from {} to {}".format(start, end - 1))

    for file_name in batch_files:
        image_path = os.path.join(input_folder, file_name)
        imp = IJ.openImage(image_path)
        if imp is None:
            continue
        if debug_mode:
            imp.show()

        IJ.run(imp, "8-bit", "")
        IJ.run(imp, "Find Edges", "")
        IJ.run(imp, "Gaussian Blur...", "sigma=" + str(blurSigma))

        if file_name == batch_files[0]:
            background = Duplicator().run(imp, 1, 1)
            if debug_mode:
                background.setTitle("Background Frame")
                background.show()

        ic = ImageCalculator()
        subtracted = ic.run("Subtract create", imp, background)
        if subtracted:
            ip = subtracted.getProcessor()
            ip.setThreshold(minThreshold, maxThreshold, ImageProcessor.NO_LUT_UPDATE)
            IJ.run(subtracted, "Convert to Mask", "method=Otsu background=Dark calculate black")

            pa.setResultsTable(rt)
            pa.analyze(subtracted)
            
            if rt.getCounter() > 0:
                output_path = os.path.join(output_folder, file_name)
                if destroy:
                    shutil.move(image_path, output_path)
                else:
                    shutil.copy(image_path, output_path)
            elif rt.getCounter() <= 0 and destroy:
                os.remove(image_path)

            rt.reset()

        imp.close()

    end_time = datetime.datetime.now()
    elapsed_time_ms = end_time.getTime() - start_time.getTime()
    minutes = elapsed_time_ms // 60000
    seconds = (elapsed_time_ms % 60000) / 1000.0
    print("Batch {0} processing time: {1} minutes and {2:.2f} seconds".format(batch_ctr, int(minutes), seconds))
    batch_ctr += 1

print("Processing completed")