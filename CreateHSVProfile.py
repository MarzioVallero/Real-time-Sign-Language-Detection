import cv2 
import numpy as np
from IPython.display import clear_output
import tkinter as tk
import ast
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
from PyQt5 import QtGui

def GUIHandle(x): #needed for createTrackbar to work in python.
    pass 

# Setup camera capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Check if a configuration file exists, else load a predefined value set
if os.path.isfile('Config\config.dat'):
    print("Configuration file found\n")
    file = open("Config\config.dat", "r")
    contents = file.read()
    config = ast.literal_eval(contents)
    file.close()
else:
    print ("Configuration file not found\n")
    config = {'HL': 0, 'SL': 29, 'VL': 24, 'HH': 40, 'SH': 255, 'VH': 255}

# Window close handler
def handle_close(event, cap):
    cap.release()

# Setup window
PATH_TO_ICON = os.path.dirname(__file__) + '/Externals/icons/politoIcon.ico'
plt.ion()
imageWin = None
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)
fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))
plt.get_current_fig_manager().window.setWindowIcon(QtGui.QIcon(PATH_TO_ICON))
plt.get_current_fig_manager().set_window_title('HSV Custom Profile Generator')
title_obj = plt.title('HSV Custom Profile Generator')
plt.setp(title_obj, color='#d4d4d4')
fig.patch.set_facecolor('#1e1e1e')
fig.patch.set_edgecolor('#1e1e1e')

# Create the RangeSliders
sliderAxHue = plt.axes([0.20, 0.2, 0.60, 0.03])
sliderHue = RangeSlider(sliderAxHue, "Hue", valmin=0, valmax=255, valstep = 1)
sliderHue.set_val((config["HL"], config["HH"]))
sliderAxSat = plt.axes([0.20, 0.15, 0.60, 0.03])
sliderSat = RangeSlider(sliderAxSat, "Sat", valmin=0, valmax=255, valstep = 1)
sliderSat.set_val((config["SL"], config["SH"]))
sliderAxVal = plt.axes([0.20, 0.1, 0.60, 0.03])
sliderVal = RangeSlider(sliderAxVal, "Val", valmin=0, valmax=255, valstep = 1)
sliderVal.set_val((config["VL"], config["VH"]))

# Calibration loop
while cap.isOpened(): 
    ret, frame = cap.read()
    
    # Convert the frame to HSV and read trackbar inputs
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Read RangeSliders' values
    config["HL"] = int(sliderHue.val[0])
    config["HH"] = int(sliderHue.val[1])
    config["SL"] = int(sliderSat.val[0])
    config["SH"] = int(sliderSat.val[1])
    config["VL"] = int(sliderVal.val[0])
    config["VH"] = int(sliderVal.val[1])
    
    # Perform thresholding according to the specified values
    thresh = cv2.inRange(hsv, (config["HL"], config["SL"], config["VL"]), (config["HH"], config["SH"], config["VH"]))
    
    # Force redraw
    fig.canvas.draw_idle()

    # Show images
    if imageWin is None:
        plt.subplot(1, 2, 1).axis("off")
        imageWin = plt.imshow(frame, "gray")
        title_obj = plt.title("Original Frame")
        plt.setp(title_obj, color='#d4d4d4')
        plt.subplot(1, 2, 2).axis("off")
        threshWin = plt.imshow(thresh, "gray")
        title_obj = plt.title("Thresholded Frame")
        plt.setp(title_obj, color='#d4d4d4')
        plt.show()
    else:
        imageWin.set_data(frame)
        threshWin.set_data(thresh)
        fig.canvas.draw()
        fig.canvas.flush_events() 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        plt.close('all')
        break
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        cap.release()
        file = open("Config\config.dat", "w")
        configToStr = repr(config)
        file.write(configToStr + "\n")
        file.close()
        plt.close('all')
        break