import cv2 
import numpy as np
from IPython.display import clear_output
import tkinter as tk
import ast
import os

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

# Create the Instructions window
cv2.namedWindow('Instructions', flags=cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Instructions', 400, height-320)
cv2.moveWindow('Instructions', int(0.5*screen_width)-int(0.5*400), int(0.5*screen_height)-int(0.5*400)+320)
instr = np.zeros((height-320, 400))
cv2.putText(instr, 'Use E to confirm', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, cv2.LINE_AA)
cv2.putText(instr, 'Use Q to exit', (90, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2, cv2.LINE_AA)
cv2.imshow('Instructions', instr)

# Create the Camera window
cv2.namedWindow('Camera', flags=cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Camera', int(0.5*screen_width)-int(0.5*400)-width, int(0.5*screen_height)-int(0.5*400))

# Create the Thresholded image output window
cv2.namedWindow('Threshold', flags=cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Threshold', int(0.5*screen_width)+int(0.5*400), int(0.5*screen_height)-int(0.5*400))

# Create the Trackbar needed to perform manual configuration
cv2.namedWindow('Trackbar', flags=cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Trackbar', 400, 320)
cv2.moveWindow('Trackbar', int(0.5*screen_width)-int(0.5*400), int(0.5*screen_height)-int(0.5*400))
cv2.createTrackbar('Hue Low', 'Trackbar', config["HL"], 255, GUIHandle)
cv2.createTrackbar('Sat Low', 'Trackbar', config["SL"], 255, GUIHandle)
cv2.createTrackbar('Val Low', 'Trackbar', config["VL"], 255, GUIHandle)
cv2.createTrackbar('Hue High', 'Trackbar', config["HH"], 255, GUIHandle)
cv2.createTrackbar('Sat High', 'Trackbar', config["SH"], 255, GUIHandle)
cv2.createTrackbar('Val High', 'Trackbar', config["VH"], 255, GUIHandle)

# Calibration loop
while cap.isOpened(): 
    ret, frame = cap.read()
    
    # Convert the frame to HSV and read trackbar inputs
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    config["HL"] = cv2.getTrackbarPos('Hue Low', 'Trackbar')
    config["SL"] = cv2.getTrackbarPos('Sat Low', 'Trackbar')
    config["VL"] = cv2.getTrackbarPos('Val Low', 'Trackbar')
    config["HH"] = cv2.getTrackbarPos('Hue High', 'Trackbar')
    config["SH"] = cv2.getTrackbarPos('Sat High', 'Trackbar')
    config["VH"] = cv2.getTrackbarPos('Val High', 'Trackbar')
    
    # Lock sliders so that they cannot invert
    if (config["HL"] > config["HH"]):
        cv2.setTrackbarPos('Hue Low', 'Trackbar', config["HH"])
    if (config["SL"] > config["SH"]):
        cv2.setTrackbarPos('Sat Low', 'Trackbar', config["SH"])
    if (config["VL"] > config["VH"]):
        cv2.setTrackbarPos('Val Low', 'Trackbar', config["VH"])
    
    # Perform thresholding according to the specified values
    thresh = cv2.inRange(hsv, (config["HL"], config["SL"], config["VL"]), (config["HH"], config["SH"], config["VH"]))
    
    # Show images
    cv2.imshow('Camera', frame)
    cv2.imshow('Threshold', thresh)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        file = open("Config\config.dat", "w")
        configToStr = repr(config)
        file.write(configToStr + "\n")
        file.close()
        cv2.destroyAllWindows()
        break