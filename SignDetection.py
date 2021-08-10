# Import all the required libraries
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 
import numpy as np
from IPython.display import clear_output
import tkinter as tk
import ast

# Computes the detections from the predictive model
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Utility definitions
WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

# Load the pipeline.config file and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore the specified checkpoint (it must match an existing model)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-41')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

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

# Camera loop
while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    # Skin tone segmentation
    # The frame is converted to HSV, then thresholded according to the Hue value
    # according to the paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf
    HSV_Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    totalMask = cv2.inRange(HSV_Frame, (config["HL"], config["SL"], config["VL"]), (config["HH"], config["SH"], config["VH"]))
    totalMask = totalMask.astype(np.uint8)
    
    # Face removal, in order to give less room for error to the gesture classifier
    # A Haar classifier detects the face, then adds its filled bounding box to the mask
    haar_face = cv2.CascadeClassifier()
    haar_face.load(cv2.samples.findFile("HaarClassifiers/HaarFrontalFaceAlt.xml"))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    faces = haar_face.detectMultiScale(gray_frame, minSize=(int(0.2*height), int(0.2*height)))
    for (x, y, w, h) in faces:
        vertices = np.array([[x,y-int(0.3*h)], [x+w, y-int(0.3*h)], [x+w, y+h], [x, y+h]])
        cv2.fillPoly(totalMask, pts = [vertices], color =(0,0,0))
    
    # The mask finally undergoes the Opening operator in order to remove pepper noise,
    # then gets applied as a bitwise operator to the frame
    totalMask = cv2.morphologyEx(totalMask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
    output = cv2.bitwise_and(frame, frame, mask = totalMask)
        
    # The masked image is then converted to a tensor for object detection
    input_tensor = tf.convert_to_tensor(np.expand_dims(output, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_qdetections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    # The bounding boxes of all detected gestures are drawn on top of the original frame
    # with their corresponding label.
    # max_boxes_to_draw=1 doesn't let two overlapping gestures to be recognized at once
    # min_score_thresh=.7 ignores all detections with an accuracy rate lower than 70%
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.7,
                agnostic_mode=False)

    # The output is displayed on an interactive window
    cv2.imshow('object detection',  image_np)
    #cv2.imshow('Masked image', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break