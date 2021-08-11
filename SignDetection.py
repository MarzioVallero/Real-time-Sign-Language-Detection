# Import all the required libraries
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from Externals.service.head_pose import HeadPoseEstimator
from Externals.service.face_alignment import CoordinateAlignmentModel
from Externals.service.face_detector import MxnetDetectionModel
from Externals.service.iris_localization import IrisLocalizationModel
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread
import sys
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

def calculate_3d_gaze(frame, poi, scale=256):
    starts, ends, pupils, centers = poi

    eye_length = norm(starts - ends, axis=1)
    ic_distance = norm(pupils - centers, axis=1)
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

    delta = np.array((delta_x * SIN_LEFT_THETA,
                      delta_y * SIN_UP_THETA))
    delta /= eye_length
    theta, pha = np.arcsin(delta)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
    # delta[0, abs(theta) < 0.1] = 0
    # delta[1, abs(pha) < 0.03] = 0

    inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    # cv2.circle(frame, tuple(pupil.astype(int)), 2, (0, 255, 255), -1)
    # cv2.circle(frame, tuple(center.astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T

def draw_sticker(src, offset, pupils, landmarks,
                 blink_thd=0.22,
                 arrow_color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()

    left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
    left_eye_width = landmarks[39, 0] - landmarks[35, 0]

    right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
    right_eye_width = landmarks[93, 0] - landmarks[89, 0]

    for mark in landmarks.reshape(-1, 2).astype(int):
        cv2.circle(src, tuple(mark), radius=1,
                   color=(0, 0, 255), thickness=-1)

    if left_eye_hight / left_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                        tuple((offset+pupils[0]).astype(int)), arrow_color, 2)

    if right_eye_hight / right_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                        tuple((offset+pupils[1]).astype(int)), arrow_color, 2)

    return src


args = sys.argv[1:]
if(len(args) == 0):
    debug = False
elif(args[0] == "--Debug"):
    debug = True
else:
    debug = False

# Utility definitions
WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)

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

fd = MxnetDetectionModel("Externals/weights/16and32", 0, .6, gpu=-1)
fa = CoordinateAlignmentModel('Externals/weights/2d106det', 0, gpu=-1)
gs = IrisLocalizationModel("Externals/weights/iris_landmark.tflite")
hp = HeadPoseEstimator("Externals/weights/object_points.npy", cap.get(3), cap.get(4))

# Check if a configuration file exists, else load a predefined value set
if os.path.isfile('Config\config.dat'):
    print("Configuration file found")
    file = open("Config\config.dat", "r")
    contents = file.read()
    config = ast.literal_eval(contents)
    file.close()
else:
    print ("Configuration file not found. Run CreateHSVProfile.py to create a local profile")
    config = {'HL': 0, 'SL': 29, 'VL': 24, 'HH': 40, 'SH': 255, 'VH': 255}

# Camera loop
while cap.isOpened(): 
    ret, frame = cap.read()

    looking = 0
    bboxes = fd.detect(frame)

    for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
        # calculate head pose
        _, euler_angle = hp.get_head_pose(landmarks)
        pitch, yaw, roll = euler_angle[:, 0]

        eye_markers = np.take(landmarks, fa.eye_bound, axis=0)
        
        eye_centers = np.average(eye_markers, axis=1)
        # eye_centers = landmarks[[34, 88]]
        
        # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
        eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

        iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
        pupil_left, radius_left = gs.draw_pupil(iris_left, frame, thickness=0)

        iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
        pupil_right, radius_right = gs.draw_pupil(iris_right, frame, thickness=0)

        pupils = np.array([pupil_left, pupil_right])

        poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
        theta, pha, delta = calculate_3d_gaze(frame, poi)

        if yaw > 30:
            end_mean = delta[0]
        elif yaw < -30:
            end_mean = delta[1]
        else:
            end_mean = np.average(delta, axis=0)

        if end_mean[0] < 0:
            zeta = arctan(end_mean[1] / end_mean[0]) + pi
        else:
            zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

        # print(zeta * 180 / pi)
        # print(zeta)
        if roll < 0:
            roll += 180
        else:
            roll -= 180

        real_angle = zeta + roll * pi / 180
        # real_angle = zeta

        # print("end mean:", end_mean)
        # print(roll, real_angle * 180 / pi)

        R = norm(end_mean)
        offset = R * cos(real_angle), R * sin(real_angle)

        landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

        if(debug):
            gs.draw_eye_markers(eye_markers, frame, thickness=1)
            draw_sticker(frame, offset, pupils, landmarks)
            cv2.circle(frame, tuple(pupil_right), radius_right, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(pupil_left), radius_left, (0, 255, 255), 2, cv2.LINE_AA)

        if (R > 45):
            print("You aren't looking!")
            looking = 5
        elif (looking > 0):
            looking -= 1

    image_np = np.array(frame)
    
    if (looking == 0):
        # Skin tone segmentation
        # The frame is converted to HSV, then thresholded according to the Hue value
        # according to the paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf
        HSV_Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        totalMask = cv2.inRange(HSV_Frame, (config["HL"], config["SL"], config["VL"]), (config["HH"], config["SH"], config["VH"]))
        totalMask = totalMask.astype(np.uint8)
        
        # Face removal, in order to give less room for error to the gesture classifier
        # A Haar classifier detects the face, then adds its filled bounding box to the mask
        haar_face = cv2.CascadeClassifier()
        haar_face.load(cv2.samples.findFile("Externals/HaarFrontalFaceAlt.xml"))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        faces = haar_face.detectMultiScale(gray_frame, minSize=(int(0.2*height), int(0.2*height)))
        for (x, y, w, h) in faces:
            vertices = np.array([[x,y-int(0.3*h)], [x+w, y-int(0.3*h)], [x+w, y+h], [x, y+h]])
            cv2.fillPoly(totalMask, pts = [vertices], color =(0,0,0))
            if(debug):
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
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
