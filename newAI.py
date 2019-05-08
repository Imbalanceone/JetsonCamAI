######## Video Object Detection Using Tensorflow-trained Classifier #########
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.
## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py
## but I changed it to make it more understandable to me.
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import config as CF
import socket
import massivefunc as mf
import time
#import PiCam as PC
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#import simple_camera
#simple_camera.show_camera()
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = CF.VIDEO_NAME
# for drawing on video finded elements
Drawing = CF.Drawing
# Grab path to current working directory
CWD_PATH = os.getcwd()#"C:\\tensorflow1\models\\research\object_detection"#"/home/rl/tensorflow1/models/research/object_detection" #os.getcwd()
if CF.Debug > 0:
	print("CWD_PATH = ", CWD_PATH)
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
if CF.Debug > 0:
	print("PATH_TO_CKPT = ", PATH_TO_CKPT)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
if CF.Debug > 0:
	print("PATH_TO_LABELS = ", PATH_TO_LABELS)
# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)
if CF.Debug > 0:
	print("PATH_TO_VIDEO = ", PATH_TO_VIDEO)

# Number of classes the object detector can identify
NUM_CLASSES = CF.NUM_CLASSES
#ip to socked server
SockIP = CF.SockIP
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

	sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def gstreamer_pipeline (capture_width=3820, capture_height=2464, display_width=1910, display_height=1232, framerate=2, flip_method=2) :
	return ('nvarguscamerasrc ! '
			'video/x-raw(memory:NVMM), '
			'width=(int)%d, height=(int)%d, '
			'format=(string)NV12, framerate=(fraction)%d/1 ! '
			'nvvidconv flip-method=%d ! '
			'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
			'videoconvert ! '
			'video/x-raw, format=(string)BGR ! appsink'  % (capture_width, capture_height, framerate, flip_method, display_width, display_height))

pipe='udpsrc port="5000" caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)RGB, depth=(string)8, width=(string)1200, height=(string)900" ! rtpvrawdepay ! videoconvert ! appsink'
# Open video file
if CF.Source == "Video":
	try:
		video = cv2.VideoCapture(PATH_TO_VIDEO)
	except:
		print("Fail load Video")
		sys.exit()
	print(gstreamer_pipeline(flip_method=0))
elif CF.Source == "Camera":
	try:
		video = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
	except:
		print("Fail load Camera")
		sys.exit()

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)# float
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)# float
if CF.Debug > 0:
	print(width, height)

sock = socket.socket()
sock.settimeout(2)
con = 0

def Draw(nboxes, nscores, nclasses):
	font = cv2.FONT_HERSHEY_SIMPLEX
	if (nboxes.any()):
		j = -1
		for i in nboxes:
			j += 1
			ymin, ymax, xmin, xmax = mf.GetCoordinates(i)
			XMid = (xmin+xmax)//2
			YMid = (ymin+ymax)//2
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=3)
			cv2.circle(frame, (XMid, YMid), 3, color=(0, 0, 255), thickness=-1)
			cv2.putText(frame, str(nclasses[j])+" "+str(int(nscores[j]*100))+"%", (xmin, ymin+25), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
			cv2.putText(frame, str(XMid)+" "+str(YMid), (XMid-60, YMid-25), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
			if CF.Debug > 0:
				print("Draw box with center as ", XMid, YMid)

try:
	sock.connect((SockIP, 9090))
	con = 1
except:
	print("connect fail")
T = time.time()
if CF.Debug > 0:
	print("Start time = ", T)
for i in range (20):
	ret, frame = video.read()
	print('frame number:', i)

if Drawing == True:
	window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
while video.isOpened():
	if (con == 0) and (time.time() - T >= 100.):
		T = time.time()
		print("try reconnect")
		try:
			sock.connect((SockIP, 9090))
			con = 1
		except:
			print("reconnect fail")
	# Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
	# i.e. a single-column array, where each item in the column has the pixel RGB value
	if CF.Debug > 0:
		print("start read frame")
	ret, frame = video.read()
	if CF.Debug > 0:
		print("successful readed")
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if CF.Debug > 0:
		print("frame")
	frame_expanded = np.expand_dims(frame, axis=0)

	if CF.Debug > 0:
		print("frame_expanded")
	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})
	if CF.Debug > 0:
		print("sess.run")
	for i in range (20):
		ret, frame = video.read()
		print('frame number:', i)
	# Draw the results of the detection (aka 'visulaize the results')
	#vis_util.visualize_boxes_and_labels_on_image_array(
	#frame,
	#np.squeeze(boxes),
	#np.squeeze(classes).astype(np.int32),
	#np.squeeze(scores),
	#category_index,
	#use_normalized_coordinates=True,
	#line_thickness=4,
	#min_score_thresh=0.80)
	# All the results have been drawn on the frame, so it's time to display it.
	JsonData = mf.GetJson(boxes, scores, classes)
	JsonData += "|"
	if CF.Debug > 0:
		print(JsonData)
	if (con == 1):
		try:
			sock.send(JsonData.encode())
		except:
			con = 0
	if Drawing == True:
		nboxes, nscores, nclasses = mf.CheckPredict(boxes, scores, classes, height, width)
		Draw(nboxes, nscores, nclasses)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		cv2.imshow('CSI camera', frame)
		if cv2.waitKey(30) == ord('q'):
			break
# Clean up
video.release()
cv2.destroyAllWindows()
