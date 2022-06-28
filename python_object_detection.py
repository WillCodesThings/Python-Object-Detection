#python_object_detection
from pydoc import classname
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
from math import ceil


img = cv2.VideoCapture(0)
classNames = []
class_names = "coco.names"
with open(class_names, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = "C:\\Users\\Aaron's Laptop\\Desktop\\python object detectiong\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "C:\\Users\\Aaron's Laptop\\Desktop\\python object detectiong\\frozen_inference_graph.pb"

def roundUp(n, d=8):
    d = int('1' + ('0' * d))
    return ceil(n * d) / d

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.8/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    ret,frame = img.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        frame = cv2.rectangle(frame, box, color=(255,255,255), thickness=2)
        cv2.putText(frame, f'{classNames[classId-1]} {roundUp(confidence,2)}', (box[0]+10, box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break