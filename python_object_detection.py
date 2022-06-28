# python_object_detection
from pydoc import classname
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
from math import ceil
import numpy as np

thres = 0.45
nms_threshold = 0.5

img = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
img.set(3, 1920)
img.set(4, 1080)
img.set(10, 150)

classNames = []
class_names = "C:\\Users\\PCuser\\Desktop\\pythonobjectdetection\\coco.names"
with open(class_names, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


configPath = "C:\\Users\\PCuser\\Desktop\\pythonobjectdetection\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = (
    "C:\\Users\\PCuser\\Desktop\\pythonobjectdetection\\frozen_inference_graph.pb"
)


def roundUp(n, d=8):
    d = int("1" + ("0" * d))
    return ceil(n * d) / d


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.8 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    ret, frame = img.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold=nms_threshold)

    for i in indices:
        try:
            box = bbox[1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(
                frame, (x, y), (x + w, h + y), color=(255, 255, 255), thickness=2
            )
            cv2.putText(
                frame,
                f"{classNames[classIds[i]-1]} {roundUp(confs[1],2)}",
                (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        except IndexError:
            pass

    # if len(classIds) != 0:
    #     for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

    #         frame = cv2.rectangle(frame, box, color=(255, 255, 255), thickness=2)
    #

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
