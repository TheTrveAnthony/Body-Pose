import cv2
import numpy as np
from fncts import *




protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"

cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

while cv2.waitKey(1) < 0:
   
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        cv2.waitKey()
        print("Camera failed to start")
        break

    skeleton(frame, net)

    cv2.imshow('Ginyu Tokusentai !!!', frame)
