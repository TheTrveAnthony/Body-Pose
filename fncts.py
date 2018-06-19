import cv2
import numpy as np



def skeleton(frame, n):
	""" This function will recognize the pose of a person and draw it, works with one person only,
		A caffee deep learning is used to perform it. Frame is the current frame, n is the caffee network model. """
	

	# In this model there are 18 points of interest, then we gotta define the connections between each point
	nPoints = 18
	POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

	# These are the input image dimensions
	inWidth = 368
	inHeight = 368

	threshold = 0.1

	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]

	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False) # the image has to be converted to a blob for the network
	n.setInput(inpBlob)
	output = n.forward()	# returns an Array containing the predictions of the model

	H = output.shape[2]
	W = output.shape[3]
    # Empty list to store the detected keypoints
	points = []

	for i in range(nPoints):
        # confidence map of corresponding body's part.
	    probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
	    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
	    x = (frameWidth * point[0]) / W
	    y = (frameHeight * point[1]) / H

	    if prob > threshold : # The threshold is useful to avoid false detections

	        points.append((int(x), int(y)))
	    else :
	        points.append(None)

    # Draw Skeleton
	for pair in POSE_PAIRS:
	    partA = pair[0]
	    partB = pair[1]

	    if points[partA] and points[partB]:
	        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
	        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
	        cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)