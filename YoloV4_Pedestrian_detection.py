''' Source: https://data-flair.training/blogs/pedestrian-detection-python-opencv/'''

import numpy as np
import cv2
import os
import imutils


NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2



def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results


if __name__ == "__main__":
    
	labelsPath = "yolov4_dependencies/coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	weights_path = "yolov4_dependencies/yolov4-tiny.weights"
	config_path = "yolov4_dependencies/yolov4-tiny.cfg"

	model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
	'''
	model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	'''

	layer_name = model.getLayerNames()
	layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
	video_capture = cv2.VideoCapture("videos/walking_pedestrians1.mp4") # Open video capture object
	got_image, bgr_img = video_capture.read() # Make sure we can read video

	if not got_image:
		print("Cannot read video source")
		sys.exit()
	bgr_img_copy = bgr_img.copy()

	# Start window thread
	cv2.startWindowThread()                           
	windowName = "FinalProject"
	cv2.namedWindow(windowName)
	windowIsOpen = True

	while True:
		if cv2.getWindowProperty(windowName,cv2.WND_PROP_VISIBLE) < 1:   # If user presses 'X' on the window,      
			cv2.destroyAllWindows()
			break
		
		if windowIsOpen:
      
			bgr_img = imutils.resize(bgr_img, width=700)
			results = pedestrian_detection(bgr_img, model, layer_name,
				personidz=LABELS.index("person"))

			for res in results:
				cv2.rectangle(bgr_img, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

			cv2.imshow(windowName,bgr_img)

			key = cv2.waitKey(1)
			if key == 27:
				break

			got_image, bgr_img = video_capture.read()
			bgr_img_copy = bgr_img.copy()

			if not got_image:
				break

	video_capture.release()
	cv2.destroyAllWindows()


