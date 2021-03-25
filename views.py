from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib.request  # import urllib because of internet 
import json
import cv2
import os
from absl.flags import FLAGS
from django.conf import settings

import uuid
import datetime

# define the path to the face detector
FACE_DETECTOR_PATH = 'cascades/haarcascade_frontalface_default.xml'




@csrf_exempt
def detect(request):
	data = {"success": False}
	if request.method == "POST":
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
			image = _grab_image(url=url) # download
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
		rects = detector.detectMultiScale(image1,1.3, 5)
		rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
		data.update({"num_faces": len(rects), "faces": rects, "success": True})
         
		 #add manully
		for (startX, startY, endX, endY) in rects:
    			cv2.rectangle(image1, (startX, startY), (endX, endY), (0, 255, 0), 2)
		newimg ='detection/'+format(str(uuid.uuid4().hex))+'.jpg'
		cv2.imshow('URL2Image',image1)
		cv2.imwrite(newimg, image1)
		cv2.waitKey()


	# return a JSON response
	return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
	if path is not None:
		image = cv2.imread(path)
	else:	
		if url is not None:
			resp = urllib.request.urlopen(url)
			data = resp.read()
		elif stream is not None:
			data = stream.read()
		
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		# add new line by me for draw faces
		# for (startX, startY, endX, endY) in rects:
    	# 		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
		# cv2.imshow('URL2Image',image)
		# cv2.waitKey()
 
	# return the image
	return image