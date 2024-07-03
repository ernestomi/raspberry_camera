import cv2
import imutils
from datetime import datetime
from picamera2 import Picamera2
import os

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()


base = 'data/{}'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))

original_frame = picam2.capture_array()
cv2.imwrite('{}-original.jpg'.format(base), original_frame)
#Rotate
frame = cv2.rotate(original_frame, 0)
#Save the images
cv2.imwrite('{}-0.jpg'.format(base), frame)
#Rotate
frame = cv2.rotate(original_frame, 1)
#Save the images
cv2.imwrite('{}-1.jpg'.format(base), frame)
#Rotate
frame = cv2.rotate(original_frame, 2)
#Save the images
cv2.imwrite('{}-2.jpg'.format(base), frame)