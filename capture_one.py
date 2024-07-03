import cv2
import imutils
from datetime import datetime
from picamera2 import Picamera2
import os

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

current_frame = None

def capture_frame():
 frame = picam2.capture_array()
 frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
 return frame

# main ------------------------------------------------------------------------
current_original_frame = capture_frame()

#Save the images
folder = 'data/{}'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))
os.makedirs(folder)
cv2.imwrite('{}-original_frame.jpg'.format(folder), current_original_frame)