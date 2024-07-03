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
 frame = cv2.rotate(frame, 0)
 return frame

def process_frame_for_motion(frame):
 frame = imutils.resize(frame, width=500)
 # Grayscale and Blur
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 gray = cv2.GaussianBlur(gray, (21, 21), 0)
 return gray

def has_movement(previous_frame, current_frame):
 delta = cv2.absdiff(previous_frame, current_frame)
 threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
 threshold = cv2.dilate(threshold, None, iterations=2)
 contours = cv2.findContours(
  threshold.copy(), 
  cv2.RETR_EXTERNAL,
  cv2.CHAIN_APPROX_SIMPLE
 )
 contours = imutils.grab_contours(contours)
 for c in contours:
  if cv2.contourArea(c) < 500:
   continue
  return True

while True:
 previous_frame = current_frame
 original_frame = capture_frame()
 current_frame = process_frame_for_motion(original_frame)
 if previous_frame is None:
  continue
 if has_movement(previous_frame, current_frame):
  print('Movement detected')
  #Save the images
  folder = 'data/{}'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))
  os.makedirs(folder)
  cv2.imwrite('{}/0-original_frame.jpg'.format(folder), original_frame)
  cv2.imwrite('{}/3-previous_frame.jpg'.format(folder), previous_frame)