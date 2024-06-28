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
 #frame = imutils.resize(frame, width=500)
 frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
 return frame

def process_frame(frame):
 # Grayscale
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #gray = cv2.GaussianBlur(gray, (21, 21), 0)
 return gray

# main ------------------------------------------------------------------------
face_classifier = cv2.CascadeClassifier(
 'classifiers/haarcascade_frontalface_default.xml')

while True:
 target_found = False
 current_original_frame = capture_frame()
 current_frame = process_frame(current_original_frame)

 #Detect face
 faces = face_classifier.detectMultiScale(
  current_frame, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40)
 )
 for (x, y, w, h) in faces:
  print('Face found')
  target_found = True
  # draw the bounding box on the frame
  cv2.rectangle(current_original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
 if target_found:
  #Save the images
  folder = 'data/{}'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))
  os.makedirs(folder)
  cv2.imwrite('{}/0-original_frame.jpg'.format(folder), current_original_frame)
  cv2.imwrite('{}/1-current_frame.jpg'.format(folder), current_frame)
 