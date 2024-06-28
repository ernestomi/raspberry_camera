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

def process_frame_for_motion(frame):
 # Grayscale and Blur
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 gray = cv2.GaussianBlur(gray, (21, 21), 0)
 return gray

def process_frame_for_face(frame):
 # Grayscale
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 return gray

# main ------------------------------------------------------------------------
face_classifier = cv2.CascadeClassifier(
 'classifiers/haarcascade_frontalface_default.xml')

while True:
 contours_found = False
 faces_found = False
 previous_frame = current_frame
 current_original_frame = capture_frame()
 current_frame = process_frame_for_motion(current_original_frame)
 if previous_frame is None:
  continue
 
 delta = cv2.absdiff(previous_frame, current_frame)
 threshold = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
 # dilate the thresholded image to fill in holes, then find contours
 # on thresholded image
 threshold = cv2.dilate(threshold, None, iterations=2)
 contours = cv2.findContours(
  threshold.copy(), 
  cv2.RETR_EXTERNAL,
  cv2.CHAIN_APPROX_SIMPLE
 )
 contours = imutils.grab_contours(contours)
 # loop over the contours
 for c in contours:
  # if the contour is too small, ignore it
  if cv2.contourArea(c) < 500:
   continue
  print('contour found')
  contours_found = True
  # compute the bounding box for the contour
  #(x, y, w, h) = cv2.boundingRect(c)
  #print('x: {}, y: {}, w: {}, h: {}'.format(x, y, w, h))
  # draw the bounding box on the frame
  #cv2.rectangle(current_original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
 if not contours_found:
  continue

 current_frame = process_frame_for_face(current_original_frame)
 #Detect face
 faces = face_classifier.detectMultiScale(
  current_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
 )
 for (x, y, w, h) in faces:
  print('Face found')
  faces_found = True
  # draw the bounding box on the frame
  cv2.rectangle(current_original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
 if faces_found:
  #Save the images
  folder = 'data/{}'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))
  os.makedirs(folder)
  cv2.imwrite('{}/0-original_frame.jpg'.format(folder), current_original_frame)
  cv2.imwrite('{}/1-current_frame.jpg'.format(folder), current_frame)

 faces_found = False
 contours_found = False

  #Save the images
  #folder = 'data/{}'.format(datetime.now().strftime('%Y%m%d%H%M%S%f'))
  #os.makedirs(folder)
  #cv2.imwrite('{}/0-original_frame.jpg'.format(folder), current_original_frame)
  #cv2.imwrite('{}/1-current_frame.jpg'.format(folder), current_frame)
  #cv2.imwrite('{}/2-previous_frame.jpg'.format(folder), previous_frame)
  #cv2.imwrite('{}/3-delta.jpg'.format(folder), delta)
  #cv2.imwrite('{}/4-threshold.jpg'.format(folder), threshold)
 