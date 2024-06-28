import cv2
import imutils
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

current_frame = None

def process_frame(frame):
 # resize the frame, convert it to grayscale, and blur it
 frame = imutils.resize(frame, width=500)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 gray = cv2.GaussianBlur(gray, (21, 21), 0)
 return gray

while True:
 previous_frame = current_frame
 current_frame = process_frame(picam2.capture_array())
 if not previous_frame:
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
  # compute the bounding box for the contour
  (x, y, w, h) = cv2.boundingRect(c)
  # draw the bounding box on the frame
  cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  #Save the images
  cv2.imwrite('delta.jpg', delta)
  cv2.imwrite('threshold.jpg', threshold)
  cv2.imwrite('current_frame.jpg', current_frame)
  cv2.imwrite('previous_frame.jpg', previous_frame)

  exit()
 