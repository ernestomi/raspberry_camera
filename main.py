import cv2
import imutils
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

firstFrame = None

while True:
 frame = picam2.capture_array()
 # resize the frame, convert it to grayscale, and blur it
 frame = imutils.resize(frame, width=500)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 gray = cv2.GaussianBlur(gray, (21, 21), 0)

 # if the first frame is None, initialize it
 if firstFrame is None:
  firstFrame = gray
  continue

 frameDelta = cv2.absdiff(firstFrame, gray)
 thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 # dilate the thresholded image to fill in holes, then find contours
 # on thresholded image
 thresh = cv2.dilate(thresh, None, iterations=2)
 contours = cv2.findContours(
  thresh.copy(), 
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
  cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  #Save the image
  cv2.imwrite('motion_detected.jpg', frame)
  exit()
 