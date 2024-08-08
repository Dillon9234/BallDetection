import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
lowerlab =np.array([25,0,140])
upperlab=np.array([255,140,220])
lowerhsv = np.array([24,80,100])
upperhsv = np.array([100,255,255])

kernal1 = np.ones((10,10),np.uint8)
kernal2 = np.ones((6,6),np.uint8)

def nothing(x):
    pass

while True:
    ret, frame = cap.read()
    if not ret:
        break
    blurredFrame = cv.GaussianBlur(frame, (11,11), 20)
    blurredFrame = cv.medianBlur(blurredFrame, 9, 0)
    HSVFrame = cv.cvtColor(blurredFrame,cv.COLOR_BGR2HSV)
    LABFrame = cv.cvtColor(blurredFrame, cv.COLOR_BGR2LAB)
    hsv = cv.inRange(HSVFrame,lowerhsv,upperhsv)
    lab = cv.inRange(LABFrame,lowerlab,upperlab)
    mask = cv.bitwise_or(lab, lab, mask=hsv)
    mask = cv.erode(mask,kernal2,iterations=1)
    mask =cv.dilate(mask,kernal1,iterations=1)
    mask = cv.GaussianBlur(mask, (11,11), 20)
    cv.imshow("Mask",mask)
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT,1, minDist=50,param1=100,param2=40, minRadius=10,maxRadius=1000) 
    if circles is not None :
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    
    cv.imshow("frame",frame)
    key = cv.waitKey(1)
    if key == ord('e'):
        break
cap.release()
cv.destroyAllWindows()