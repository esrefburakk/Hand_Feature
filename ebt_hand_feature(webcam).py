import cv2
import numpy as np 
import math

def findMaxContours(contours):
    max_i = 0
    max_area = 0

    for i in range(len(contours)):
        area_hand = cv2.contourArea(contours[i])
        if max_area < area_hand:
            max_area = area_hand
            max_i = i

    cnt = contours[max_i]

    return cnt

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    roi1 = frame[50:250,200:400]
    cv2.rectangle(frame,(200,50),(400,250),(0,0,255),0)

    hsv = cv2.cvtColor(roi1,cv2.COLOR_BGR2HSV)

    lower_color = np.array([0,35,71],dtype=np.uint8)
    upper_color = np.array([180,134,255],dtype=np.uint8)

    mask = cv2.inRange(hsv,lower_color,upper_color)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = -1)
    mask = cv2.medianBlur(mask,15)

    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = findMaxContours(contours)

        extRight = tuple(max_contour[max_contour[:,:,0].argmax()][0]) 
        extLeft = tuple(max_contour[max_contour[:,:,0].argmin()][0]) 
        extTop = tuple(max_contour[max_contour[:,:,1].argmin()][0])
        extButtom = tuple(max_contour[max_contour[:,:,1].argmax()][0])

        cv2.circle(roi1,extLeft,5,(0,255,0),2)
        cv2.circle(roi1,extRight,5,(0,255,0),2)
        cv2.circle(roi1,extTop,5,(0,255,0),2)


        cv2.line(roi1,extLeft,extTop,(0,255,0),2)
        cv2.line(roi1,extTop,extRight,(0,255,0),2)
        cv2.line(roi1,extRight,extLeft,(0,255,0),2)
            
            
        a = math.sqrt((extRight[0]-extTop[0])**2+(extRight[1]-extTop[1])**2)
        b = math.sqrt((extTop[0]-extLeft[0])**2+(extTop[1]-extLeft[1])**2)
        c = math.sqrt((extRight[0]-extLeft[0])**2+(extRight[1]-extLeft[1])**2)

        try:
            angle_ab= int(math.acos((a**2+c**2-b**2)/(2*b*c))*57)
            cv2.putText(roi1,str(angle_ab),(extRight[0]-100+50,extRight[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
            if angle_ab > 70:
                cv2.rectangle(frame,(0,0),(100,100),(0,255,0),-1)
            else:
                pass
        except:
             cv2.putText(roi1," ? ",(extRight[0]-100+50,extRight[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

    cv2.imshow("ROI",roi1)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
