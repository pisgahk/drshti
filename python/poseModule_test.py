import cv2
import time
import poseModule as pm



cap = cv2.VideoCapture(0) # 0 is for webcam, add file name to read from file.
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img)

    if len(lmList) != 0:
            print(lmList[14]) #print only the results of position 14.
            cv2.Circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED       )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)