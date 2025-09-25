import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (25, 0, 255),
        2,
    )  # Displaying the FPS

    cv2.imshow("Cam Frame", img)
    cv2.waitKey(1)
