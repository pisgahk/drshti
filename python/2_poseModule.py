import cv2
import mediapipe as mp
import time


class poseDetector:

    def __init__(
        self,
        mode=False,
        upBody=False,
        smooth=True,
        detectionConfidence=0.5,
        trackingConfidence=0.5,
    ):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.upBody,
            self.smooth,
            self.detectionConfidence,
            self.trackConfidence,
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )

        return img

    def getPosition(self, img, draw=True):

        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList


def main():
    cap = cv2.VideoCapture(0)  # 0 is for webcam, add file name to read from file.
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        if len(lmList) != 0:
            print(lmList[14])  # print only the results of position 14.
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


# Check also poseModule_test.py for further testing of this created module.
