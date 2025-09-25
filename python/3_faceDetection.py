import cv2
import time
import mediapipe as mp


class faceDetector:

    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection  # face detection module.
        self.mpDraw = mp.solutions.drawing_utils  # for drawing utils.
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        print(self.results)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(
                #     img, detection
                # )  # has some red marks on the face with a green box around

                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)

                # Making the bounding box manually.
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                bboxs.append([bbox, detection.score])

                img = self.fancyDraw(img, bbox)

                cv2.putText(
                    img,
                    f"{int(detection.score[0] * 100)}%",
                    (bbox[0], bbox[1] - 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (25, 0, 255),
                    2,
                )  # Writing the confidence value on the bounding box.
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (25, 34, 255), 2)

        colour = (255, 243, 255)

        # Top Left x,y
        # cv2.line(img, (x, y), (x + l, y), colour, t)
        # cv2.line(img, (x, y), (x, y + l), colour, t)

        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), colour, t)
        cv2.line(img, (x1, y), (x1, y + l), colour, t)

        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), colour, t)
        cv2.line(img, (x, y1), (x, y1 - l), colour, t)

        # Bottom Right x1, y1
        # cv2.line(img, (x1, y1), (x1 - l, y1), colour, t)
        # cv2.line(img, (x1, y1), (x1, y1 - l), colour, t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = faceDetector()

    while True:
        success, img = cap.read()

        img, bboxs = detector.findFaces(img)
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

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
