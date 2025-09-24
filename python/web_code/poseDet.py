import cv2
import time
import mediapipe as mp

class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 enable_segmentation=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     enable_segmentation=self.enable_segmentation,
                                     smooth_landmarks=self.smooth_landmarks,
                                     min_detection_confidence=self.min_detection_confidence,
                                     min_tracking_confidence=self.min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img,
                                       self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        """Return a list of [id, x, y] pixel coordinates of landmarks."""
        lm_list = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=True)
        if lm_list:
            print(lm_list[0])  # print first landmark for demonstration

        c_time = time.time()
        fps = 1 / (c_time - p_time) if c_time != p_time else 0
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
