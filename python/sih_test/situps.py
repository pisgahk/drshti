import cv2
import time
import mediapipe as mp
import math


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            enable_segmentation=self.enable_segmentation,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )
        return img

    def find_position(self, img, draw=True):
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

    count = 0
    stage = None  # 'down' or 'up'

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=True)

        if lm_list:
            # We’ll use shoulders and hips to detect trunk angle:
            # left shoulder id=11, left hip id=23 (or use right side 12 & 24)
            shoulder = lm_list[11]  # [id, x, y]
            hip = lm_list[23]
            knee = lm_list[25]

            # Calculate angle at hip (torso vs thigh):
            def angle(a, b, c):
                ax, ay = a[1], a[2]
                bx, by = b[1], b[2]
                cx, cy = c[1], c[2]
                ang = math.degrees(
                    math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
                )
                return abs(ang)

            hip_angle = angle(shoulder, hip, knee)
            # When lying back hip_angle is large (~160–180°), when sitting up ~70–90°
            if hip_angle < 100:
                stage = "up"
            if hip_angle > 150 and stage == "up":
                stage = "down"
                count += 1

            cv2.putText(
                img,
                f"Angle: {int(hip_angle)}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"Count: {count}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                3,
            )

        c_time = time.time()
        fps = 1 / (c_time - p_time) if c_time != p_time else 0
        p_time = c_time

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

        cv2.imshow("Sit-up Counter", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
