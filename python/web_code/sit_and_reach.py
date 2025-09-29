import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
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

    def find_position(self, img):
        lm_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy, lm.visibility])
        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    # calibration (cm per pixel)
    scale_cm_per_pixel = 0.3  # adjust for your setup

    while True:
        success, img = cap.read()
        if not success:
            break

        h, w, _ = img.shape

        img = detector.find_pose(img)
        lm_list = detector.find_position(img)

        full_body = False
        reach_distance_m = 0.0

        if lm_list:
            # Key landmarks: head(0), hips(23,24), ankles(27,28)
            key_ids = [0, 11, 12, 23, 24, 27, 28]
            visibilities = [lm_list[i][3] for i in key_ids if i < len(lm_list)]
            if len(visibilities) == len(key_ids) and all(v > 0.5 for v in visibilities):
                full_body = True

            if full_body:
                # Compute reach distance (hips to fingertips)
                # Hip center
                hip_x = (lm_list[23][1] + lm_list[24][1]) / 2
                hip_y = (lm_list[23][2] + lm_list[24][2]) / 2

                # Fingertip (use left wrist id 15 and right wrist id 16)
                left_wrist = lm_list[15]
                right_wrist = lm_list[16]
                hand_x = (left_wrist[1] + right_wrist[1]) / 2
                hand_y = (left_wrist[2] + right_wrist[2]) / 2

                pixel_distance = np.sqrt((hand_x - hip_x) ** 2 + (hand_y - hip_y) ** 2)
                reach_distance_cm = pixel_distance * scale_cm_per_pixel
                reach_distance_m = reach_distance_cm / 100.0

                cv2.putText(
                    img,
                    f"Reach: {reach_distance_m:.2f} m",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

        # Draw dot for frame status
        dot_color = (0, 255, 0) if full_body else (0, 0, 255)
        cv2.circle(img, (w - 20, 20), 10, dot_color, -1)

        cv2.imshow("Sit and Reach Test", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
