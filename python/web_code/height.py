# import cv2
# import mediapipe as mp
#
#
# class PoseDetector:
#     def __init__(
#         self,
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         smooth_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#     ):
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(
#             static_image_mode=static_image_mode,
#             model_complexity=model_complexity,
#             enable_segmentation=enable_segmentation,
#             smooth_landmarks=smooth_landmarks,
#             min_detection_confidence=min_detection_confidence,
#             min_tracking_confidence=min_tracking_confidence,
#         )
#         self.mpDraw = mp.solutions.drawing_utils
#         self.results = None
#
#     def find_pose(self, img, draw=True):
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(img_rgb)
#         if self.results.pose_landmarks and draw:
#             self.mpDraw.draw_landmarks(
#                 img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
#             )
#         return img
#
#     def find_position(self, img, draw=True):
#         lm_list = []
#         if self.results and self.results.pose_landmarks:
#             h, w, _ = img.shape
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lm_list.append([id, cx, cy, lm.visibility])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return lm_list
#
#
# def main():
#     cap = cv2.VideoCapture(0)
#     detector = PoseDetector()
#
#     # scale: real cm per pixel at your camera setup
#     scale_cm_per_pixel = 0.3  # calibrate this!
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         img = detector.find_pose(img)
#         lm_list = detector.find_position(img, draw=True)
#
#         if lm_list:
#             # Use top of head and ankle to estimate total body length
#             # Nose (id 0) is near top of head; left ankle 27, right ankle 28
#             nose = lm_list[0]
#             ankle_left = lm_list[27]
#             ankle_right = lm_list[28]
#
#             ankle_y = int((ankle_left[2] + ankle_right[2]) / 2)
#             head_y = nose[2]
#
#             pixel_height = abs(ankle_y - head_y)
#             estimated_height_cm = pixel_height * scale_cm_per_pixel
#
#             cv2.putText(
#                 img,
#                 f"Pixel Height: {pixel_height}px",
#                 (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 255),
#                 2,
#             )
#             cv2.putText(
#                 img,
#                 f"Estimated Height: {estimated_height_cm:.1f} cm",
#                 (10, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.2,
#                 (255, 0, 0),
#                 3,
#             )
#
#         else:
#             cv2.putText(
#                 img,
#                 "Step into frame",
#                 (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 2,
#             )
#
#         cv2.imshow("Height Estimator", img)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import time
from collections import deque


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
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=smooth_landmarks,
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

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy, lm.visibility])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    scale_cm_per_pixel = 0.3  # calibrate for your camera/distance

    # For rolling average
    height_buffer = deque(maxlen=50)  # roughly 2 sec at 25 fps
    last_avg_time = 0
    avg_height_ft = None
    avg_height_m = None
    printed = False

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=True)

        h, w, _ = img.shape

        full_body = False
        if lm_list:
            key_ids = [0, 11, 12, 23, 24, 27, 28]  # head, shoulders, hips, ankles
            visibilities = [lm_list[i][3] for i in key_ids if i < len(lm_list)]
            if len(visibilities) == len(key_ids) and all(v > 0.5 for v in visibilities):
                full_body = True

        if full_body and lm_list:
            # Head and ankle
            head = lm_list[0]
            ankle_left = lm_list[27]
            ankle_right = lm_list[28]
            ankle_y = int((ankle_left[2] + ankle_right[2]) / 2)
            head_y = head[2]

            pixel_height = abs(ankle_y - head_y)
            height_cm = pixel_height * scale_cm_per_pixel
            height_m = height_cm / 100.0
            height_ft = height_cm / 30.48  # 1 ft = 30.48 cm

            # store rolling average
            height_buffer.append((height_ft, height_m))

            # compute average every ~10 sec
            now = time.time()
            if now - last_avg_time > 10 and len(height_buffer) > 0:
                avg_height_ft = sum(hf for hf, hm in height_buffer) / len(height_buffer)
                avg_height_m = sum(hm for hf, hm in height_buffer) / len(height_buffer)
                last_avg_time = now
                printed = False  # reset printing

            # display current height top right
            cv2.putText(
                img,
                f"{height_ft:.2f} ft / {height_m:.2f} m",
                (w - 350, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            # display average for ~10 sec after calculation
            if avg_height_ft is not None and now - last_avg_time < 10:
                cv2.putText(
                    img,
                    f"Avg: {avg_height_ft:.2f} ft / {avg_height_m:.2f} m",
                    (w - 380, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                if not printed:
                    print(
                        f"Average height over last period: {avg_height_ft:.2f} ft / {avg_height_m:.2f} m"
                    )
                    printed = True

        else:
            cv2.putText(
                img,
                "Move until full body visible",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Height Estimator", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
