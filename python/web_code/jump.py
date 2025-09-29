# import cv2
# import time
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
#         self.mode = static_image_mode
#         self.model_complexity = model_complexity
#         self.enable_segmentation = enable_segmentation
#         self.smooth_landmarks = smooth_landmarks
#         self.min_detection_confidence = min_detection_confidence
#         self.min_tracking_confidence = min_tracking_confidence
#
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(
#             static_image_mode=self.mode,
#             model_complexity=self.model_complexity,
#             enable_segmentation=self.enable_segmentation,
#             smooth_landmarks=self.smooth_landmarks,
#             min_detection_confidence=self.min_detection_confidence,
#             min_tracking_confidence=self.min_tracking_confidence,
#         )
#
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
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, _ = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lm_list.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return lm_list
#
#
# def main():
#     cap = cv2.VideoCapture(0)
#     detector = PoseDetector()
#
#     baseline_y = None
#     max_jump_pixels = 0
#     scale_cm_per_pixel = 0.3  # <-- adjust after calibration (cm per pixel)
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         img = detector.find_pose(img)
#         lm_list = detector.find_position(img, draw=True)
#
#         if lm_list:
#             # Use average of left and right ankles (id 27 & 28) or hips (23 & 24)
#             ankle_left = lm_list[27]
#             ankle_right = lm_list[28]
#             ankle_y = int((ankle_left[2] + ankle_right[2]) / 2)
#
#             # Press 's' to set baseline standing ankle_y
#             if baseline_y is None:
#                 cv2.putText(
#                     img,
#                     "Press 's' to set baseline",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 255, 255),
#                     2,
#                 )
#
#             if baseline_y is not None:
#                 jump_pixels = baseline_y - ankle_y  # positive when you jump up
#                 if jump_pixels > max_jump_pixels:
#                     max_jump_pixels = jump_pixels
#
#                 jump_height_cm = max_jump_pixels * scale_cm_per_pixel
#
#                 cv2.putText(
#                     img,
#                     f"Jump Height: {jump_height_cm:.1f} cm",
#                     (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1.2,
#                     (255, 0, 0),
#                     3,
#                 )
#             else:
#                 cv2.putText(
#                     img,
#                     f"Baseline not set",
#                     (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 0, 255),
#                     2,
#                 )
#
#             # Draw ankle point and baseline line
#             cv2.circle(img, (ankle_left[1], ankle_left[2]), 8, (0, 255, 0), -1)
#             if baseline_y is not None:
#                 cv2.line(
#                     img, (0, baseline_y), (img.shape[1], baseline_y), (0, 255, 0), 2
#                 )
#
#         cv2.imshow("Jump Tracker", img)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         if key == ord("s") and lm_list:
#             # set baseline when standing still
#             ankle_left = lm_list[27]
#             ankle_right = lm_list[28]
#             baseline_y = int((ankle_left[2] + ankle_right[2]) / 2)
#             max_jump_pixels = 0  # reset
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()


import cv2
import time
import mediapipe as mp


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
                lm_list.append([id, cx, cy, lm.visibility])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    baseline_y = None
    max_jump_pixels = 0
    scale_cm_per_pixel = 0.3  # adjust after calibration

    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=True)

        full_body = False
        if lm_list:
            # Check visibility of key points: nose(0), left ankle(27), right ankle(28)
            # and maybe left shoulder(11), right shoulder(12)
            key_ids = [0, 11, 12, 23, 24, 27, 28]
            visibilities = [lm_list[i][3] for i in key_ids if i < len(lm_list)]
            if len(visibilities) == len(key_ids) and all(v > 0.5 for v in visibilities):
                full_body = True

        if full_body:
            ankle_left = lm_list[27]
            ankle_right = lm_list[28]
            ankle_y = int((ankle_left[2] + ankle_right[2]) / 2)

            if baseline_y is None:
                cv2.putText(
                    img,
                    "Press 's' to set baseline",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

            if baseline_y is not None:
                jump_pixels = baseline_y - ankle_y  # positive when jumping up
                if jump_pixels > max_jump_pixels:
                    max_jump_pixels = jump_pixels

                jump_height_cm = max_jump_pixels * scale_cm_per_pixel
                cv2.putText(
                    img,
                    f"Jump Height: {jump_height_cm:.1f} cm",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    3,
                )
            else:
                cv2.putText(
                    img,
                    "Baseline not set",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                img,
                "Step back until full body is visible",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Jump Tracker", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and full_body:
            ankle_left = lm_list[27]
            ankle_right = lm_list[28]
            baseline_y = int((ankle_left[2] + ankle_right[2]) / 2)
            max_jump_pixels = 0  # reset

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
