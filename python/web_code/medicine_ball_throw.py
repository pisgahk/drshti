import cv2
import mediapipe as mp
import numpy as np
import time

# Pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Calibration: cm per pixel
scale_cm_per_pixel = 0.3  # adjust to your setup

# Ball colour range (HSV) – example for a red ball
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

cap = cv2.VideoCapture(0)

ball_positions = []
tracking_ball = False
last_throw_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    full_body = False
    lm_list = []
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy, lm.visibility])

    if lm_list:
        key_ids = [0, 11, 12, 23, 24, 27, 28]
        visibilities = [lm_list[i][3] for i in key_ids if i < len(lm_list)]
        if len(visibilities) == len(key_ids) and all(v > 0.5 for v in visibilities):
            full_body = True

    if full_body:
        # Ball detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius > 5:
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                ball_positions.append(center)
                tracking_ball = True
                last_throw_time = time.time()
        else:
            # if ball lost but we were tracking, compute metrics
            if tracking_ball and time.time() - last_throw_time > 1:
                # compute horizontal distance
                if len(ball_positions) > 1:
                    x_coords = [p[0] for p in ball_positions]
                    y_coords = [p[1] for p in ball_positions]
                    dist_px = max(x_coords) - min(x_coords)
                    dist_cm = dist_px * scale_cm_per_pixel
                    dist_m = dist_cm / 100.0
                    print(f"Estimated throw distance: {dist_m:.2f} m")
                ball_positions = []
                tracking_ball = False

        # Draw trajectory
        for pt in ball_positions:
            cv2.circle(frame, pt, 3, (255, 0, 0), -1)

        cv2.putText(
            frame,
            "Full body detected: Tracking enabled",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Move into frame until full body visible",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Medicine Ball Throw", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
