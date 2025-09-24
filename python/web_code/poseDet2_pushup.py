import cv2
import numpy as np
import time
import csv
from datetime import datetime
from pose_detector_web import PoseDetector  # the PoseDetector from before
import matplotlib.pyplot as plt


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    p_time = 0
    counter = 0
    stage = None

    # CSV setup
    csv_file = open("pushup_log.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "pushup_number"])

    # Matplotlib live graph setup
    plt.ion()  # interactive mode
    fig, ax = plt.subplots()
    x_vals, y_vals = [], []
    (line,) = ax.plot([], [], "b-")
    ax.set_xlabel("Time")
    ax.set_ylabel("Push-up count")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_pose(img)
        lm_list = detector.find_position(img, draw=False)

        if lm_list:
            # Left arm
            left_shoulder = lm_list[11][1:3]
            left_elbow = lm_list[13][1:3]
            left_wrist = lm_list[15][1:3]
            # Right arm
            right_shoulder = lm_list[12][1:3]
            right_elbow = lm_list[14][1:3]
            right_wrist = lm_list[16][1:3]

            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Show angles
            cv2.putText(
                img,
                str(int(left_angle)),
                (left_elbow[0], left_elbow[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                str(int(right_angle)),
                (right_elbow[0], right_elbow[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Push-up detection both arms
            if left_angle < 90 and right_angle < 90:
                stage = "down"
            if left_angle > 150 and right_angle > 150 and stage == "down":
                stage = "up"
                counter += 1

                # Save to CSV with timestamp
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([ts, counter])
                csv_file.flush()

                # Update graph data
                x_vals.append(ts)
                y_vals.append(counter)
                line.set_xdata(range(len(y_vals)))  # just index on x-axis
                line.set_ydata(y_vals)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Show counter
            cv2.putText(
                img,
                f"Pushups: {counter}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if c_time != p_time else 0
        p_time = c_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

        cv2.imshow("Pushup Counter", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
