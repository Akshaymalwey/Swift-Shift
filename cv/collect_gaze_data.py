import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/face_landmarker.task"
OUTPUT_FILE = "data/gaze_samples_test.csv"


# -----------------------------
# MediaPipe landmark definitions
# -----------------------------

LEFT_EYE = {
    "left": 33,
    "right": 133,
    "top": 159,
    "bottom": 145,
}

RIGHT_EYE = {
    "left": 362,
    "right": 263,
    "top": 386,
    "bottom": 374,
}

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# 3D facial reference points used for head pose.
FACE_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye
        (225.0, 170.0, -135.0),   # Right eye
        (-150.0, -150.0, -125.0), # Left mouth
        (150.0, -150.0, -125.0),  # Right mouth
    ],
    dtype=np.float64,
)

LANDMARK_IDS = [
    1,    # Nose
    152,  # Chin
    33,   # Left eye
    263,  # Right eye
    61,   # Left mouth
    291,  # Right mouth
]


def get_xy(landmarks, index):
    landmark = landmarks[index]

    return np.array([
        landmark.x,
        landmark.y,
    ])


def get_iris_center(landmarks, indices):
    points = [
        get_xy(landmarks, index)
        for index in indices
    ]

    return np.mean(points, axis=0)


def get_normalized_iris_position(
    landmarks,
    eye,
    iris_indices,
):
    left = get_xy(landmarks, eye["left"])
    right = get_xy(landmarks, eye["right"])

    top = get_xy(landmarks, eye["top"])
    bottom = get_xy(landmarks, eye["bottom"])

    iris = get_iris_center(
        landmarks,
        iris_indices,
    )

    eye_width = right[0] - left[0]

    if abs(eye_width) < 1e-6:
        return None

    iris_x = (iris[0] - left[0]) / eye_width

    eye_height = bottom[1] - top[1]

    if abs(eye_height) < 1e-6:
        return None

    iris_y = (iris[1] - top[1]) / eye_height

    return iris_x, iris_y


def calculate_head_pose(
    landmarks,
    frame_width,
    frame_height,
):
    face_2d_points = []

    for landmark_id in LANDMARK_IDS:
        landmark = landmarks[landmark_id]

        x = landmark.x * frame_width
        y = landmark.y * frame_height

        face_2d_points.append((x, y))

    face_2d_points = np.array(
        face_2d_points,
        dtype=np.float64,
    )

    focal_length = frame_width

    center = (
        frame_width / 2,
        frame_height / 2,
    )

    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D_POINTS,
        face_2d_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(
        rotation_matrix
    )

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll


def create_csv_if_needed():
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    if not os.path.exists(OUTPUT_FILE):
        with open(
            OUTPUT_FILE,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "yaw",
                "pitch",
                "roll",
                "left_iris_x",
                "left_iris_y",
                "right_iris_x",
                "right_iris_y",
                "avg_iris_x",
                "avg_iris_y",
                "target",
            ])


def save_sample(
    timestamp,
    pose,
    left_gaze,
    right_gaze,
    target,
):
    pitch, yaw, roll = pose

    left_x, left_y = left_gaze
    right_x, right_y = right_gaze

    avg_x = (left_x + right_x) / 2
    avg_y = (left_y + right_y) / 2

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            yaw,
            pitch,
            roll,
            left_x,
            left_y,
            right_x,
            right_y,
            avg_x,
            avg_y,
            target,
        ])


def main():

    create_csv_if_needed()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    timestamp_ms = 0

    current_target = None
    last_saved_time = 0

    mac_samples = 0
    monitor_samples = 0

    print()
    print("====================================")
    print("       GazeSwitch Data Collector")
    print("====================================")
    print()
    print("Controls:")
    print("  M → Looking at MacBook")
    print("  O → Looking at external Monitor")
    print("  N → Clear current target")
    print("  Q → Quit")
    print()
    print("Start by looking at the MacBook and press M.")
    print()

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Failed to read camera frame.")
                break

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            pose = None
            left_gaze = None
            right_gaze = None

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                pose = calculate_head_pose(
                    landmarks,
                    width,
                    height,
                )

                left_gaze = get_normalized_iris_position(
                    landmarks,
                    LEFT_EYE,
                    LEFT_IRIS,
                )

                right_gaze = get_normalized_iris_position(
                    landmarks,
                    RIGHT_EYE,
                    RIGHT_IRIS,
                )

            # ---------------------------------
            # Save a sample every 100 ms
            # ---------------------------------

            current_time = time.time()

            if (
                current_target is not None
                and pose is not None
                and left_gaze is not None
                and right_gaze is not None
                and current_time - last_saved_time >= 0.1
            ):

                save_sample(
                    current_time,
                    pose,
                    left_gaze,
                    right_gaze,
                    current_target,
                )

                last_saved_time = current_time

                if current_target == "mac":
                    mac_samples += 1
                elif current_target == "monitor":
                    monitor_samples += 1

            # ---------------------------------
            # Display information
            # ---------------------------------

            if pose is not None:
                pitch, yaw, roll = pose

                cv2.putText(
                    frame,
                    f"Yaw: {yaw:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Pitch: {pitch:.1f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Roll: {roll:.1f}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if left_gaze and right_gaze:

                left_x, left_y = left_gaze
                right_x, right_y = right_gaze

                avg_x = (left_x + right_x) / 2
                avg_y = (left_y + right_y) / 2

                cv2.putText(
                    frame,
                    f"Iris X: {avg_x:.2f}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Iris Y: {avg_y:.2f}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Current target

            target_text = (
                current_target.upper()
                if current_target
                else "NONE"
            )

            cv2.putText(
                frame,
                f"TARGET: {target_text}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"MAC samples: {mac_samples}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"MONITOR samples: {monitor_samples}",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "M=Mac  O=Monitor  N=None  Q=Quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "GazeSwitch - Data Collector",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("m"):
                current_target = "mac"
                print("Collecting: MACBOOK")

            elif key == ord("o"):
                current_target = "monitor"
                print("Collecting: MONITOR")

            elif key == ord("n"):
                current_target = None
                print("Collection paused.")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("====================================")
    print("Collection finished")
    print("====================================")
    print(f"MacBook samples: {mac_samples}")
    print(f"Monitor samples: {monitor_samples}")
    print(f"Saved to: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()