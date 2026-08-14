import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/face_landmarker.task"


# MediaPipe eye landmark indices.
#
# These represent the outer/inner/top/bottom
# boundaries of each eye.

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


def get_landmark_xy(landmarks, index):
    landmark = landmarks[index]

    return np.array([
        landmark.x,
        landmark.y,
    ])


def get_iris_center(landmarks, indices):

    points = []

    for index in indices:
        points.append(
            get_landmark_xy(landmarks, index)
        )

    return np.mean(points, axis=0)


def get_normalized_iris_position(
    landmarks,
    eye,
    iris_indices,
):

    left = get_landmark_xy(
        landmarks,
        eye["left"]
    )

    right = get_landmark_xy(
        landmarks,
        eye["right"]
    )

    top = get_landmark_xy(
        landmarks,
        eye["top"]
    )

    bottom = get_landmark_xy(
        landmarks,
        eye["bottom"]
    )

    iris = get_iris_center(
        landmarks,
        iris_indices
    )

    # Horizontal normalization
    eye_width = right[0] - left[0]

    if abs(eye_width) < 1e-6:
        return None

    iris_x = (iris[0] - left[0]) / eye_width

    # Vertical normalization
    eye_height = bottom[1] - top[1]

    if abs(eye_height) < 1e-6:
        return None

    iris_y = (iris[1] - top[1]) / eye_height

    return iris_x, iris_y


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    timestamp_ms = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
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

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

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

                if left_gaze and right_gaze:

                    left_x, left_y = left_gaze
                    right_x, right_y = right_gaze

                    avg_x = (left_x + right_x) / 2
                    avg_y = (left_y + right_y) / 2

                    cv2.putText(
                        frame,
                        f"Left eye:  X={left_x:.2f} Y={left_y:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Right eye: X={right_x:.2f} Y={right_y:.2f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Average:   X={avg_x:.2f} Y={avg_y:.2f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(
                "GazeSwitch - Eye Gaze",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()