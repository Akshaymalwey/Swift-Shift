import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/face_landmarker.task"


# Important facial landmarks we want to inspect.
LANDMARKS = {
    "nose": 1,

    # Eyes
    "left_eye": 33,
    "right_eye": 263,

    # Mouth
    "left_mouth": 61,
    "right_mouth": 291,

    # Cheeks / sides of face
    "left_face": 234,
    "right_face": 454,

    # Jaw
    "left_jaw": 172,
    "right_jaw": 397,

    # Forehead / upper face
    "left_forehead": 127,
    "right_forehead": 356,

    # Chin
    "chin": 152,
}


def get_point(landmarks, index, width, height):

    landmark = landmarks[index]

    return np.array([
        landmark.x * width,
        landmark.y * height,
        landmark.z,
    ])


def draw_point(frame, point, label):

    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        frame,
        (x, y),
        5,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x + 7, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
    )


def distance_2d(a, b):

    return np.linalg.norm(
        a[:2] - b[:2]
    )


def main():

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

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                points = {}

                for name, index in LANDMARKS.items():

                    points[name] = get_point(
                        landmarks,
                        index,
                        width,
                        height,
                    )

                    draw_point(
                        frame,
                        points[name],
                        name,
                    )

                # --------------------------------
                # Face geometry measurements
                # --------------------------------

                nose = points["nose"]

                left_eye = points["left_eye"]
                right_eye = points["right_eye"]

                left_face = points["left_face"]
                right_face = points["right_face"]

                left_jaw = points["left_jaw"]
                right_jaw = points["right_jaw"]

                left_forehead = points["left_forehead"]
                right_forehead = points["right_forehead"]

                left_mouth = points["left_mouth"]
                right_mouth = points["right_mouth"]

                # Horizontal distances from nose.
                left_eye_dist = distance_2d(
                    left_eye,
                    nose,
                )

                right_eye_dist = distance_2d(
                    right_eye,
                    nose,
                )

                left_face_dist = distance_2d(
                    left_face,
                    nose,
                )

                right_face_dist = distance_2d(
                    right_face,
                    nose,
                )

                left_jaw_dist = distance_2d(
                    left_jaw,
                    nose,
                )

                right_jaw_dist = distance_2d(
                    right_jaw,
                    nose,
                )

                left_forehead_dist = distance_2d(
                    left_forehead,
                    nose,
                )

                right_forehead_dist = distance_2d(
                    right_forehead,
                    nose,
                )

                # --------------------------------
                # Asymmetry ratios
                # --------------------------------

                eye_ratio = (
                    left_eye_dist /
                    max(right_eye_dist, 1e-6)
                )

                face_ratio = (
                    left_face_dist /
                    max(right_face_dist, 1e-6)
                )

                jaw_ratio = (
                    left_jaw_dist /
                    max(right_jaw_dist, 1e-6)
                )

                forehead_ratio = (
                    left_forehead_dist /
                    max(right_forehead_dist, 1e-6)
                )

                # --------------------------------
                # Display measurements
                # --------------------------------

                y = 30

                measurements = [
                    f"Eye ratio:       {eye_ratio:.3f}",
                    f"Face ratio:      {face_ratio:.3f}",
                    f"Jaw ratio:       {jaw_ratio:.3f}",
                    f"Forehead ratio:  {forehead_ratio:.3f}",
                ]

                for text in measurements:

                    cv2.putText(
                        frame,
                        text,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    y += 30

            cv2.putText(
                frame,
                "Look at MacBook / Monitor and observe ratios",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "GazeSwitch - Face Geometry",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()