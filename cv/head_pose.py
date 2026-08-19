import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/face_landmarker.task"


# Approximate 3D coordinates of facial landmarks.
# Coordinate system:
#
# X → left/right
# Y → up/down
# Z → depth
#
# Units don't matter; relative geometry does.

FACE_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


# MediaPipe landmark indices corresponding to the points above.
#
# These are approximate:
#
# 1   → nose
# 152 → chin
# 33  → left eye
# 263 → right eye
# 61  → left mouth
# 291 → right mouth

LANDMARK_IDS = [
    1,
    152,
    33,
    263,
    61,
    291,
]


def calculate_head_pose(face_landmarks, frame_width, frame_height):

    face_2d_points = []

    for landmark_id in LANDMARK_IDS:

        landmark = face_landmarks[landmark_id]

        x = landmark.x * frame_width
        y = landmark.y * frame_height

        face_2d_points.append((x, y))

    face_2d_points = np.array(
        face_2d_points,
        dtype=np.float64
    )

    # Approximate camera parameters.
    focal_length = frame_width

    center = (
        frame_width / 2,
        frame_height / 2
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
        dtype=np.float64
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
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

    yaw = np.degrees(
        np.arctan2(
            rotation_matrix[1,0],
            rotation_matrix[0,0],
        )
    )

    return yaw


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

            frame_height, frame_width, _ = frame.shape

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

                face_landmarks = result.face_landmarks[0]

                pose = calculate_head_pose(
                    face_landmarks,
                    frame_width,
                    frame_height
                )

                if pose is not None:

                    yaw = pose

                    cv2.putText(
                        frame,
                        f"Yaw:   {yaw:.2f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )


            cv2.imshow(
                "GazeSwitch - Head Pose",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()