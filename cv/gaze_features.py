import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/face_landmarker.task"


# Facial landmarks used for head/face geometry
LANDMARKS = {
    "nose": 1,
    "left_eye": 33,
    "right_eye": 263,
    "left_face": 234,
    "right_face": 454,
    "left_jaw": 172,
    "right_jaw": 397,
    "left_forehead": 127,
    "right_forehead": 356,
    "chin": 152,
}


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


FACE_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


POSE_LANDMARK_IDS = [
    1,
    152,
    33,
    263,
    61,
    291,
]


def get_point(landmarks, index, width, height):

    landmark = landmarks[index]

    return np.array(
        [
            landmark.x * width,
            landmark.y * height,
            landmark.z,
        ]
    )


def distance_2d(a, b):

    return np.linalg.norm(
        a[:2] - b[:2]
    )


def calculate_yaw(
    face_landmarks,
    frame_width,
    frame_height,
):

    points_2d = []

    for landmark_id in POSE_LANDMARK_IDS:

        landmark = face_landmarks[landmark_id]

        points_2d.append(
            (
                landmark.x * frame_width,
                landmark.y * frame_height,
            )
        )

    points_2d = np.array(
        points_2d,
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

    distortion = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D_POINTS,
        points_2d,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    yaw = np.degrees(
        np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0],
        )
    )

    return yaw


def calculate_iris_center(
    landmarks,
    iris_ids,
    width,
    height,
):

    points = []

    for index in iris_ids:

        landmark = landmarks[index]

        points.append(
            [
                landmark.x * width,
                landmark.y * height,
            ]
        )

    return np.mean(
        np.array(points),
        axis=0,
    )


def calculate_iris_position(
    landmarks,
    iris_ids,
    eye_left_id,
    eye_right_id,
    width,
    height,
):

    iris = calculate_iris_center(
        landmarks,
        iris_ids,
        width,
        height,
    )

    eye_left = get_point(
        landmarks,
        eye_left_id,
        width,
        height,
    )

    eye_right = get_point(
        landmarks,
        eye_right_id,
        width,
        height,
    )

    eye_width = distance_2d(
        eye_left,
        eye_right,
    )

    if eye_width < 1e-6:
        return None

    # Project iris position onto the horizontal
    # direction of the eye.
    eye_vector = (
        eye_right[:2] -
        eye_left[:2]
    )

    eye_unit = (
        eye_vector /
        np.linalg.norm(eye_vector)
    )

    relative_vector = (
        iris -
        eye_left[:2]
    )

    iris_x = np.dot(
        relative_vector,
        eye_unit,
    ) / eye_width

    return iris_x


def draw_landmark(
    frame,
    point,
    label,
):

    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        frame,
        (x, y),
        4,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x + 6, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )


def put_text(
    frame,
    text,
    y,
):

    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
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

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            height, width, _ = frame.shape

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
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

                    draw_landmark(
                        frame,
                        points[name],
                        name,
                    )

                # -----------------------------
                # FACE GEOMETRY
                # -----------------------------

                nose = points["nose"]

                left_face = points["left_face"]
                right_face = points["right_face"]

                left_face_dist = distance_2d(
                    left_face,
                    nose,
                )

                right_face_dist = distance_2d(
                    right_face,
                    nose,
                )

                face_ratio = (
                    left_face_dist /
                    max(right_face_dist, 1e-6)
                )

                # -----------------------------
                # YAW
                # -----------------------------

                yaw = calculate_yaw(
                    landmarks,
                    width,
                    height,
                )

                # -----------------------------
                # IRIS
                # -----------------------------

                left_iris_x = calculate_iris_position(
                    landmarks,
                    LEFT_IRIS,
                    33,
                    133,
                    width,
                    height,
                )

                right_iris_x = calculate_iris_position(
                    landmarks,
                    RIGHT_IRIS,
                    263,
                    362,
                    width,
                    height,
                )

                if (
                    left_iris_x is not None
                    and right_iris_x is not None
                ):

                    avg_iris_x = (
                        left_iris_x +
                        right_iris_x
                    ) / 2

                else:

                    avg_iris_x = None

                # -----------------------------
                # DISPLAY VALUES
                # -----------------------------

                put_text(
                    frame,
                    f"Face Ratio: {face_ratio:.3f}",
                    30,
                )

                if yaw is not None:

                    put_text(
                        frame,
                        f"Yaw:        {yaw:.3f}",
                        60,
                    )

                if avg_iris_x is not None:

                    put_text(
                        frame,
                        f"Iris X:     {avg_iris_x:.3f}",
                        90,
                    )

                    put_text(
                        frame,
                        f"Left Iris:  {left_iris_x:.3f}",
                        120,
                    )

                    put_text(
                        frame,
                        f"Right Iris: {right_iris_x:.3f}",
                        150,
                    )

            cv2.putText(
                frame,
                "Q = quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "GazeSwitch - Combined Features",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()