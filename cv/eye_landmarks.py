import cv2
import mediapipe as mp


MODEL_PATH = "models/face_landmarker.task"


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
                data=rgb_frame
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # Draw every facial landmark very small.
                for i, landmark in enumerate(landmarks):

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

                # Highlight the eye/iris regions.
                #
                # These are MediaPipe Face Landmarker
                # iris landmark indices.

                iris_indices = [
                    # Left iris
                    468, 469, 470, 471, 472,

                    # Right iris
                    473, 474, 475, 476, 477,
                ]

                for index in iris_indices:

                    if index >= len(landmarks):
                        continue

                    landmark = landmarks[index]

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        str(index),
                        (x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 255),
                        1,
                    )

            cv2.imshow(
                "GazeSwitch - Eye Landmarks",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()