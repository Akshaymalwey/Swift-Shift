import cv2
import mediapipe as mp


MODEL_PATH = "models/face_landmarker.task"


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Camera opened successfully.")
    print("Press 'q' to quit.")

    # MediaPipe Tasks API
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

    frame_timestamp_ms = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame.")
                break

            # OpenCV: BGR
            # MediaPipe: SRGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            frame_timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms
            )

            # Draw face landmarks
            if result.face_landmarks:

                for face_landmarks in result.face_landmarks:

                    h, w, _ = frame.shape

                    for landmark in face_landmarks:

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )

            cv2.imshow(
                "Swift Switch - Face Tracking",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()