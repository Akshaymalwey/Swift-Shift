import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


MODEL_FILE = "models/gaze_classifier.joblib"
TEST_FILE = "data/gaze_samples_test.csv"

FEATURES = [
    "yaw",
    "pitch",
    "roll",
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
]


def main():
    print("Loading model...")

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(TEST_FILE)

    X_test = df[FEATURES]
    y_test = df["target"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print("==============================")
    print("Fresh Dataset Evaluation")
    print("==============================")
    print()

    print(
        f"Test samples: {len(df)}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print()
    print("Classification Report:")
    print()

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

    print("Confusion Matrix:")
    print()

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["mac", "monitor"],
        )
    )


if __name__ == "__main__":
    main()