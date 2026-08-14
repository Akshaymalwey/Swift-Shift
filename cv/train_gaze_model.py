import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_FILE = "data/gaze_samples.csv"


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

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df["target"]

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        model,
        "models/gaze_classifier.joblib",
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print("==============================")
    print("Gaze Model Evaluation")
    print("==============================")
    print()

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