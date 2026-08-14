import pandas as pd
import numpy as np


DATA_FILE = "data/gaze_samples.csv"


def evaluate_threshold(df, threshold):
    predictions = np.where(
        df["yaw"] < threshold,
        "monitor",
        "mac",
    )

    correct = predictions == df["target"]

    return correct.mean()


def main():
    df = pd.read_csv(DATA_FILE)

    best_threshold = None
    best_accuracy = 0.0

    # Test thresholds across the observed range.
    thresholds = np.arange(
        df["yaw"].min(),
        df["yaw"].max(),
        0.1,
    )

    for threshold in thresholds:

        accuracy = evaluate_threshold(
            df,
            threshold,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print()
    print("==============================")
    print("Yaw Threshold Analysis")
    print("==============================")
    print()

    print(f"Best threshold : {best_threshold:.2f}")
    print(f"Accuracy       : {best_accuracy * 100:.2f}%")
    print()

    # Show nearby thresholds.
    print("Nearby thresholds:")
    print()

    for threshold in np.arange(
        best_threshold - 5,
        best_threshold + 5.1,
        1,
    ):

        accuracy = evaluate_threshold(
            df,
            threshold,
        )

        print(
            f"{threshold:7.2f}° → "
            f"{accuracy * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()