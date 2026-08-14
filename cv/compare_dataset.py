import pandas as pd


TRAIN_FILE = "data/gaze_samples.csv"
TEST_FILE = "data/gaze_samples_test.csv"


FEATURES = [
    "yaw",
    "pitch",
    "roll",
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
    "avg_iris_x",
    "avg_iris_y",
]


def main():

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print()
    print("==============================================")
    print("Dataset Comparison")
    print("==============================================")

    for target in ["mac", "monitor"]:

        print()
        print(f"TARGET: {target}")
        print("-" * 60)

        train_target = train[
            train["target"] == target
        ]

        test_target = test[
            test["target"] == target
        ]

        for feature in FEATURES:

            train_mean = train_target[feature].mean()
            train_std = train_target[feature].std()

            test_mean = test_target[feature].mean()
            test_std = test_target[feature].std()

            print(
                f"{feature:15s} "
                f"Train: {train_mean:7.3f} ± {train_std:6.3f}   "
                f"Test: {test_mean:7.3f} ± {test_std:6.3f}"
            )


if __name__ == "__main__":
    main()