import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "data/gaze_samples.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    print("\nDataset:")
    print(df.head())

    print("\nSample counts:")
    print(df["target"].value_counts())

    print("\nStatistics:")
    print(
        df.groupby("target")[
            [
                "yaw",
                "pitch",
                "roll",
                "left_iris_x",
                "right_iris_x",
                "avg_iris_x",
                "avg_iris_y",
            ]
        ].agg(["mean", "std", "min", "max"])
    )

    # -----------------------------
    # Yaw distribution
    # -----------------------------

    plt.figure(figsize=(10, 6))

    for target in ["mac", "monitor"]:
        subset = df[df["target"] == target]

        plt.hist(
            subset["yaw"],
            bins=30,
            alpha=0.6,
            label=target,
        )

    plt.xlabel("Yaw")
    plt.ylabel("Number of samples")
    plt.title("Head Pose: Yaw Distribution")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()

    # -----------------------------
    # Iris X distribution
    # -----------------------------

    plt.figure(figsize=(10, 6))

    for target in ["mac", "monitor"]:
        subset = df[df["target"] == target]

        plt.hist(
            subset["avg_iris_x"],
            bins=30,
            alpha=0.6,
            label=target,
        )

    plt.xlabel("Average Iris X")
    plt.ylabel("Number of samples")
    plt.title("Eye Gaze: Iris X Distribution")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()

    # -----------------------------
    # Yaw vs Iris X
    # -----------------------------

    plt.figure(figsize=(10, 6))

    for target in ["mac", "monitor"]:
        subset = df[df["target"] == target]

        plt.scatter(
            subset["yaw"],
            subset["avg_iris_x"],
            alpha=0.5,
            label=target,
        )

    plt.xlabel("Yaw")
    plt.ylabel("Average Iris X")
    plt.title("Yaw vs Iris Position")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()


if __name__ == "__main__":
    main()