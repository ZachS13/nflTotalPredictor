import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler


PROCESSED_DATA_PATH = "data/processed/2010-2026_scores_processed.csv"


FEATURE_COLUMNS = [
    "Week",

    "home_ppg",
    "home_points_allowed",
    "home_win_pct",
    "home_last_3_ppg",
    "home_last_3_points_allowed",
    "home_last_5_ppg",
    "home_last_5_points_allowed",
    "home_avg_total_points",

    "away_ppg",
    "away_points_allowed",
    "away_win_pct",
    "away_last_3_ppg",
    "away_last_3_points_allowed",
    "away_last_5_ppg",
    "away_last_5_points_allowed",
    "away_avg_total_points",
]


TARGET_COLUMN = "target_residual"


def load_processed_data():
    return pd.read_csv(
        PROCESSED_DATA_PATH
    )


def get_weekly_split(
    df,
    season,
    week
):
    previous_seasons = df[
        df["Season"] < season
    ]

    current_season_previous_weeks = df[
        (df["Season"] == season)
        & (df["Week"] < week)
    ]

    train_df = pd.concat(
        [
            previous_seasons,
            current_season_previous_weeks
        ],
        ignore_index=True
    )

    test_df = df[
        (df["Season"] == season)
        & (df["Week"] == week)
    ].copy()

    return (
        train_df,
        test_df
    )


def prepare_train_test_data(
    train_df,
    test_df
):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_df[FEATURE_COLUMNS]
    )

    X_test = scaler.transform(
        test_df[FEATURE_COLUMNS]
    )

    y_train = train_df[
        TARGET_COLUMN
    ].values

    y_test = test_df[
        TARGET_COLUMN
    ].values

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).unsqueeze(1)

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    ).unsqueeze(1)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


def get_available_weeks(
    df,
    season
):
    weeks = df[
        df["Season"] == season
    ]["Week"].unique()

    return sorted(
        int(week)
        for week in weeks
    )


if __name__ == "__main__":
    df = load_processed_data()

    print(
        "Dataset rows:",
        len(df)
    )

    print(
        "\nAvailable seasons:"
    )

    print(
        sorted(
            df["Season"].unique()
        )
    )

    print(
        "\nFeature count:",
        len(FEATURE_COLUMNS)
    )

    example_season = 2025

    print(
        f"\nAvailable weeks in "
        f"{example_season}:"
    )

    print(
        get_available_weeks(
            df,
            example_season
        )
    )

    example_week = 10

    train_df, test_df = (
        get_weekly_split(
            df,
            example_season,
            example_week
        )
    )

    print(
        f"\nExample Prediction:"
        f" {example_season} Week "
        f"{example_week}"
    )

    print(
        "Training rows:",
        len(train_df)
    )

    print(
        "Test rows:",
        len(test_df)
    )

    print(
        "Latest training season:",
        train_df["Season"].max()
    )

    current_season_train = train_df[
        train_df["Season"]
        == example_season
    ]

    print(
        f"{example_season} games "
        f"already available for training:",
        len(current_season_train)
    )

    if len(
        current_season_train
    ) > 0:
        print(
            "Latest current-season "
            "training week:",
            current_season_train[
                "Week"
            ].max()
        )