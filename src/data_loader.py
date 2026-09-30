import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


PROCESSED_DATA_PATH = "data/processed/2010-2026_scores_processed.csv"

FEATURE_COLUMNS = [
    "home_ppg",
    "home_points_allowed",
    "home_win_pct",
    "home_last_3_ppg",
    "away_ppg",
    "away_points_allowed",
    "away_win_pct",
    "away_last_3_ppg",
]

TARGET_COLUMN = "total_points"


def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


def split_data(df):
    train_df = df[df["Season"] <= 2023].copy()
    val_df = df[df["Season"] == 2024].copy()
    test_df = df[df["Season"] >= 2025].copy()

    return train_df, val_df, test_df


def prepare_data(train_df, val_df, test_df):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    X_val = scaler.transform(val_df[FEATURE_COLUMNS])
    X_test = scaler.transform(test_df[FEATURE_COLUMNS])

    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

if __name__ == "__main__":
    df = load_processed_data()

    train_df, val_df, test_df = split_data(df)

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))

    print("\nTrain seasons:")
    print(train_df["Season"].min(), "to", train_df["Season"].max())

    print("\nValidation seasons:")
    print(val_df["Season"].unique())

    print("\nTest seasons:")
    print(test_df["Season"].unique())

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        train_df,
        val_df,
        test_df
    )

    print("\nTensor shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)