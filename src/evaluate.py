import torch

from data_loader import (
    load_processed_data,
    split_data,
    prepare_data,
    FEATURE_COLUMNS
)

from model import NFLTotalModel


def evaluate_model():
    df = load_processed_data()

    train_df, val_df, test_df = split_data(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = prepare_data(
        train_df,
        val_df,
        test_df
    )

    model = NFLTotalModel(
        input_size=len(FEATURE_COLUMNS)
    )

    model.load_state_dict(
        torch.load("models/nfl_total_model.pth")
    )

    model.eval()

    with torch.no_grad():
        predictions = model(X_test)

        mae = torch.mean(
            torch.abs(predictions - y_test)
        )

        mse = torch.mean(
            (predictions - y_test) ** 2
        )

        rmse = torch.sqrt(mse)

    # Simple baseline: add each team's scoring average
    baseline_predictions = (
        test_df["home_ppg"] +
        test_df["away_ppg"]
    )

    baseline_actual = test_df["total_points"]

    baseline_mae = (
        baseline_predictions - baseline_actual
    ).abs().mean()

    print(f"Test Games: {len(y_test)}")
    print(f"Neural Network MAE: {mae.item():.2f}")
    print(f"Neural Network RMSE: {rmse.item():.2f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")


if __name__ == "__main__":
    evaluate_model()