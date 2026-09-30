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

    train_df, val_df, test_df = split_data(
        df
    )

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
        input_size=len(
            FEATURE_COLUMNS
        )
    )

    model.load_state_dict(
        torch.load(
            "models/nfl_total_model.pth"
        )
    )

    model.eval()

    with torch.no_grad():

        residual_predictions = model(
            X_test
        ).squeeze()

    baseline_predictions = torch.tensor(
        test_df[
            "baseline_total"
        ].values,
        dtype=torch.float32
    )

    actual_totals = torch.tensor(
        test_df[
            "total_points"
        ].values,
        dtype=torch.float32
    )

    final_predictions = (
        baseline_predictions
        + residual_predictions
    )

    neural_mae = torch.mean(
        torch.abs(
            final_predictions
            - actual_totals
        )
    )

    neural_mse = torch.mean(
        (
            final_predictions
            - actual_totals
        ) ** 2
    )

    neural_rmse = torch.sqrt(
        neural_mse
    )

    baseline_mae = torch.mean(
        torch.abs(
            baseline_predictions
            - actual_totals
        )
    )

    print(
        f"Test Games: "
        f"{len(actual_totals)}"
    )

    print(
        f"Residual Neural Network MAE: "
        f"{neural_mae.item():.2f}"
    )

    print(
        f"Residual Neural Network RMSE: "
        f"{neural_rmse.item():.2f}"
    )

    print(
        f"Baseline MAE: "
        f"{baseline_mae.item():.2f}"
    )


if __name__ == "__main__":
    evaluate_model()