import torch

from data_loader import (
    load_processed_data,
    get_weekly_split,
    get_available_weeks,
    prepare_train_test_data,
    FEATURE_COLUMNS
)

from model import NFLTotalModel


EPOCHS = 300

TEST_SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026
]


def train_model(
    X_train,
    y_train
):
    torch.manual_seed(42)

    model = NFLTotalModel(
        input_size=len(
            FEATURE_COLUMNS
        )
    )

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for _ in range(EPOCHS):
        model.train()

        predictions = model(
            X_train
        )

        loss = criterion(
            predictions,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model


def evaluate_week(
    model,
    X_test,
    test_df
):
    model.eval()

    with torch.no_grad():
        residual_predictions = (
            model(
                X_test
            ).squeeze()
        )

    if residual_predictions.ndim == 0:
        residual_predictions = (
            residual_predictions.unsqueeze(0)
        )

    baseline_predictions = (
        torch.tensor(
            test_df[
                "baseline_total"
            ].values,
            dtype=torch.float32
        )
    )

    actual_totals = (
        torch.tensor(
            test_df[
                "total_points"
            ].values,
            dtype=torch.float32
        )
    )

    final_predictions = (
        baseline_predictions
        + residual_predictions
    )

    neural_errors = torch.abs(
        final_predictions
        - actual_totals
    )

    baseline_errors = torch.abs(
        baseline_predictions
        - actual_totals
    )

    return (
        final_predictions,
        baseline_predictions,
        actual_totals,
        neural_errors,
        baseline_errors
    )


def print_week_predictions(
    season,
    week,
    test_df,
    final_predictions,
    baseline_predictions,
    actual_totals,
    neural_errors,
    baseline_errors
):
    print(
        f"\n{'=' * 65}"
    )

    print(
        f"{season} - Week {week}"
    )

    print(
        f"{'=' * 65}"
    )

    games = test_df.reset_index(
        drop=True
    )

    for index, game in (
        games.iterrows()
    ):
        print(
            f"{game['AwayTeam']} "
            f"@ "
            f"{game['HomeTeam']}"
        )

        print(
            f"  Neural Prediction:   "
            f"{final_predictions[index].item():.1f}"
        )

        print(
            f"  Baseline Prediction: "
            f"{baseline_predictions[index].item():.1f}"
        )

        print(
            f"  Actual Total:        "
            f"{actual_totals[index].item():.0f}"
        )

        print(
            f"  Neural Error:        "
            f"{neural_errors[index].item():.1f}"
        )

        print(
            f"  Baseline Error:      "
            f"{baseline_errors[index].item():.1f}"
        )

        print()


def walk_forward():
    df = load_processed_data()

    all_neural_errors = []
    all_baseline_errors = []

    season_results = []

    for season in TEST_SEASONS:

        available_weeks = (
            get_available_weeks(
                df,
                season
            )
        )

        if not available_weeks:
            continue

        season_neural_errors = []
        season_baseline_errors = []

        print(
            f"\n{'#' * 65}"
        )

        print(
            f"TESTING SEASON {season}"
        )

        print(
            f"{'#' * 65}"
        )

        for week in available_weeks:

            train_df, test_df = (
                get_weekly_split(
                    df,
                    season,
                    week
                )
            )

            if len(test_df) == 0:
                continue

            if len(train_df) == 0:
                continue

            (
                X_train,
                X_test,
                y_train,
                y_test,
                scaler
            ) = prepare_train_test_data(
                train_df,
                test_df
            )

            model = train_model(
                X_train,
                y_train
            )

            (
                final_predictions,
                baseline_predictions,
                actual_totals,
                neural_errors,
                baseline_errors
            ) = evaluate_week(
                model,
                X_test,
                test_df
            )

            print_week_predictions(
                season,
                week,
                test_df,
                final_predictions,
                baseline_predictions,
                actual_totals,
                neural_errors,
                baseline_errors
            )

            week_neural_mae = (
                neural_errors
                .mean()
                .item()
            )

            week_baseline_mae = (
                baseline_errors
                .mean()
                .item()
            )

            print(
                f"Week {week} "
                f"Neural MAE: "
                f"{week_neural_mae:.2f}"
            )

            print(
                f"Week {week} "
                f"Baseline MAE: "
                f"{week_baseline_mae:.2f}"
            )

            season_neural_errors.extend(
                neural_errors.tolist()
            )

            season_baseline_errors.extend(
                baseline_errors.tolist()
            )

            all_neural_errors.extend(
                neural_errors.tolist()
            )

            all_baseline_errors.extend(
                baseline_errors.tolist()
            )

        if (
            season_neural_errors
            and season_baseline_errors
        ):
            season_neural_mae = (
                sum(
                    season_neural_errors
                )
                / len(
                    season_neural_errors
                )
            )

            season_baseline_mae = (
                sum(
                    season_baseline_errors
                )
                / len(
                    season_baseline_errors
                )
            )

            season_results.append(
                (
                    season,
                    season_neural_mae,
                    season_baseline_mae
                )
            )

            print(
                f"\n{'-' * 65}"
            )

            print(
                f"{season} Season Results"
            )

            print(
                f"{'-' * 65}"
            )

            print(
                f"Neural Network MAE: "
                f"{season_neural_mae:.2f}"
            )

            print(
                f"Baseline MAE: "
                f"{season_baseline_mae:.2f}"
            )

    print(
        f"\n{'=' * 65}"
    )

    print(
        "SEASON SUMMARY"
    )

    print(
        f"{'=' * 65}"
    )

    for (
        season,
        neural_mae,
        baseline_mae
    ) in season_results:

        difference = (
            baseline_mae
            - neural_mae
        )

        print(
            f"{season}: "
            f"Neural {neural_mae:.2f} | "
            f"Baseline {baseline_mae:.2f} | "
            f"Difference {difference:+.2f}"
        )

    if (
        all_neural_errors
        and all_baseline_errors
    ):
        overall_neural_mae = (
            sum(
                all_neural_errors
            )
            / len(
                all_neural_errors
            )
        )

        overall_baseline_mae = (
            sum(
                all_baseline_errors
            )
            / len(
                all_baseline_errors
            )
        )

        print(
            f"\n{'=' * 65}"
        )

        print(
            "OVERALL WEEKLY "
            "WALK-FORWARD RESULTS"
        )

        print(
            f"{'=' * 65}"
        )

        print(
            f"Games Tested: "
            f"{len(all_neural_errors)}"
        )

        print(
            f"Neural Network MAE: "
            f"{overall_neural_mae:.2f}"
        )

        print(
            f"Baseline MAE: "
            f"{overall_baseline_mae:.2f}"
        )

        improvement = (
            overall_baseline_mae
            - overall_neural_mae
        )

        print(
            f"MAE Improvement: "
            f"{improvement:+.2f}"
        )


if __name__ == "__main__":
    walk_forward()