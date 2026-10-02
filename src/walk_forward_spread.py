import numpy as np
import torch

from sklearn.preprocessing import StandardScaler

try:
    from src.data_loader import (
        load_processed_data,
        FEATURE_COLUMNS,
        get_weekly_split,
        get_available_weeks,
    )

    from src.spread_model import NFLSpreadModel

except ModuleNotFoundError:
    from data_loader import (
        load_processed_data,
        FEATURE_COLUMNS,
        get_weekly_split,
        get_available_weeks,
    )

    from spread_model import NFLSpreadModel


SPREAD_TARGET = (
    "target_margin_residual"
)


def train_model(
    X_train,
    y_train
):
    torch.manual_seed(42)

    model = NFLSpreadModel(
        input_size=len(
            FEATURE_COLUMNS
        )
    )

    loss_function = (
        torch.nn.MSELoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 300

    for _ in range(
        epochs
    ):
        model.train()

        optimizer.zero_grad()

        predictions = model(
            X_train
        )

        loss = loss_function(
            predictions,
            y_train
        )

        loss.backward()

        optimizer.step()

    return model


def format_spread(
    home_team,
    away_team,
    home_margin
):
    if home_margin > 0:

        return (
            f"{home_team} "
            f"-{abs(home_margin):.1f}"
        )

    if home_margin < 0:

        return (
            f"{away_team} "
            f"-{abs(home_margin):.1f}"
        )

    return "PICK"


def get_winner_from_margin(
    home_team,
    away_team,
    margin
):
    if margin > 0:
        return home_team

    if margin < 0:
        return away_team

    return "TIE"


def walk_forward_spread(
    start_season=2010,
    end_season=2026
):
    df = load_processed_data()

    all_neural_errors = []
    all_baseline_errors = []

    neural_winner_correct = 0
    baseline_winner_correct = 0

    total_non_tie_games = 0

    print(
        "\n=============================="
    )

    print(
        "NFL RESIDUAL SPREAD "
        "WALK-FORWARD TEST"
    )

    print(
        "=============================="
    )

    for season in range(
        start_season,
        end_season + 1
    ):

        season_neural_errors = []
        season_baseline_errors = []

        season_neural_wins = 0
        season_baseline_wins = 0

        season_non_tie_games = 0

        weeks = get_available_weeks(
            df,
            season
        )

        print(
            f"\n\n========== "
            f"{season} "
            f"=========="
        )

        for week in weeks:

            train_df, test_df = (
                get_weekly_split(
                    df,
                    season,
                    week
                )
            )

            if (
                train_df.empty
                or test_df.empty
            ):
                continue

            # -------------------------
            # SCALE FEATURES
            # -------------------------

            scaler = StandardScaler()

            X_train = scaler.fit_transform(
                train_df[
                    FEATURE_COLUMNS
                ]
            )

            X_test = scaler.transform(
                test_df[
                    FEATURE_COLUMNS
                ]
            )

            y_train = train_df[
                SPREAD_TARGET
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

            # -------------------------
            # TRAIN MODEL
            # -------------------------

            model = train_model(
                X_train,
                y_train
            )

            model.eval()

            with torch.no_grad():

                predicted_residuals = (
                    model(
                        X_test
                    )
                    .squeeze(1)
                    .numpy()
                )

            # -------------------------
            # CREATE FINAL MARGINS
            # -------------------------

            baseline_margins = (
                test_df[
                    "baseline_margin"
                ].values
            )

            predicted_margins = (
                baseline_margins
                + predicted_residuals
            )

            actual_margins = (
                test_df[
                    "home_margin"
                ].values
            )

            week_neural_errors = []
            week_baseline_errors = []

            print(
                f"\n{season} "
                f"Week {week}"
            )

            # -------------------------
            # GAME RESULTS
            # -------------------------

            for i, (_, game) in enumerate(
                test_df.iterrows()
            ):

                home_team = (
                    game["HomeTeam"]
                )

                away_team = (
                    game["AwayTeam"]
                )

                neural_adjustment = float(
                    predicted_residuals[i]
                )

                predicted_margin = float(
                    predicted_margins[i]
                )

                baseline_margin = float(
                    baseline_margins[i]
                )

                actual_margin = float(
                    actual_margins[i]
                )

                neural_error = abs(
                    predicted_margin
                    - actual_margin
                )

                baseline_error = abs(
                    baseline_margin
                    - actual_margin
                )

                week_neural_errors.append(
                    neural_error
                )

                week_baseline_errors.append(
                    baseline_error
                )

                season_neural_errors.append(
                    neural_error
                )

                season_baseline_errors.append(
                    baseline_error
                )

                all_neural_errors.append(
                    neural_error
                )

                all_baseline_errors.append(
                    baseline_error
                )

                actual_winner = (
                    get_winner_from_margin(
                        home_team,
                        away_team,
                        actual_margin
                    )
                )

                neural_winner = (
                    get_winner_from_margin(
                        home_team,
                        away_team,
                        predicted_margin
                    )
                )

                baseline_winner = (
                    get_winner_from_margin(
                        home_team,
                        away_team,
                        baseline_margin
                    )
                )

                if actual_winner != "TIE":

                    total_non_tie_games += 1

                    season_non_tie_games += 1

                    if (
                        neural_winner
                        == actual_winner
                    ):
                        neural_winner_correct += 1

                        season_neural_wins += 1

                    if (
                        baseline_winner
                        == actual_winner
                    ):
                        baseline_winner_correct += 1

                        season_baseline_wins += 1

                predicted_spread = (
                    format_spread(
                        home_team,
                        away_team,
                        predicted_margin
                    )
                )

                baseline_spread = (
                    format_spread(
                        home_team,
                        away_team,
                        baseline_margin
                    )
                )

                actual_spread = (
                    format_spread(
                        home_team,
                        away_team,
                        actual_margin
                    )
                )

                print(
                    f"{away_team} "
                    f"@ {home_team}"
                )

                print(
                    f"  Baseline Spread:    "
                    f"{baseline_spread}"
                )

                print(
                    f"  Neural Adjustment:  "
                    f"{neural_adjustment:+.2f}"
                )

                print(
                    f"  Neural Spread:      "
                    f"{predicted_spread}"
                )

                print(
                    f"  Actual Margin:      "
                    f"{actual_spread}"
                )

                print(
                    f"  Neural Error:       "
                    f"{neural_error:.2f}"
                )

                print(
                    f"  Baseline Error:     "
                    f"{baseline_error:.2f}"
                )

            # -------------------------
            # WEEK RESULTS
            # -------------------------

            if week_neural_errors:

                neural_week_mae = (
                    np.mean(
                        week_neural_errors
                    )
                )

                baseline_week_mae = (
                    np.mean(
                        week_baseline_errors
                    )
                )

                print(
                    f"\nWeek {week} "
                    f"Neural MAE: "
                    f"{neural_week_mae:.2f}"
                )

                print(
                    f"Week {week} "
                    f"Baseline MAE: "
                    f"{baseline_week_mae:.2f}"
                )

        # -------------------------
        # SEASON RESULTS
        # -------------------------

        if season_neural_errors:

            season_neural_mae = (
                np.mean(
                    season_neural_errors
                )
            )

            season_baseline_mae = (
                np.mean(
                    season_baseline_errors
                )
            )

            if (
                season_non_tie_games > 0
            ):

                season_neural_accuracy = (
                    season_neural_wins
                    / season_non_tie_games
                    * 100
                )

                season_baseline_accuracy = (
                    season_baseline_wins
                    / season_non_tie_games
                    * 100
                )

            else:

                season_neural_accuracy = 0

                season_baseline_accuracy = 0

            print(
                f"\n===== "
                f"{season} "
                f"RESULTS ====="
            )

            print(
                f"Neural Margin MAE: "
                f"{season_neural_mae:.2f}"
            )

            print(
                f"Baseline Margin MAE: "
                f"{season_baseline_mae:.2f}"
            )

            print(
                f"Neural Winner Accuracy: "
                f"{season_neural_accuracy:.2f}%"
            )

            print(
                f"Baseline Winner Accuracy: "
                f"{season_baseline_accuracy:.2f}%"
            )

    # -------------------------
    # OVERALL RESULTS
    # -------------------------

    print(
        "\n\n=============================="
    )

    print(
        "OVERALL SPREAD RESULTS"
    )

    print(
        "=============================="
    )

    if all_neural_errors:

        neural_mae = np.mean(
            all_neural_errors
        )

        baseline_mae = np.mean(
            all_baseline_errors
        )

        print(
            f"Games Tested: "
            f"{len(all_neural_errors)}"
        )

        print(
            f"Neural Margin MAE: "
            f"{neural_mae:.2f}"
        )

        print(
            f"Baseline Margin MAE: "
            f"{baseline_mae:.2f}"
        )

        print(
            f"MAE Improvement: "
            f"{baseline_mae - neural_mae:+.2f}"
        )

        if total_non_tie_games > 0:

            neural_accuracy = (
                neural_winner_correct
                / total_non_tie_games
                * 100
            )

            baseline_accuracy = (
                baseline_winner_correct
                / total_non_tie_games
                * 100
            )

            print(
                f"\nNeural Winner Accuracy: "
                f"{neural_accuracy:.2f}%"
            )

            print(
                f"Baseline Winner Accuracy: "
                f"{baseline_accuracy:.2f}%"
            )


def main():
    walk_forward_spread(
        start_season=2010,
        end_season=2026
    )


if __name__ == "__main__":
    main()