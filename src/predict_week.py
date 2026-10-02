import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

try:
    from src.data_loader import FEATURE_COLUMNS
    from src.model import NFLTotalModel
    from src.spread_model import NFLSpreadModel

    from src.preprocess import (
        load_data,
        clean_regular_season_games,
        create_team_history,
        get_team_features,
        update_team_history,
    )

except ModuleNotFoundError:
    from data_loader import FEATURE_COLUMNS
    from model import NFLTotalModel
    from spread_model import NFLSpreadModel

    from preprocess import (
        load_data,
        clean_regular_season_games,
        create_team_history,
        get_team_features,
        update_team_history,
    )


BASE_DIR = Path(__file__).resolve().parent.parent

UPCOMING_PATH = (
    BASE_DIR
    / "data"
    / "upcoming"
    / "upcoming_games.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "upcoming"
    / "predictions.csv"
)

TOTAL_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nfl_current_model.pth"
)

TOTAL_SCALER_PATH = (
    BASE_DIR
    / "models"
    / "nfl_current_scaler.pkl"
)

SPREAD_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nfl_spread_model.pth"
)

SPREAD_SCALER_PATH = (
    BASE_DIR
    / "models"
    / "nfl_spread_scaler.pkl"
)


def build_team_histories(
    games_df,
    season,
    prediction_week,
):
    team_histories = defaultdict(
        create_team_history
    )

    season_games = games_df[
        (games_df["Season"] == season)
        & (games_df["Week"] < prediction_week)
    ].copy()

    season_games = season_games.sort_values(
        ["Week"]
    )

    for _, game in season_games.iterrows():

        home_team = game["HomeTeam"]
        away_team = game["AwayTeam"]

        home_score = int(
            game["HomeScore"]
        )

        away_score = int(
            game["AwayScore"]
        )

        update_team_history(
            team_histories[home_team],
            home_score,
            away_score,
        )

        update_team_history(
            team_histories[away_team],
            away_score,
            home_score,
        )

    return team_histories


def create_feature_row(
    week,
    home_features,
    away_features,
):
    return {
        "Week": week,

        "home_ppg":
            home_features["ppg"],

        "home_points_allowed":
            home_features[
                "points_allowed"
            ],

        "home_win_pct":
            home_features[
                "win_pct"
            ],

        "home_last_3_ppg":
            home_features[
                "last_3_ppg"
            ],

        "home_last_3_points_allowed":
            home_features[
                "last_3_points_allowed"
            ],

        "home_last_5_ppg":
            home_features[
                "last_5_ppg"
            ],

        "home_last_5_points_allowed":
            home_features[
                "last_5_points_allowed"
            ],

        "home_avg_total_points":
            home_features[
                "avg_total_points"
            ],

        "away_ppg":
            away_features["ppg"],

        "away_points_allowed":
            away_features[
                "points_allowed"
            ],

        "away_win_pct":
            away_features[
                "win_pct"
            ],

        "away_last_3_ppg":
            away_features[
                "last_3_ppg"
            ],

        "away_last_3_points_allowed":
            away_features[
                "last_3_points_allowed"
            ],

        "away_last_5_ppg":
            away_features[
                "last_5_ppg"
            ],

        "away_last_5_points_allowed":
            away_features[
                "last_5_points_allowed"
            ],

        "away_avg_total_points":
            away_features[
                "avg_total_points"
            ],
    }


def calculate_baseline_total(
    home_features,
    away_features,
):
    return (
        home_features["ppg"]
        + away_features["ppg"]
    )


def calculate_baseline_margin(
    home_features,
    away_features,
):
    home_scoring_diff = (
        home_features["ppg"]
        - home_features["points_allowed"]
    )

    away_scoring_diff = (
        away_features["ppg"]
        - away_features["points_allowed"]
    )

    return (
        home_scoring_diff
        - away_scoring_diff
    )


def format_spread(
    home_team,
    away_team,
    home_margin,
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


def get_predicted_winner(
    home_team,
    away_team,
    home_margin,
):
    if home_margin > 0:
        return home_team

    if home_margin < 0:
        return away_team

    return "TIE"


def load_models():
    total_model = NFLTotalModel(
        input_size=len(FEATURE_COLUMNS)
    )

    total_model.load_state_dict(
        torch.load(
            TOTAL_MODEL_PATH,
            map_location="cpu",
        )
    )

    total_model.eval()

    spread_model = NFLSpreadModel(
        input_size=len(FEATURE_COLUMNS)
    )

    spread_model.load_state_dict(
        torch.load(
            SPREAD_MODEL_PATH,
            map_location="cpu",
        )
    )

    spread_model.eval()

    with open(
        TOTAL_SCALER_PATH,
        "rb",
    ) as file:
        total_scaler = pickle.load(file)

    with open(
        SPREAD_SCALER_PATH,
        "rb",
    ) as file:
        spread_scaler = pickle.load(file)

    return (
        total_model,
        total_scaler,
        spread_model,
        spread_scaler,
    )


def predict_week():
    print("\n==============================")
    print("NFL WEEKLY PREDICTIONS")
    print("==============================")

    upcoming_df = pd.read_csv(
        UPCOMING_PATH
    )

    if upcoming_df.empty:
        print(
            "No upcoming games found."
        )
        return

    raw_df = load_data()

    games_df = (
        clean_regular_season_games(
            raw_df
        )
    )

    (
        total_model,
        total_scaler,
        spread_model,
        spread_scaler,
    ) = load_models()

    predictions = []

    grouped_games = upcoming_df.groupby(
        ["Season", "Week"]
    )

    for (
        season,
        week
    ), week_games in grouped_games:

        season = int(season)
        week = int(week)

        print(
            f"\n{season} Week {week}"
        )

        team_histories = (
            build_team_histories(
                games_df,
                season,
                week,
            )
        )

        for _, game in week_games.iterrows():

            away_team = (
                game["AwayTeam"]
            )

            home_team = (
                game["HomeTeam"]
            )

            home_features = (
                get_team_features(
                    team_histories[
                        home_team
                    ]
                )
            )

            away_features = (
                get_team_features(
                    team_histories[
                        away_team
                    ]
                )
            )

            if (
                home_features is None
                or away_features is None
            ):
                print(
                    f"\nSkipping "
                    f"{away_team} @ "
                    f"{home_team}"
                )

                print(
                    "Not enough previous "
                    "season data."
                )

                continue

            feature_row = (
                create_feature_row(
                    week,
                    home_features,
                    away_features,
                )
            )

            feature_df = pd.DataFrame(
                [feature_row]
            )

            feature_df = feature_df[
                FEATURE_COLUMNS
            ]

            # -------------------------
            # TOTAL PREDICTION
            # -------------------------

            baseline_total = (
                calculate_baseline_total(
                    home_features,
                    away_features,
                )
            )

            total_scaled = (
                total_scaler.transform(
                    feature_df
                )
            )

            total_tensor = torch.tensor(
                total_scaled,
                dtype=torch.float32,
            )

            with torch.no_grad():

                total_adjustment = (
                    total_model(
                        total_tensor
                    )
                    .item()
                )

            predicted_total = (
                baseline_total
                + total_adjustment
            )

            # -------------------------
            # SPREAD PREDICTION
            # -------------------------

            baseline_margin = (
                calculate_baseline_margin(
                    home_features,
                    away_features,
                )
            )

            spread_scaled = (
                spread_scaler.transform(
                    feature_df
                )
            )

            spread_tensor = torch.tensor(
                spread_scaled,
                dtype=torch.float32,
            )

            with torch.no_grad():

                spread_adjustment = (
                    spread_model(
                        spread_tensor
                    )
                    .item()
                )

            predicted_home_margin = (
                baseline_margin
                + spread_adjustment
            )

            predicted_spread = (
                format_spread(
                    home_team,
                    away_team,
                    predicted_home_margin,
                )
            )

            predicted_winner = (
                get_predicted_winner(
                    home_team,
                    away_team,
                    predicted_home_margin,
                )
            )

            # -------------------------
            # DISPLAY
            # -------------------------

            print(
                f"\n{away_team} @ "
                f"{home_team}"
            )

            print(
                f"  Baseline Total:       "
                f"{baseline_total:.2f}"
            )

            print(
                f"  Total Adjustment:     "
                f"{total_adjustment:+.2f}"
            )

            print(
                f"  Predicted Total:      "
                f"{predicted_total:.2f}"
            )

            print()

            print(
                f"  Baseline Margin:      "
                f"{baseline_margin:+.2f}"
            )

            print(
                f"  Spread Adjustment:    "
                f"{spread_adjustment:+.2f}"
            )

            print(
                f"  Predicted Spread:     "
                f"{predicted_spread}"
            )

            print(
                f"  Predicted Winner:     "
                f"{predicted_winner}"
            )

            predictions.append(
                {
                    "Season":
                        season,

                    "Week":
                        week,

                    "AwayTeam":
                        away_team,

                    "HomeTeam":
                        home_team,

                    "BaselineTotal":
                        round(
                            baseline_total,
                            2,
                        ),

                    "TotalAdjustment":
                        round(
                            total_adjustment,
                            2,
                        ),

                    "PredictedTotal":
                        round(
                            predicted_total,
                            2,
                        ),

                    "BaselineMargin":
                        round(
                            baseline_margin,
                            2,
                        ),

                    "SpreadAdjustment":
                        round(
                            spread_adjustment,
                            2,
                        ),

                    "PredictedHomeMargin":
                        round(
                            predicted_home_margin,
                            2,
                        ),

                    "PredictedSpread":
                        predicted_spread,

                    "PredictedWinner":
                        predicted_winner,
                }
            )

    predictions_df = pd.DataFrame(
        predictions
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        "\n=============================="
    )

    print(
        "Predictions saved to:"
    )

    print(
        PREDICTIONS_PATH
    )

    print(
        "==============================\n"
    )


def main():
    predict_week()


if __name__ == "__main__":
    main()