import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

try:
    from src.data_loader import FEATURE_COLUMNS
    from src.model import NFLTotalModel

    from src.preprocess import (
        load_data,
        clean_regular_season_games,
        create_team_history,
        get_team_features,
        update_team_history
    )

except ModuleNotFoundError:
    from data_loader import FEATURE_COLUMNS
    from model import NFLTotalModel

    from preprocess import (
        load_data,
        clean_regular_season_games,
        create_team_history,
        get_team_features,
        update_team_history
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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nfl_current_model.pth"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "nfl_current_scaler.pkl"
)


def build_team_histories(
    games,
    season,
    prediction_week
):
    team_history = defaultdict(
        create_team_history
    )

    season_games = games[
        (games["Season"] == season)
        & (games["Week"] < prediction_week)
    ].copy()

    season_games = season_games.sort_values(
        "Week"
    )

    for _, game in season_games.iterrows():

        home_team = game["HomeTeam"]
        away_team = game["AwayTeam"]

        update_team_history(
            team_history[home_team],
            game["HomeScore"],
            game["AwayScore"]
        )

        update_team_history(
            team_history[away_team],
            game["AwayScore"],
            game["HomeScore"]
        )

    return team_history


def build_prediction_row(
    season,
    week,
    home_team,
    away_team,
    team_history
):
    home_features = get_team_features(
        team_history[home_team]
    )

    away_features = get_team_features(
        team_history[away_team]
    )

    if home_features is None:
        raise ValueError(
            f"No previous games found "
            f"for {home_team}"
        )

    if away_features is None:
        raise ValueError(
            f"No previous games found "
            f"for {away_team}"
        )

    baseline_total = (
        home_features["ppg"]
        + away_features["ppg"]
    )

    row = {
        "Week": week,

        "home_ppg":
            home_features["ppg"],

        "home_points_allowed":
            home_features[
                "points_allowed"
            ],

        "home_win_pct":
            home_features["win_pct"],

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
            away_features["win_pct"],

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
            ]
    }

    return row, baseline_total


def load_model():
    model = NFLTotalModel(
        input_size=len(
            FEATURE_COLUMNS
        )
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            weights_only=True
        )
    )

    model.eval()

    return model


def load_scaler():
    with open(
        SCALER_PATH,
        "rb"
    ) as file:
        return pickle.load(file)


def predict_week():
    upcoming_games = pd.read_csv(
        UPCOMING_PATH
    )

    raw_games = load_data()

    completed_games = (
        clean_regular_season_games(
            raw_games
        )
    )

    model = load_model()
    scaler = load_scaler()

    predictions_output = []

    for (
        season,
        week
    ), week_games in upcoming_games.groupby(
        [
            "Season",
            "Week"
        ]
    ):

        team_history = (
            build_team_histories(
                completed_games,
                season,
                week
            )
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"{season} Week {week}"
        )

        print(
            f"{'=' * 60}"
        )

        for _, game in week_games.iterrows():

            away_team = game[
                "AwayTeam"
            ]

            home_team = game[
                "HomeTeam"
            ]

            feature_row, baseline = (
                build_prediction_row(
                    season,
                    week,
                    home_team,
                    away_team,
                    team_history
                )
            )

            feature_df = pd.DataFrame(
                [feature_row]
            )

            feature_df = feature_df[
                FEATURE_COLUMNS
            ]

            scaled_features = (
                scaler.transform(
                    feature_df
                )
            )

            X = torch.tensor(
                scaled_features,
                dtype=torch.float32
            )

            with torch.no_grad():
                residual = (
                    model(X)
                    .item()
                )

            predicted_total = (
                baseline
                + residual
            )

            print(
                f"\n{away_team} "
                f"@ {home_team}"
            )

            print(
                f"  Baseline Total:     "
                f"{baseline:.1f}"
            )

            print(
                f"  Neural Adjustment: "
                f"{residual:+.1f}"
            )

            print(
                f"  Predicted Total:   "
                f"{predicted_total:.1f}"
            )

            predictions_output.append({
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
                        baseline,
                        2
                    ),

                "NeuralAdjustment":
                    round(
                        residual,
                        2
                    ),

                "PredictedTotal":
                    round(
                        predicted_total,
                        2
                    )
            })

    predictions_df = pd.DataFrame(
        predictions_output
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print(
        f"\nPredictions saved to:"
        f"\n{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    predict_week()