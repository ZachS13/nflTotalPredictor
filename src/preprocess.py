import pandas as pd
from collections import defaultdict, deque
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "2010-2026_scores.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "2010-2026_scores_processed.csv"


def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def clean_regular_season_games(df):
    df = df.copy()

    # Keep completed games only
    df = df[df["GameStatus"] == "FINAL"]

    # Keep regular-season weeks only
    df = df[df["Week"].astype(str).str.match(r"^WEEK \d+$")]

    # Convert "WEEK 4" -> 4
    df["Week"] = (
        df["Week"]
        .str.replace("WEEK ", "", regex=False)
        .astype(int)
    )

    # Remove rows missing required game data
    df = df.dropna(
        subset=[
            "Season",
            "Week",
            "HomeTeam",
            "AwayTeam",
            "HomeScore",
            "AwayScore",
        ]
    )

    df["Season"] = df["Season"].astype(int)
    df["Week"] = df["Week"].astype(int)
    df["HomeScore"] = df["HomeScore"].astype(int)
    df["AwayScore"] = df["AwayScore"].astype(int)

    # Keep games in chronological order
    df = df.sort_values(
        by=["Season", "Week"]
    ).reset_index(drop=True)

    return df


def create_team_history():
    return {
        "games": 0,
        "points_scored": 0,
        "points_allowed": 0,
        "wins": 0,
        "recent_points": deque(maxlen=5),
        "recent_points_allowed": deque(maxlen=5),
    }


def get_team_features(history):
    games = history["games"]

    if games == 0:
        return None

    ppg = (
        history["points_scored"]
        / games
    )

    points_allowed = (
        history["points_allowed"]
        / games
    )

    win_pct = (
        history["wins"]
        / games
    )

    recent_points = list(
        history["recent_points"]
    )

    recent_points_allowed = list(
        history["recent_points_allowed"]
    )

    last_3_points = recent_points[-3:]
    last_3_allowed = recent_points_allowed[-3:]

    last_5_points = recent_points[-5:]
    last_5_allowed = recent_points_allowed[-5:]

    last_3_ppg = (
        sum(last_3_points)
        / len(last_3_points)
    )

    last_3_points_allowed = (
        sum(last_3_allowed)
        / len(last_3_allowed)
    )

    last_5_ppg = (
        sum(last_5_points)
        / len(last_5_points)
    )

    last_5_points_allowed = (
        sum(last_5_allowed)
        / len(last_5_allowed)
    )

    avg_total_points = (
        history["points_scored"]
        + history["points_allowed"]
    ) / games

    return {
        "ppg": ppg,
        "points_allowed": points_allowed,
        "win_pct": win_pct,
        "last_3_ppg": last_3_ppg,
        "last_3_points_allowed": last_3_points_allowed,
        "last_5_ppg": last_5_ppg,
        "last_5_points_allowed": last_5_points_allowed,
        "avg_total_points": avg_total_points,
    }


def update_team_history(
    history,
    points_scored,
    points_allowed,
):
    history["games"] += 1
    history["points_scored"] += points_scored
    history["points_allowed"] += points_allowed

    if points_scored > points_allowed:
        history["wins"] += 1

    history["recent_points"].append(
        points_scored
    )

    history["recent_points_allowed"].append(
        points_allowed
    )


def build_features(df):
    processed_games = []

    current_season = None

    team_histories = defaultdict(
        create_team_history
    )

    for _, game in df.iterrows():

        season = int(game["Season"])
        week = int(game["Week"])

        home_team = game["HomeTeam"]
        away_team = game["AwayTeam"]

        home_score = int(game["HomeScore"])
        away_score = int(game["AwayScore"])

        # Reset histories each season
        if current_season != season:
            current_season = season

            team_histories = defaultdict(
                create_team_history
            )

        home_history = team_histories[
            home_team
        ]

        away_history = team_histories[
            away_team
        ]

        # Build features BEFORE updating
        # with the current game's result
        home_features = get_team_features(
            home_history
        )

        away_features = get_team_features(
            away_history
        )

        # Both teams need at least one
        # previous game in the season
        if (
            home_features is not None
            and away_features is not None
        ):

            # -------------------------
            # TOTAL POINTS
            # -------------------------

            total_points = (
                home_score
                + away_score
            )

            baseline_total = (
                home_features["ppg"]
                + away_features["ppg"]
            )

            target_residual = (
                total_points
                - baseline_total
            )

            # -------------------------
            # SPREAD / MARGIN
            # -------------------------

            # Positive = home team won
            # Negative = away team won
            home_margin = (
                home_score
                - away_score
            )

            home_scoring_diff = (
                home_features["ppg"]
                - home_features[
                    "points_allowed"
                ]
            )

            away_scoring_diff = (
                away_features["ppg"]
                - away_features[
                    "points_allowed"
                ]
            )

            # Positive = favors home
            # Negative = favors away
            baseline_margin = (
                home_scoring_diff
                - away_scoring_diff
            )

            # Neural network spread target
            target_margin_residual = (
                home_margin
                - baseline_margin
            )

            processed_games.append(
                {
                    "Season": season,
                    "Week": week,
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,

                    # HOME FEATURES
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

                    # AWAY FEATURES
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

                    # TOTAL TARGETS
                    "baseline_total":
                        baseline_total,

                    "total_points":
                        total_points,

                    "target_residual":
                        target_residual,

                    # SPREAD TARGETS
                    "baseline_margin":
                        baseline_margin,

                    "home_margin":
                        home_margin,

                    "target_margin_residual":
                        target_margin_residual,
                }
            )

        # Update histories AFTER creating
        # features for this game
        update_team_history(
            home_history,
            home_score,
            away_score,
        )

        update_team_history(
            away_history,
            away_score,
            home_score,
        )

    return pd.DataFrame(
        processed_games
    )


def save_processed_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )


def main():
    print("Loading raw NFL data...")

    df = load_data()

    print(f"Raw rows: {len(df)}")

    print("Cleaning regular-season games...")

    df = clean_regular_season_games(
        df
    )

    print(f"Completed regular-season games: {len(df)}")

    print("Building team features...")

    processed_df = build_features(
        df
    )

    print(f"Processed games: {len(processed_df)}")

    print(f"Columns: {len(processed_df.columns)}")

    save_processed_data(
        processed_df
    )

    print("\nSaved processed data to:")

    print(PROCESSED_DATA_PATH)

    print("\nSample:")

    print(
        processed_df[
            [
                "Season",
                "Week",
                "AwayTeam",
                "HomeTeam",
                "total_points",
                "baseline_total",
                "home_margin",
                "baseline_margin",
                "target_margin_residual",
            ]
        ].tail()
    )


if __name__ == "__main__":
    main()