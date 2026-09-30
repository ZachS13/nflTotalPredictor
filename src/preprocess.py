import os
from collections import defaultdict, deque

import pandas as pd


RAW_DATA_PATH = "data/raw/2010-2026_scores.csv"
PROCESSED_DATA_PATH = "data/processed/2010-2026_scores_processed.csv"


def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def clean_regular_season_games(df):
    df = df.copy()

    # Keep completed games only
    df = df[df["GameStatus"] == "FINAL"]

    # Keep regular-season weeks only
    df = df[df["Week"].str.match(r"^WEEK \d+$")]

    # Convert "WEEK 1" -> 1
    df["Week"] = (
        df["Week"]
        .str.replace("WEEK ", "", regex=False)
        .astype(int)
    )

    # Remove anything missing essential game information
    df = df.dropna(
        subset=[
            "HomeTeam",
            "AwayTeam",
            "HomeScore",
            "AwayScore"
        ]
    )

    # Make scores integers
    df["HomeScore"] = df["HomeScore"].astype(int)
    df["AwayScore"] = df["AwayScore"].astype(int)

    # Sort chronologically enough for our purposes
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
        "recent_points": deque(maxlen=3)
    }


def get_team_features(history):
    if history["games"] == 0:
        return None

    games = history["games"]

    return {
        "ppg": history["points_scored"] / games,
        "points_allowed": history["points_allowed"] / games,
        "win_pct": history["wins"] / games,
        "last_3_ppg": sum(history["recent_points"]) / len(history["recent_points"])
    }


def update_team_history(history, points_scored, points_allowed):
    history["games"] += 1
    history["points_scored"] += points_scored
    history["points_allowed"] += points_allowed

    if points_scored > points_allowed:
        history["wins"] += 1

    history["recent_points"].append(points_scored)


def build_features(df):
    processed_rows = []

    # Reset histories each season
    for season, season_games in df.groupby("Season"):

        team_history = defaultdict(create_team_history)

        season_games = season_games.sort_values("Week")

        for _, game in season_games.iterrows():

            home_team = game["HomeTeam"]
            away_team = game["AwayTeam"]

            home_history = team_history[home_team]
            away_history = team_history[away_team]

            home_features = get_team_features(home_history)
            away_features = get_team_features(away_history)

            # Skip games where either team has no previous games.
            # This will mostly remove Week 1.
            if home_features is not None and away_features is not None:

                total_points = game["HomeScore"] + game["AwayScore"]

                processed_rows.append({
                    "Season": season,
                    "Week": game["Week"],
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,

                    "home_ppg": home_features["ppg"],
                    "home_points_allowed": home_features["points_allowed"],
                    "home_win_pct": home_features["win_pct"],
                    "home_last_3_ppg": home_features["last_3_ppg"],

                    "away_ppg": away_features["ppg"],
                    "away_points_allowed": away_features["points_allowed"],
                    "away_win_pct": away_features["win_pct"],
                    "away_last_3_ppg": away_features["last_3_ppg"],

                    "total_points": total_points
                })

            # Update histories AFTER creating the features.
            # This prevents the current game's result from leaking into
            # the inputs used to predict that same game.
            update_team_history(
                home_history,
                game["HomeScore"],
                game["AwayScore"]
            )

            update_team_history(
                away_history,
                game["AwayScore"],
                game["HomeScore"]
            )

    return pd.DataFrame(processed_rows)


def save_processed_data(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main():
    df = load_data()

    print(f"Raw rows: {len(df)}")

    df = clean_regular_season_games(df)

    print(f"Regular-season completed games: {len(df)}")

    processed_df = build_features(df)

    print(f"Processed rows: {len(processed_df)}")

    print("\nColumns:")
    print(processed_df.columns.tolist())

    print("\nFirst 5 rows:")
    print(processed_df.head())

    print("\nMissing values:")
    print(processed_df.isnull().sum())

    save_processed_data(processed_df)

    print(f"\nSaved processed data to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()