# NFL Game Predictor

## Overview

The NFL Game Predictor is a machine learning project built with Python and PyTorch that predicts NFL game totals and point spreads using historical and current-season team performance.

The project was created to learn more about machine learning, neural networks, feature engineering, model evaluation, and using trained models to make real-world predictions.

The project currently contains two neural network models:

- **Total Points Model** – predicts the combined number of points scored in a game.
- **Spread Model** – predicts the expected point differential between the home and away teams.

Both models use residual learning, where a simple statistical baseline is calculated first and the neural network learns how much that baseline should be adjusted.

## How It Works

Historical NFL game results are processed into pregame team statistics.

The model only uses information that would have been available before the game being predicted.

The general prediction pipeline is:

```text
Historical NFL Games
        ↓
Feature Engineering
        ↓
Statistical Baseline
        ↓
Neural Network Adjustment
        ↓
Final Prediction
```

For an upcoming game, the system can produce results such as:

```text
BUF @ NE

Predicted Total: 49.8
Predicted Spread: BUF -5.7
Predicted Winner: BUF
```

## Current Features

The models currently use 17 input features.

### Game Information

- Week

### Home Team Features

- Points per game
- Points allowed per game
- Win percentage
- Last 3 games points per game
- Last 3 games points allowed
- Last 5 games points per game
- Last 5 games points allowed
- Average total points in games

### Away Team Features

- Points per game
- Points allowed per game
- Win percentage
- Last 3 games points per game
- Last 3 games points allowed
- Last 5 games points per game
- Last 5 games points allowed
- Average total points in games

Team histories reset at the beginning of each NFL season.

Features are calculated before the current game's result is added to team history to prevent future information from leaking into the prediction.

## Neural Network Architecture

Both models currently use the same feed-forward neural network structure:

```text
17 Input Features
        ↓
64 Neurons
        ↓
ReLU
        ↓
32 Neurons
        ↓
ReLU
        ↓
16 Neurons
        ↓
ReLU
        ↓
1 Output
```

The models currently use:

- PyTorch
- Mean Squared Error loss
- Adam optimizer
- StandardScaler feature normalization
- 300 training epochs

The output has a different meaning depending on the model.

```text
Total Model
→ Total Points Adjustment

Spread Model
→ Point Margin Adjustment
```

## Total Points Model

The original version of the project trained a neural network to directly predict the total number of points scored.

A simple statistical baseline performed better than the original neural network, so the model was changed to use residual learning.

### Total Baseline

The current baseline is:

```text
Baseline Total =
Home Points Per Game
+
Away Points Per Game
```

For example:

```text
Home PPG: 26.7
Away PPG: 22.3

Baseline Total: 49.0
```

The neural network then predicts how much this baseline should be adjusted.

```text
Baseline Total:      49.0
Neural Adjustment:   -4.2
Predicted Total:     44.8
```

The training target is:

```text
Target Residual =
Actual Total - Baseline Total
```

The final prediction is:

```text
Predicted Total =
Baseline Total + Neural Adjustment
```

## Spread Model

The spread model predicts the expected point differential between the two teams.

Historical point margin is calculated as:

```text
Home Margin =
Home Score - Away Score
```

A positive value means the home team won.

```text
Home 31
Away 24

Home Margin = +7
```

A negative value means the away team won.

```text
Home 20
Away 27

Home Margin = -7
```

Internally, the model always predicts from the home team's perspective.

The result is converted into standard spread-style output when displayed.

For example:

```text
Predicted Home Margin: -6.4
```

means:

```text
Away Team -6.4
```

## Spread Baseline

The spread model also uses a statistical baseline.

First, each team's average scoring differential is calculated.

```text
Home Scoring Differential =
Home PPG - Home Points Allowed

Away Scoring Differential =
Away PPG - Away Points Allowed
```

The baseline margin is then:

```text
Baseline Margin =
Home Scoring Differential
-
Away Scoring Differential
```

For example:

```text
Home Team

PPG:             28
Points Allowed:  21

Scoring Differential: +7
```

```text
Away Team

PPG:             24
Points Allowed:  26

Scoring Differential: -2
```

The resulting baseline is:

```text
Baseline Margin = +7 - (-2)

Baseline Margin = +9
```

This means the baseline expects the home team to win by approximately 9 points.

## Spread Residual Learning

The first spread model directly predicted the actual home margin.

Historical walk-forward testing produced:

```text
Games Tested:             3,934

Neural Margin MAE:        11.96
Baseline Margin MAE:      12.00
MAE Improvement:          +0.04

Neural Winner Accuracy:   59.56%
Baseline Winner Accuracy: 61.47%
```

The spread model was then changed to residual learning.

Instead of predicting:

```text
Neural Network → Home Margin
```

the model now predicts:

```text
Neural Network → Baseline Margin Adjustment
```

The training target is:

```text
Target Margin Residual =
Actual Home Margin
-
Baseline Margin
```

The final prediction is:

```text
Predicted Home Margin =
Baseline Margin
+
Neural Adjustment
```

For example:

```text
Baseline Margin:      -3.2
Neural Adjustment:    -2.4

Predicted Margin:     -5.6
```

If the away team is Buffalo, the displayed prediction becomes:

```text
BUF -5.6
```

## Current Spread Results

After changing the spread model to residual learning, historical walk-forward results improved.

```text
Games Tested:             3,934

Neural Margin MAE:        11.55
Baseline Margin MAE:      12.00
MAE Improvement:          +0.45

Neural Winner Accuracy:   60.07%
Baseline Winner Accuracy: 61.47%
```

The residual neural network improved point-margin accuracy compared with both the original direct neural network and the statistical baseline.

The statistical baseline currently remains slightly more accurate at selecting the winning team.

## Walk-Forward Validation

Both prediction systems are evaluated using weekly walk-forward testing.

This simulates how the models would have performed if they had actually been running during previous NFL seasons.

For example, when predicting Week 10 of the 2025 season:

```text
Training Data:

2010–2024
+
2025 Weeks before Week 10
```

The model then predicts:

```text
2025 Week 10
```

After Week 10 is completed, those games become available when predicting Week 11.

This process prevents the model from training on games that occurred after the game being tested.

## Total Model Historical Performance

The total-points model has also been evaluated using weekly walk-forward testing.

```text
Games Tested:       1,311
Neural Network MAE: 10.99
Baseline MAE:       11.46
MAE Improvement:    +0.47
```

Season results:

```text
2021: Neural 10.97 | Baseline 11.89 | Improvement +0.92
2022: Neural 11.07 | Baseline 11.24 | Improvement +0.17
2023: Neural 11.04 | Baseline 11.68 | Improvement +0.64
2024: Neural 10.46 | Baseline 10.88 | Improvement +0.42
2025: Neural 11.14 | Baseline 11.26 | Improvement +0.12
2026: Neural 13.18 | Baseline 14.38 | Improvement +1.20
```

The residual neural network outperformed the simple total-points baseline across each tested season.

## Current Prediction Pipeline

Upcoming-game predictions now combine both trained neural networks.

```text
Completed NFL Games
        ↓
Preprocessing
        ↓
Current Team Statistics
        ↓
        ├───────────────┐
        ↓               ↓
Total Baseline     Spread Baseline
        ↓               ↓
Total Model        Spread Model
        ↓               ↓
Adjustment         Adjustment
        ↓               ↓
Predicted Total    Predicted Margin
        └───────┬───────┘
                ↓
        Weekly Predictions
```

A weekly prediction can contain:

```text
BUF @ NE

Baseline Total:       45.7
Total Adjustment:     +4.4
Predicted Total:      50.1

Baseline Margin:      -3.2
Spread Adjustment:    -2.4
Predicted Spread:     BUF -5.6
Predicted Winner:     BUF
```

## Project Structure

```text
nflTotalPredictor/
│
├── data/
│   ├── raw/
│   │   └── games.csv
│   │
│   ├── processed/
│   │   └── games_processed.csv
│   │
│   └── upcoming/
│       ├── upcoming_games.csv
│       └── predictions.csv
│
├── models/
│   ├── nfl_current_model.pth
│   ├── nfl_current_scaler.pkl
│   ├── nfl_spread_model.pth
│   └── nfl_spread_scaler.pkl
│
├── src/
│   ├── preprocess.py
│   ├── data_loader.py
│   ├── model.py
│   ├── spread_model.py
│   ├── train.py
│   ├── train_current.py
│   ├── train_spread.py
│   ├── evaluate.py
│   ├── walk_forward.py
│   ├── walk_forward_spread.py
│   └── predict_week.py
│
├── tests/
│
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Running the Project

Activate the Python virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

### Preprocess Historical Data

```powershell
python .\src\preprocess.py
```

This creates the processed training data and calculates:

- Team features
- Total baseline
- Total residual
- Actual total
- Spread baseline
- Actual home margin
- Spread residual

### Train the Current Total Model

```powershell
python .\src\train_current.py
```

This creates:

```text
models/nfl_current_model.pth
models/nfl_current_scaler.pkl
```

### Train the Current Spread Model

```powershell
python .\src\train_spread.py
```

This creates:

```text
models/nfl_spread_model.pth
models/nfl_spread_scaler.pkl
```

### Walk-Forward Test Total Predictions

```powershell
python .\src\walk_forward.py
```

### Walk-Forward Test Spread Predictions

```powershell
python .\src\walk_forward_spread.py
```

### Generate Upcoming Predictions

Upcoming matchups are stored in:

```text
data/upcoming/upcoming_games.csv
```

Run:

```powershell
python .\src\predict_week.py
```

Predictions are saved to:

```text
data/upcoming/predictions.csv
```

The prediction file contains:

```text
Season
Week
AwayTeam
HomeTeam

BaselineTotal
TotalAdjustment
PredictedTotal

BaselineMargin
SpreadAdjustment
PredictedHomeMargin
PredictedSpread
PredictedWinner
```

## Technology

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib