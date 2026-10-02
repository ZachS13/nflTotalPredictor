# NFL Total Points Predictor

## Overview

The NFL Total Points Predictor is a machine learning project built with Python and PyTorch that predicts the combined total score of NFL games using historical and current-season team performance.

The project was created as a way to learn more about machine learning, neural networks, feature engineering, model evaluation, and using trained models to make real-world predictions.

Rather than predicting the game total entirely from scratch, the current model uses a simple statistical baseline and trains a neural network to predict how much that baseline should be adjusted.

## How It Works

The prediction system follows this general process:

1. Load historical NFL game results.
2. Calculate team statistics using only games played before the game being predicted.
3. Create a baseline total using each team's scoring average.
4. Pass additional team statistics into a PyTorch neural network.
5. Predict an adjustment to the baseline.
6. Add the neural adjustment to the baseline to produce the final predicted total.

Example:

```text
Baseline Total:      45.7
Neural Adjustment:   +4.4
Predicted Total:     50.1
```

The neural network is learning when the simple scoring-average baseline tends to be too high or too low.

## Current Features

The model currently uses 17 input features.

### Game Information

- Week

### Home Team

- Points per game
- Points allowed per game
- Win percentage
- Last 3 games points per game
- Last 3 games points allowed
- Last 5 games points per game
- Last 5 games points allowed
- Average total points in games

### Away Team

- Points per game
- Points allowed per game
- Win percentage
- Last 3 games points per game
- Last 3 games points allowed
- Last 5 games points per game
- Last 5 games points allowed
- Average total points in games

All statistics are calculated using only information that would have been available before the game being predicted.

## Model Architecture

The current PyTorch neural network is a feed-forward regression model.

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

The final output represents the predicted adjustment to the baseline total.

The model currently uses:

- Mean Squared Error loss
- Adam optimizer
- StandardScaler feature normalization
- 300 training epochs

## Residual Learning

The first version of the project attempted to directly predict the total number of points scored in a game.

A simple baseline using:

```text
Home PPG + Away PPG
```

performed better than the original neural network.

The model was then changed to use residual learning.

Instead of predicting:

```text
Neural Network → Total Points
```

the current model predicts:

```text
Neural Network → Baseline Adjustment
```

The final prediction becomes:

```text
Predicted Total =
Baseline Total + Neural Adjustment
```

This approach improved performance over the simple baseline during historical testing.

## Walk-Forward Evaluation

To avoid data leakage and more realistically simulate real predictions, the project uses weekly walk-forward validation.

For example:

```text
Predict Week 5

Training Data:
2010–2025
+
Current Season Weeks 2–4

Test Data:
Current Season Week 5
```

After Week 5 is completed, those games become available for training when predicting Week 6.

This process is repeated throughout each season.

## Current Historical Results

The model was walk-forward tested across 1,311 NFL games.

```text
Neural Network MAE: 10.99
Baseline MAE:       11.46
MAE Improvement:    +0.47
```

The neural network outperformed the baseline in each tested season from 2021 through 2026.

### Season Results

```text
2021: Neural 10.97 | Baseline 11.89 | Improvement +0.92
2022: Neural 11.07 | Baseline 11.24 | Improvement +0.17
2023: Neural 11.04 | Baseline 11.68 | Improvement +0.64
2024: Neural 10.46 | Baseline 10.88 | Improvement +0.42
2025: Neural 11.14 | Baseline 11.26 | Improvement +0.12
2026: Neural 13.18 | Baseline 14.38 | Improvement +1.20
```

## Current Prediction Workflow

The project contains a separate model used for upcoming-game predictions.

```text
Raw NFL Data
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Train Current Model
      ↓
Load Upcoming Matchups
      ↓
Calculate Current Team Statistics
      ↓
Baseline Prediction
      ↓
Neural Adjustment
      ↓
Final Predicted Total
```

The current model is retrained using all completed games available at the time of prediction.

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
│   ├── nfl_total_model.pth
│   ├── nfl_current_model.pth
│   └── nfl_current_scaler.pkl
│
├── src/
│   ├── preprocess.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── walk_forward.py
│   ├── train_current.py
│   └── predict_week.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_model.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Running the Project

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Run the full prediction workflow:

```powershell
python .\main.py
```

This will:

1. Preprocess the raw NFL data.
2. Train the current neural network.
3. Generate predictions for the upcoming games.

## Running Individual Components

Each part of the project can also be run separately.

### Preprocess Data

```powershell
python .\src\preprocess.py
```

### Historical Walk-Forward Testing

```powershell
python .\src\walk_forward.py
```

### Train Current Model

```powershell
python .\src\train_current.py
```

### Predict Upcoming Games

```powershell
python .\src\predict_week.py
```

## Upcoming Games

Upcoming matchups are stored in:

```text
data/upcoming/upcoming_games.csv
```

Example:

```csv
Season,Week,AwayTeam,HomeTeam
2026,4,PIT,CLE
2026,4,NE,BUF
2026,4,KC,LV
```

Predictions are saved to:

```text
data/upcoming/predictions.csv
```

Example:

```csv
Season,Week,AwayTeam,HomeTeam,BaselineTotal,NeuralAdjustment,PredictedTotal
2026,4,PIT,CLE,35.67,4.26,39.92
2026,4,NE,BUF,45.67,4.38,50.04
2026,4,KC,LV,58.67,-9.64,49.02
```

## Technology

- Python

- PyTorch

- Pandas

- NumPy

- Scikit-learn

- Matplotlib