import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from data_loader import (
    load_processed_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

from model import NFLTotalModel


BASE_DIR = Path(__file__).resolve().parent.parent

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

EPOCHS = 300
LEARNING_RATE = 0.001


def train_current_model():
    torch.manual_seed(42)

    df = load_processed_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).unsqueeze(1)

    model = NFLTotalModel(
        input_size=len(FEATURE_COLUMNS)
    )

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        f"Training on {len(df)} completed games"
    )

    print(
        f"Seasons: "
        f"{df['Season'].min()} "
        f"through "
        f"{df['Season'].max()}"
    )

    for epoch in range(EPOCHS):
        model.train()

        predictions = model(
            X_tensor
        )

        loss = criterion(
            predictions,
            y_tensor
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            mae = torch.mean(
                torch.abs(
                    predictions
                    - y_tensor
                )
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"- Loss: {loss.item():.2f} "
                f"- Residual MAE: {mae.item():.2f}"
            )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    with open(
        SCALER_PATH,
        "wb"
    ) as file:
        pickle.dump(
            scaler,
            file
        )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nScaler saved to:"
        f"\n{SCALER_PATH}"
    )


if __name__ == "__main__":
    train_current_model()