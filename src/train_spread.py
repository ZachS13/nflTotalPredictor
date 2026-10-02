import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

try:
    from src.data_loader import (
        load_processed_data,
        FEATURE_COLUMNS
    )

    from src.spread_model import NFLSpreadModel

except ModuleNotFoundError:
    from data_loader import (
        load_processed_data,
        FEATURE_COLUMNS
    )

    from spread_model import NFLSpreadModel


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nfl_spread_model.pth"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "nfl_spread_scaler.pkl"
)

TARGET_COLUMN = "home_margin"


def train_spread_model():
    torch.manual_seed(42)

    print("Loading processed data...")

    df = load_processed_data()

    print(
        f"Training games: {len(df)}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    X = df[FEATURE_COLUMNS].values
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

    model = NFLSpreadModel(
        input_size=len(FEATURE_COLUMNS)
    )

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 300

    print("\nTraining spread model...\n")

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        predictions = model(
            X_tensor
        )

        loss = loss_function(
            predictions,
            y_tensor
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 25 == 0
        ):
            with torch.no_grad():

                mae = torch.mean(
                    torch.abs(
                        predictions
                        - y_tensor
                    )
                ).item()

            print(
                f"Epoch {epoch + 1:3d}/{epochs} "
                f"| Loss: {loss.item():.2f} "
                f"| Margin MAE: {mae:.2f}"
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
        f"\nSpread model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Spread scaler saved to: "
        f"{SCALER_PATH}"
    )


def main():
    train_spread_model()


if __name__ == "__main__":
    main()