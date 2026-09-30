import torch

from data_loader import (
    load_processed_data,
    split_data,
    prepare_data,
    FEATURE_COLUMNS
)

from model import NFLTotalModel


def train_model():
    # Load and prepare data
    df = load_processed_data()
    train_df, val_df, test_df = split_data(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = prepare_data(
        train_df,
        val_df,
        test_df
    )

    # Create model, loss function, and optimizer
    model = NFLTotalModel(
        input_size=len(FEATURE_COLUMNS)
    )

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Train the model
    epochs = 300

    for epoch in range(epochs):
        model.train()

        predictions = model(X_train)

        loss = criterion(
            predictions,
            y_train
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check validation performance
        if (epoch + 1) % 25 == 0:
            model.eval()

            with torch.no_grad():
                val_predictions = model(X_val)

                val_loss = criterion(
                    val_predictions,
                    y_val
                )

                val_mae = torch.mean(
                    torch.abs(val_predictions - y_val)
                )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Train Loss: {loss.item():.2f} "
                f"- Val Loss: {val_loss.item():.2f} "
                f"- Val MAE: {val_mae.item():.2f}"
            )

    # Save trained model
    torch.save(
        model.state_dict(),
        "models/nfl_total_model.pth"
    )

    print("\nModel saved to models/nfl_total_model.pth")

    return model


if __name__ == "__main__":
    train_model()