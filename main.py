from src.preprocess import main as preprocess_data
from src.train_current import train_current_model
from src.predict_week import predict_week


def main():
    print("\n==============================")
    print("NFL Total Predictor")
    print("==============================")

    print("\n[1/3] Preprocessing data...")
    preprocess_data()

    print("\n[2/3] Training current model...")
    train_current_model()

    print("\n[3/3] Predicting upcoming games...")
    predict_week()

    print("\n==============================")
    print("Finished")
    print("==============================")


if __name__ == "__main__":
    main()