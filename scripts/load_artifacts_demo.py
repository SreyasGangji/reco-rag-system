import joblib
from pathlib import Path

def main():
    models_dir = Path("artifacts/models")
    data_dir = Path("artifacts/data")

    # Load models
    ridge_model = joblib.load(models_dir / "ridge_model.pkl")
    rf_model = joblib.load(models_dir / "rf_model.pkl")
    gb_model = joblib.load(models_dir / "gb_model.pkl")

    # Load supporting data
    movie_stats = joblib.load(data_dir / "movie_stats.pkl")
    user_stats = joblib.load(data_dir / "user_stats.pkl")
    items = joblib.load(data_dir / "items.pkl")
    features = joblib.load(data_dir / "features.pkl")

    print("Artifacts loaded successfully.\n")

    print("Loaded models:")
    print(type(ridge_model))
    print(type(rf_model))
    print(type(gb_model))

    print("\nLoaded data artifacts:")
    print("movie_stats shape:", movie_stats.shape)
    print("user_stats shape:", user_stats.shape)
    print("items shape:", items.shape)
    print("features:", features)

if __name__ == "__main__":
    main()