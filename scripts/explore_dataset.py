import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def main():
    # Path to ratings file
    ratings_path = Path("data/raw/ml-100k/u.data")

    # Load dataset
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    items_path = Path("data/raw/ml-100k/u.item")

    # u.item is "|" separated and uses latin-1 encoding
    # We only need movie_id and title for now
    items = pd.read_csv(
        items_path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"]
    )

    # Show first few rows
    print("\nFirst 5 rows:")
    print(ratings.head())

    # Dataset shape
    print("\nDataset shape:")
    print(ratings.shape)

    # Number of unique users
    print("\nUnique users:", ratings["user_id"].nunique())

    # Number of unique movies
    print("Unique movies:", ratings["movie_id"].nunique())

    # Rating distribution
    print("\nRating distribution:")
    print(ratings["rating"].value_counts().sort_index())

    # Movie statistics
    movie_stats = ratings.groupby("movie_id").agg(
        avg_movie_rating=("rating", "mean"),
        movie_rating_count=("rating", "count")
    ).reset_index()

    print("\nMovie statistics:")
    print(movie_stats.head())

    # User statistics
    user_stats = ratings.groupby("user_id").agg(
        avg_user_rating=("rating", "mean"),
        user_rating_count=("rating", "count")
    ).reset_index()

    print("\nUser statistics:")
    print(user_stats.head())

    # Merge movie features into ratings
    data = ratings.merge(movie_stats, on="movie_id")

    # Merge user features into ratings
    data = data.merge(user_stats, on="user_id")

    print("\nFeature table preview:")
    print(data.head())

    print("\nFeature table shape:")
    print(data.shape)

    # Define features and target
    features = [
        "user_id",
        "movie_id",
        "avg_movie_rating",
        "movie_rating_count",
        "avg_user_rating",
        "user_rating_count"
    ]

    X = data[features]
    y = data["rating"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    # Train Ridge Regression model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)

    # Predict on test set
    ridge_predictions = ridge_model.predict(X_test)

    # Evaluate Ridge model
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))

    print("\nRidge Evaluation")
    print("Ridge RMSE:", ridge_rmse)

    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Predict on test set
    rf_predictions = rf_model.predict(X_test)

    # Evaluate Random Forest model
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

    print("\nRandom Forest Evaluation")
    print("Random Forest RMSE:", rf_rmse)

    # Train Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    # Predict on test set
    gb_predictions = gb_model.predict(X_test)

    # Evaluate model
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

    print("\nGradient Boosting Evaluation")
    print("Gradient Boosting RMSE:", gb_rmse)


    # Ensemble predictions
    ensemble_predictions = (
        ridge_predictions +
        rf_predictions +
        gb_predictions
    ) / 3

    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))

    print("\nEnsemble Evaluation")
    print("Ensemble RMSE:", ensemble_rmse)


    # Create artifact directories
    models_dir = Path("artifacts/models")
    data_dir = Path("artifacts/data")

    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save trained models
    joblib.dump(ridge_model, models_dir / "ridge_model.pkl")
    joblib.dump(rf_model, models_dir / "rf_model.pkl")
    joblib.dump(gb_model, models_dir / "gb_model.pkl")

    # Save supporting data artifacts
    movie_stats.to_csv(data_dir / "movie_stats.csv", index=False)
    user_stats.to_csv(data_dir / "user_stats.csv", index=False)
    items.to_csv(data_dir / "items.csv", index=False)
    ratings.to_csv(data_dir / "ratings.csv", index=False)
    
    joblib.dump(features, data_dir / "features.pkl")

    print("\nArtifacts saved successfully.")
    print("Models saved in:", models_dir)
    print("Data artifacts saved in:", data_dir)


    # Example recommendation for a user using Ridge model
    user_id = 50

    # Movies already rated by this user
    rated_movies = ratings[ratings["user_id"] == user_id]["movie_id"].tolist()

    # All movies
    all_movies = items["movie_id"].unique()

    # Candidate movies (not rated yet)
    candidate_movies = [m for m in all_movies if m not in rated_movies]

    # Create candidate feature rows
    candidate_data = pd.DataFrame({
        "user_id": user_id,
        "movie_id": candidate_movies
    })

    # Add movie features
    candidate_data = candidate_data.merge(movie_stats, on="movie_id")

    # Add user features
    user_feature = user_stats[user_stats["user_id"] == user_id]
    candidate_data = candidate_data.merge(user_feature, on="user_id")

    # Predict ratings using Ridge model
    candidate_data["predicted_rating"] = ridge_model.predict(candidate_data[features])

    # Top 10 recommendations
    recommendations = candidate_data.sort_values(
        "predicted_rating", ascending=False
    ).head(10)

    # Add movie titles to recommendations
    recommendations = recommendations.merge(items, on="movie_id", how="left")

    print("\nTop 10 recommended movies for user", user_id)
    print(recommendations[["movie_id", "title", "predicted_rating"]])


if __name__ == "__main__":
    main()