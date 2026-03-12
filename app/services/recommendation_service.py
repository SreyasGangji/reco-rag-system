from pathlib import Path
import joblib
import pandas as pd


class RecommendationService:
    def __init__(self):
        models_dir = Path("artifacts/models")
        data_dir = Path("artifacts/data")

        # Load trained models
        self.ridge_model = joblib.load(models_dir / "ridge_model.pkl")
        self.rf_model = joblib.load(models_dir / "rf_model.pkl")
        self.gb_model = joblib.load(models_dir / "gb_model.pkl")

        # Load supporting data artifacts
        self.movie_stats = joblib.load(data_dir / "movie_stats.pkl")
        self.user_stats = joblib.load(data_dir / "user_stats.pkl")
        self.items = joblib.load(data_dir / "items.pkl")
        self.features = joblib.load(data_dir / "features.pkl")

        # Load ratings so we know which movies a user already rated
        self.ratings = pd.read_csv(
            Path("data/raw/ml-100k/u.data"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

    def user_exists(self, user_id: int) -> bool:
        return user_id in self.user_stats["user_id"].values

    def get_recommendations(self, user_id: int, k: int = 10) -> list[dict]:
        # Movies already rated by this user
        rated_movies = self.ratings[
            self.ratings["user_id"] == user_id
        ]["movie_id"].tolist()

        # All possible movies
        all_movies = self.items["movie_id"].unique()

        # Candidate movies = unseen movies
        candidate_movies = [m for m in all_movies if m not in rated_movies]

        # Build candidate rows
        candidate_data = pd.DataFrame({
            "user_id": user_id,
            "movie_id": candidate_movies
        })

        # Add movie-level features
        candidate_data = candidate_data.merge(
            self.movie_stats,
            on="movie_id",
            how="left"
        )

        # Add user-level features
        user_feature = self.user_stats[self.user_stats["user_id"] == user_id]
        candidate_data = candidate_data.merge(
            user_feature,
            on="user_id",
            how="left"
        )

        # Predict using all three models
        ridge_pred = self.ridge_model.predict(candidate_data[self.features])
        rf_pred = self.rf_model.predict(candidate_data[self.features])
        gb_pred = self.gb_model.predict(candidate_data[self.features])

        # Ensemble prediction
        candidate_data["predicted_rating"] = (
            ridge_pred + rf_pred + gb_pred
        ) / 3

        # Sort and keep top-k
        recommendations = candidate_data.sort_values(
            "predicted_rating",
            ascending=False
        ).head(k)

        # Add titles
        recommendations = recommendations.merge(
            self.items,
            on="movie_id",
            how="left"
        )

        # Format result
        output = recommendations[["movie_id", "title", "predicted_rating"]].copy()
        output["predicted_rating"] = output["predicted_rating"].round(3)

        return output.to_dict(orient="records")


recommendation_service = RecommendationService()