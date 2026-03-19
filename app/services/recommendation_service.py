from pathlib import Path
import joblib
import pandas as pd
from app.services.retrieval_service import retrieval_service
from app.services.explanation_service import explanation_service

class RecommendationService:
    def __init__(self):
        models_dir = Path("artifacts/models")
        data_dir = Path("artifacts/data")

        # Load trained models
        self.ridge_model = joblib.load(models_dir / "ridge_model.pkl")
        self.rf_model = joblib.load(models_dir / "rf_model.pkl")
        self.gb_model = joblib.load(models_dir / "gb_model.pkl")

        # Load supporting data artifacts
        self.movie_stats = pd.read_csv(data_dir / "movie_stats.csv")
        self.user_stats = pd.read_csv(data_dir / "user_stats.csv")
        self.items = pd.read_csv(data_dir / "items.csv")
        self.features = joblib.load(data_dir / "features.pkl")
        self.movie_corpus = pd.read_csv(Path("artifacts/data/movie_corpus.csv"))

        # Load ratings so we know which movies a user already rated
        self.ratings = pd.read_csv(data_dir / "ratings.csv")

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
            self.movie_corpus[["movie_id", "title", "genres_text", "combined_text"]],
            on="movie_id",
            how="left"
        )

    
        # Format result
        output = recommendations[["movie_id", "title", "predicted_rating", "genres_text", "combined_text"]].copy()
        output["predicted_rating"] = output["predicted_rating"].round(3)

        results = []
        for _, row in output.iterrows():
            similar_movies = retrieval_service.search_similar_movies(
                row["combined_text"],
                k=3
            )

            # remove self-match if present
            similar_titles = [
                m["title"]
                for m in similar_movies
                if m["movie_id"] != row["movie_id"]
            ][:2]

            explanation = explanation_service.build_explanation(
                title=row["title"],
                predicted_rating=float(row["predicted_rating"]),
                similar_context=similar_titles
            )    

            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "predicted_rating": float(row["predicted_rating"]),
                "similar_context": similar_titles,
                "explanation": explanation
            })

        return results


recommendation_service = RecommendationService()