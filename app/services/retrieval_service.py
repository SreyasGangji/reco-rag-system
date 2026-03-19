from pathlib import Path


import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalService:
    def __init__(self):
        index_path = Path("artifacts/data/movie_index.faiss")

        metadata_path = Path("artifacts/data/movie_index_metadata.csv")
        self.index = faiss.read_index(str(index_path))

        self.metadata = pd.read_csv(metadata_path)

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(self.model_name)

    def search_similar_movies(self, query_text: str, k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode([query_text])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            row = self.metadata.iloc[idx]
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres_text": row["genres_text"],
                "combined_text": row["combined_text"],
                "distance": float(dist)
            })

        return results


retrieval_service = RetrievalService()