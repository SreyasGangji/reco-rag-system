import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    # Load FAISS index
    index_path = Path("artifacts/data/movie_index.faiss")
    index = faiss.read_index(str(index_path))

    # Load metadata
    metadata_path = Path("artifacts/data/movie_index_metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Example query
    query = "Title: Toy Story (1995). Genres: Animation, Children's, Comedy"
    print("Query:", query)

    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS index
    k = 5
    distances, indices = index.search(query_embedding, k)

    print("\nTop similar movies:")
    for rank, idx in enumerate(indices[0]):
        row = metadata.iloc[idx]
        print(
            f"{rank + 1}. "
            f"movie_id={row['movie_id']}, "
            f"title={row['title']}, "
            f"genres={row['genres_text']}"
        )


if __name__ == "__main__":
    main()