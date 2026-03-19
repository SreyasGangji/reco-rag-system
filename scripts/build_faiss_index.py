from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    # Load movie corpus
    corpus_path = Path("artifacts/data/movie_corpus.pkl")
    corpus = pd.read_pickle(corpus_path)

    print("Loaded corpus shape:", corpus.shape)
    print("Sample text:")
    print(corpus["combined_text"].iloc[0])

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print("\nLoading embedding model:", model_name)
    model = SentenceTransformer(model_name)

    # Convert text to embeddings
    texts = corpus["combined_text"].tolist()
    print("\nGenerating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Convert to numpy float32 (FAISS requires float32)
    embeddings = np.array(embeddings).astype("float32")

    print("Embeddings shape:", embeddings.shape)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("FAISS index built.")
    print("Number of vectors in index:", index.ntotal)

    # Save index
    index_path = Path("artifacts/data/movie_index.faiss")
    faiss.write_index(index, str(index_path))

    # Save metadata mapping
    metadata_path = Path("artifacts/data/movie_index_metadata.csv")
    corpus[["movie_id", "title", "genres_text", "combined_text"]].to_csv(metadata_path, index=False)
   
    print("\nSaved index to:", index_path)
    print("Saved metadata to:", metadata_path)


if __name__ == "__main__":
    main()