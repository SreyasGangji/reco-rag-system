import pandas as pd
from pathlib import Path


def load_items():
    items_path = Path("data/raw/ml-100k/u.item")

    # Load full dataset
    columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama",
        "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    items = pd.read_csv(
        items_path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=columns
    )

    return items


def build_genre_text(row):
    genres = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    active_genres = [g for g in genres if row[g] == 1]

    return ", ".join(active_genres)


def main():
    items = load_items()

    # Create genre text
    items["genres_text"] = items.apply(build_genre_text, axis=1)

    # Create combined text (THIS IS IMPORTANT)
    items["combined_text"] = (
        "Title: " + items["title"] +
        ". Genres: " + items["genres_text"]
    )

    # Keep only relevant columns
    corpus = items[["movie_id", "title", "genres_text", "combined_text"]]

    print("\nSample corpus:")
    print(corpus.head())

    print("\nExample combined text:")
    print(corpus["combined_text"].iloc[0])

    # Save corpus
    output_path = Path("artifacts/data/movie_corpus.csv")
    corpus.to_csv(output_path, index=False)

    print("\nMovie corpus saved to:", output_path)


if __name__ == "__main__":
    main()