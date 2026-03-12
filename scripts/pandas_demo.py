import pandas as pd

data = {
    "user_id": [1, 1, 2, 2, 4],
    "movie_id": [50, 100, 50, 100, 150],
    "rating": [5, 4, 3, 5, 4]
}

df = pd.DataFrame(data)
print(df)
print(df["rating"])
print(df["user_id"].__len__())
print(df.groupby("movie_id")["rating"])
print(df.groupby("movie_id").agg({
    "rating": ["mean", "count", "min", "max"]
}))
print(df.groupby("movie_id").agg({
    "user_id": ["nunique", "count"],
    "rating": ["count"]
}))