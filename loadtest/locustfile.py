from locust import HttpUser, task, between
import random


class RecommendationUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_recommendations(self):
        user_id = random.randint(1, 943)
        k = random.choice([5, 10])

        self.client.get(
            f"/recommend?user_id={user_id}&k={k}",
            name="/recommend"
        )

    @task
    def get_similar_movies(self):
        queries = [
            "toy story",
            "movies similar to star wars",
            "animated comedy movies",
            "romantic drama films",
            "sci-fi adventure movies"
        ]

        query = random.choice(queries)

        self.client.get(
            f"/similar?query={query}&k=5",
            name="/similar"
        )