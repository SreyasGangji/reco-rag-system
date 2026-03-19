from fastapi import APIRouter
from app.services.retrieval_service import retrieval_service

router = APIRouter()


@router.get("/similar")
def similar_movies(query: str, k: int = 5):
    results = retrieval_service.search_similar_movies(query, k)

    return {
        "query": query,
        "k": k,
        "results": results
    }