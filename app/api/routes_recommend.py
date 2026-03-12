from fastapi import APIRouter, HTTPException
from app.services.recommendation_service import recommendation_service

router = APIRouter()


@router.get("/recommend")
def recommend(user_id: int, k: int = 10):
    if not recommendation_service.user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")

    recommendations = recommendation_service.get_recommendations(user_id, k)

    return {
        "user_id": user_id,
        "k": k,
        "recommendations": recommendations
    }