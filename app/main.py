from fastapi import FastAPI
from app.core.logging import setup_logging
from app.api.routes_health import router as health_router
from app.api.routes_recommend import router as recommend_router
from app.api.routes_retrieval import router as retrieval_router
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(health_router)
app.include_router(recommend_router)
app.include_router(retrieval_router)