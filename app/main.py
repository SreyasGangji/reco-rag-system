from fastapi import FastAPI
from app.core.logging import setup_logging
import logging

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
def health():
    logger.info("Health endpoint called")
    return {"status": "ok"}