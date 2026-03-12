import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_ENV: str = os.getenv("APP_ENV", "dev")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    ARTIFACT_DIR: str = os.getenv("ARTIFACT_DIR", "artifacts")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()