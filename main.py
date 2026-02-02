from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

from app.config import settings
from app.db import get_db
from app.routes import job_data_router
from app.routes.proposals import router as proposals_router
from app.routes.profile import router as profile_router
from app.routes.analytics import router as analytics_router
from app.middleware.auth import verify_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])

app = FastAPI(
    title="AI Proposal Generator Backend",
    version="1.0.0",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - parse origins from config
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins and origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "AI Proposal Generator Backend",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/auth/verify")
async def verify_api_key_endpoint(api_key: str = Depends(verify_api_key)):
    """
    Verify if the provided API key is valid.
    
    Send X-API-Key header with your API key.
    Returns success if valid, 401/403 if invalid.
    """
    return {
        "valid": True,
        "message": "API key is valid. You have access to protected endpoints."
    }


app.include_router(job_data_router, prefix="/api/job-data")
app.include_router(proposals_router)
app.include_router(profile_router)
app.include_router(analytics_router)

# Serve frontend static files - mount AFTER API routes to avoid conflicts
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
