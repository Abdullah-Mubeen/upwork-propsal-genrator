"""
Profile Management Routes

CRUD operations for user profile with authentication.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime

from app.middleware.auth import verify_api_key
from app.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/profile", tags=["profile"])


# ===================== MODELS =====================

class ProfileUpdate(BaseModel):
    """Profile update request"""
    name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    upwork_url: Optional[str] = Field(None, max_length=500)
    bio: Optional[str] = Field(None, max_length=1000)
    hourly_rate: Optional[float] = Field(None, ge=0)
    skills: Optional[list[str]] = None
    timezone: Optional[str] = None


class ProfileResponse(BaseModel):
    """Profile response"""
    success: bool
    profile: Dict[str, Any]


# ===================== ENDPOINTS =====================

@router.get("", response_model=ProfileResponse)
async def get_profile(api_key: str = Depends(verify_api_key)):
    """Get current user profile"""
    try:
        db = get_db()
        profile = db.get_profile()
        return ProfileResponse(success=True, profile=profile)
    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        raise HTTPException(500, str(e))


@router.put("", response_model=ProfileResponse)
async def update_profile(
    data: ProfileUpdate,
    api_key: str = Depends(verify_api_key)
):
    """Update user profile"""
    try:
        db = get_db()
        # Filter out None values
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}
        profile = db.update_profile(update_data)
        return ProfileResponse(success=True, profile=profile)
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(500, str(e))
