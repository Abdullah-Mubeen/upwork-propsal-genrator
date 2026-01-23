"""
API Key Authentication Middleware

Security layer for protecting training data and history endpoints.
- Public: Proposal generation (index.html)
- Protected: Training data management, history viewing

Usage:
    from app.middleware.auth import verify_api_key
    
    @router.post("/upload")
    async def upload_job_data(
        job_data: JobDataUploadRequest,
        api_key: str = Depends(verify_api_key)
    ):
        ...
"""

import logging
from typing import Optional
from fastapi import HTTPException, Security, Depends, Header, status
from fastapi.security import APIKeyHeader
from app.config import settings

logger = logging.getLogger(__name__)

# API Key header configuration
API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def get_valid_api_key() -> str:
    """
    Get the valid API key from settings.
    
    Returns:
        The configured API key
        
    Raises:
        ValueError: If API key is not configured
    """
    api_key = settings.ADMIN_API_KEY
    if not api_key:
        logger.warning("ADMIN_API_KEY not configured in environment variables")
        # Return a default key for development - CHANGE IN PRODUCTION
        return "dev-api-key-change-me"
    return api_key


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Verify the API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        The verified API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    valid_key = get_valid_api_key()
    
    if api_key != valid_key:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. Access denied.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    logger.debug("API key verified successfully")
    return api_key


def get_api_key_header() -> APIKeyHeader:
    """
    Get the API key header security scheme for OpenAPI docs.
    
    Returns:
        APIKeyHeader instance
    """
    return api_key_header


class APIKeyDependency:
    """
    Dependency class for API key verification with optional bypass.
    
    Usage:
        # Required authentication
        @router.post("/protected")
        async def protected_endpoint(auth: str = Depends(APIKeyDependency())):
            ...
        
        # Optional authentication (for mixed access)
        @router.get("/mixed")  
        async def mixed_endpoint(auth: str = Depends(APIKeyDependency(required=False))):
            if auth:
                # Authenticated access
            else:
                # Public access
    """
    
    def __init__(self, required: bool = True):
        """
        Initialize the dependency.
        
        Args:
            required: If True, raises error when API key is missing/invalid.
                     If False, returns None for invalid keys.
        """
        self.required = required
    
    async def __call__(
        self,
        api_key: Optional[str] = Security(api_key_header)
    ) -> Optional[str]:
        """
        Verify API key based on required setting.
        
        Args:
            api_key: API key from header
            
        Returns:
            Verified API key or None if not required
        """
        if not api_key:
            if self.required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required. Please provide X-API-Key header.",
                    headers={"WWW-Authenticate": "ApiKey"}
                )
            return None
        
        valid_key = get_valid_api_key()
        
        if api_key != valid_key:
            if self.required:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid API key. Access denied.",
                    headers={"WWW-Authenticate": "ApiKey"}
                )
            return None
        
        return api_key


# Endpoint to verify API key (for frontend validation)
async def check_api_key(api_key: str = Security(api_key_header)) -> dict:
    """
    Check if provided API key is valid.
    
    Returns:
        Success response if valid
    """
    await verify_api_key(api_key)
    return {"valid": True, "message": "API key is valid"}
