"""
API Key Authentication Middleware

Security layer with permission-based access control.
- Master Admin: Full access (ADMIN_API_KEY from env)
- Generated Keys: Custom permissions, expiry, usage limits

Permissions: read, write, generate, training, admin
"""

import logging
import hashlib
from typing import Optional, List
from datetime import datetime
from fastapi import HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader
from app.config import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def hash_key(key: str) -> str:
    """SHA-256 hash of API key"""
    return hashlib.sha256(key.encode()).hexdigest()


def get_master_key() -> str:
    """Get master admin key from settings"""
    return settings.ADMIN_API_KEY or "dev-api-key-change-me"


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    required_permissions: List[str] = None
) -> dict:
    """
    Verify API key and check permissions.
    
    Returns dict with: key_prefix, permissions, is_admin
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Please provide X-API-Key header."
        )
    
    # Check if master admin key
    if api_key == get_master_key():
        return {"key_prefix": "master", "permissions": ["admin", "read", "write", "generate", "training"], "is_admin": True}
    
    # Check database for generated key
    from app.db import get_db
    db = get_db()
    
    key_hash = hash_key(api_key)
    key_doc = db.db["api_keys"].find_one({"key_hash": key_hash})
    
    if not key_doc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key. Access denied.")
    
    # Check if active
    if not key_doc.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key has been revoked.")
    
    # Check expiry
    if key_doc.get("expires_at") and key_doc["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key has expired.")
    
    # Check usage limit
    if key_doc.get("usage_limit") and key_doc.get("used_count", 0) >= key_doc["usage_limit"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key usage limit exceeded.")
    
    # Check required permissions
    key_permissions = key_doc.get("permissions", [])
    if required_permissions:
        if "admin" not in key_permissions:  # Admin bypasses permission checks
            for perm in required_permissions:
                if perm not in key_permissions:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied. Required: {perm}"
                    )
    
    # Update usage stats
    db.db["api_keys"].update_one(
        {"_id": key_doc["_id"]},
        {"$inc": {"used_count": 1}, "$set": {"last_used": datetime.utcnow()}}
    )
    
    return {
        "key_prefix": key_doc.get("key_prefix", api_key[:8]),
        "permissions": key_permissions,
        "is_admin": "admin" in key_permissions
    }


async def verify_super_admin_key(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """
    Verify ONLY the master admin key from .env.
    Generated keys with 'admin' permission do NOT have super admin access.
    There is only ONE super admin.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Please provide X-API-Key header."
        )
    
    if api_key != get_master_key():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super Admin access required. Only the master admin can access this."
        )
    
    return {"key_prefix": "master", "permissions": ["admin", "read", "write", "generate", "training"], "is_admin": True, "is_super_admin": True}


# Alias for backward compatibility
verify_admin_key = verify_super_admin_key


async def verify_generate_permission(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Verify key has generate permission"""
    return await verify_api_key(api_key, required_permissions=["generate"])


async def verify_training_permission(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Verify key has training permission"""
    return await verify_api_key(api_key, required_permissions=["training"])


async def verify_write_permission(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Verify key has write permission"""
    return await verify_api_key(api_key, required_permissions=["write"])


class APIKeyDependency:
    """Dependency class for API key verification with required permissions."""
    
    def __init__(self, required: bool = True, permissions: List[str] = None):
        self.required = required
        self.permissions = permissions or []
    
    async def __call__(self, api_key: Optional[str] = Security(api_key_header)) -> Optional[dict]:
        if not api_key and not self.required:
            return None
        return await verify_api_key(api_key, required_permissions=self.permissions)


async def check_api_key(api_key: str = Security(api_key_header)) -> dict:
    """Check if provided API key is valid."""
    result = await verify_api_key(api_key)
    return {"valid": True, "permissions": result["permissions"], "is_admin": result["is_admin"]}
