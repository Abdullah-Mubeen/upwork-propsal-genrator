"""
Simplified Auth Middleware - 3 Roles Only

- Super Admin: Master key from .env - manages keys
- Admin: Full access (generate, read, write, training)  
- User: Generate proposals only
"""
import hashlib
from datetime import datetime
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def get_master_key() -> str:
    return settings.ADMIN_API_KEY or "dev-key"

async def verify_key(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Verify any valid key - returns role info"""
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key required")
    
    # Super Admin check
    if api_key == get_master_key():
        return {"role": "super_admin", "name": "Super Admin", "permissions": ["admin", "generate", "read", "write", "training"]}
    
    # Check DB for generated key
    from app.db import get_db
    db = get_db()
    doc = db.db["api_keys"].find_one({"key_hash": hash_key(api_key)})
    
    if not doc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid API key")
    if not doc.get("is_active", True):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Key revoked")
    if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Key expired")
    
    # Update last used
    db.db["api_keys"].update_one({"_id": doc["_id"]}, {"$set": {"last_used": datetime.utcnow()}})
    
    role = doc.get("role", "user")
    perms = ["admin", "generate", "read", "write", "training"] if role == "admin" else ["generate"]
    return {"role": role, "name": doc.get("name", "User"), "permissions": perms}

async def verify_super_admin(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Only master key from .env"""
    if not api_key or api_key != get_master_key():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Super Admin access required")
    return {"role": "super_admin", "name": "Super Admin"}

async def verify_admin(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Admin or Super Admin"""
    result = await verify_key(api_key)
    if result["role"] not in ("admin", "super_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return result

async def verify_user(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Any valid key (user, admin, super_admin)"""
    return await verify_key(api_key)

# Aliases for backward compatibility
verify_api_key = verify_key
verify_admin_key = verify_super_admin
verify_generate_permission = verify_user
verify_training_permission = verify_admin
verify_write_permission = verify_admin
