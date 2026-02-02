"""
Admin Routes - API Key Management & Activity Logging

Master admin can:
- Create/revoke API keys with custom permissions
- Set expiration, usage limits
- View activity logs
"""
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from enum import Enum

from app.middleware.auth import verify_admin_key
from app.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


# ===================== ENUMS & MODELS =====================

class Permission(str, Enum):
    READ = "read"           # View proposals, analytics
    WRITE = "write"         # Create/edit proposals
    GENERATE = "generate"   # Use AI generation
    TRAINING = "training"   # Upload training data
    # Note: No 'admin' permission - only super admin (master key) has admin access

class KeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="User's full name")
    email: Optional[str] = Field(None, max_length=200, description="User's email (optional)")
    permissions: List[Permission] = Field(default=[Permission.READ, Permission.GENERATE])
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Days until expiry (null=never)")
    usage_limit: Optional[int] = Field(None, ge=1, description="Max API calls (null=unlimited)")
    description: Optional[str] = Field(None, max_length=500)


class UpdateKeyRequest(BaseModel):
    permissions: Optional[List[Permission]] = None
    is_active: Optional[bool] = None
    usage_limit: Optional[int] = Field(None, ge=1)
    description: Optional[str] = Field(None, max_length=500)


class KeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str  # First 8 chars for identification
    permissions: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]
    usage_limit: Optional[int]
    used_count: int
    last_used: Optional[datetime]
    description: Optional[str]


# ===================== HELPERS =====================

def hash_key(key: str) -> str:
    """SHA-256 hash of API key"""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate secure random API key"""
    return secrets.token_hex(32)


# ===================== ROUTES =====================

@router.post("/keys", dependencies=[Depends(verify_admin_key)])
async def create_api_key(req: CreateKeyRequest):
    """Create new API key with specified permissions for a user"""
    db = get_db()
    
    raw_key = generate_api_key()
    key_hash = hash_key(raw_key)
    
    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)
    
    key_doc = {
        "key_hash": key_hash,
        "key_prefix": raw_key[:8],
        "name": req.name,  # User's full name
        "email": req.email,  # User's email (optional)
        "permissions": [p.value for p in req.permissions],
        "is_active": True,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "usage_limit": req.usage_limit,
        "used_count": 0,
        "last_used": None,
        "description": req.description
    }
    
    result = db.db["api_keys"].insert_one(key_doc)
    
    # Log activity
    db.log_activity("Super Admin", "create_user", req.name, {"email": req.email, "permissions": key_doc["permissions"]})
    
    return {
        "success": True,
        "message": "API key created. Save this key - it won't be shown again!",
        "api_key": raw_key,
        "key_id": str(result.inserted_id),
        "name": req.name,
        "permissions": key_doc["permissions"],
        "expires_at": expires_at
    }


@router.get("/keys", dependencies=[Depends(verify_admin_key)])
async def list_api_keys():
    """List all users with API access"""
    db = get_db()
    
    keys = list(db.db["api_keys"].find({}, {"key_hash": 0}).sort("created_at", -1))
    
    result = []
    for k in keys:
        # Check if expired
        status = "active"
        if not k.get("is_active"):
            status = "revoked"
        elif k.get("expires_at") and k["expires_at"] < datetime.utcnow():
            status = "expired"
        
        result.append({
            "id": str(k["_id"]),
            "name": k["name"],
            "email": k.get("email"),
            "key_prefix": k.get("key_prefix", "********"),
            "permissions": k["permissions"],
            "status": status,
            "is_active": k.get("is_active", True),
            "created_at": k["created_at"],
            "expires_at": k.get("expires_at"),
            "usage_limit": k.get("usage_limit"),
            "used_count": k.get("used_count", 0),
            "last_used": k.get("last_used"),
            "description": k.get("description")
        })
    
    return {"success": True, "keys": result, "total": len(result)}


@router.get("/keys/{key_id}", dependencies=[Depends(verify_admin_key)])
async def get_api_key(key_id: str):
    """Get details of specific API key"""
    db = get_db()
    
    from bson import ObjectId
    try:
        key = db.db["api_keys"].find_one({"_id": ObjectId(key_id)}, {"key_hash": 0})
    except:
        raise HTTPException(status_code=400, detail="Invalid key ID")
    
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    return {
        "success": True,
        "key": {
            "id": str(key["_id"]),
            "name": key["name"],
            "key_prefix": key.get("key_prefix", "********"),
            "permissions": key["permissions"],
            "is_active": key.get("is_active", True),
            "created_at": key["created_at"],
            "expires_at": key.get("expires_at"),
            "usage_limit": key.get("usage_limit"),
            "used_count": key.get("used_count", 0),
            "last_used": key.get("last_used"),
            "description": key.get("description")
        }
    }


@router.put("/keys/{key_id}", dependencies=[Depends(verify_admin_key)])
async def update_api_key(key_id: str, req: UpdateKeyRequest):
    """Update API key permissions or status"""
    db = get_db()
    
    from bson import ObjectId
    try:
        oid = ObjectId(key_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid key ID")
    
    update = {"$set": {"updated_at": datetime.utcnow()}}
    
    if req.permissions is not None:
        update["$set"]["permissions"] = [p.value for p in req.permissions]
    if req.is_active is not None:
        update["$set"]["is_active"] = req.is_active
    if req.usage_limit is not None:
        update["$set"]["usage_limit"] = req.usage_limit
    if req.description is not None:
        update["$set"]["description"] = req.description
    
    result = db.db["api_keys"].update_one({"_id": oid}, update)
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    
    # Get user name for logging
    key_data = db.db["api_keys"].find_one({"_id": oid})
    user_name = key_data.get("name", "Unknown") if key_data else "Unknown"
    db.log_activity("Super Admin", "update_user", user_name, {"changes": list(update["$set"].keys())})
    
    return {"success": True, "message": "Key updated"}


@router.delete("/keys/{key_id}", dependencies=[Depends(verify_admin_key)])
async def revoke_api_key(key_id: str):
    """Revoke (soft delete) an API key"""
    db = get_db()
    
    from bson import ObjectId
    try:
        oid = ObjectId(key_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid key ID")
    
    # Get user name before revoking
    key_data = db.db["api_keys"].find_one({"_id": oid})
    user_name = key_data.get("name", "Unknown") if key_data else "Unknown"
    
    result = db.db["api_keys"].update_one(
        {"_id": oid},
        {"$set": {"is_active": False, "revoked_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    
    db.log_activity("Super Admin", "revoke_user", user_name, {})
    
    return {"success": True, "message": "Key revoked"}


@router.get("/activity", dependencies=[Depends(verify_admin_key)])
async def get_activity_log(
    limit: int = 100,
    action: Optional[str] = None,
    user_name: Optional[str] = None
):
    """Get activity log with optional filters"""
    db = get_db()
    
    query = {}
    if action:
        query["action"] = action
    if user_name:
        query["user_name"] = {"$regex": user_name, "$options": "i"}
    
    logs = list(db.db["activity_log"].find(query).sort("timestamp", -1).limit(limit))
    
    return {
        "success": True,
        "logs": [{
            "id": str(l["_id"]),
            "user_name": l.get("user_name", l.get("key_prefix", "System")),
            "action": l["action"],
            "target": l.get("target", l.get("resource")),
            "details": l.get("details"),
            "ip": l.get("ip"),
            "timestamp": l["timestamp"]
        } for l in logs],
        "total": len(logs)
    }


@router.get("/stats", dependencies=[Depends(verify_admin_key)])
async def get_admin_stats():
    """Get admin dashboard statistics"""
    db = get_db()
    
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)
    
    total_keys = db.db["api_keys"].count_documents({})
    active_keys = db.db["api_keys"].count_documents({"is_active": True})
    
    # Activity stats
    today_actions = db.db["activity_log"].count_documents({"timestamp": {"$gte": today}})
    week_actions = db.db["activity_log"].count_documents({"timestamp": {"$gte": week_ago}})
    
    # Top actions
    pipeline = [
        {"$match": {"timestamp": {"$gte": week_ago}}},
        {"$group": {"_id": "$action", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    top_actions = list(db.db["activity_log"].aggregate(pipeline))
    
    # Most active keys
    pipeline = [
        {"$match": {"timestamp": {"$gte": week_ago}}},
        {"$group": {"_id": "$key_prefix", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    active_users = list(db.db["activity_log"].aggregate(pipeline))
    
    return {
        "success": True,
        "keys": {"total": total_keys, "active": active_keys},
        "activity": {"today": today_actions, "week": week_actions},
        "top_actions": [{"action": a["_id"], "count": a["count"]} for a in top_actions],
        "active_users": [{"key_prefix": u["_id"], "count": u["count"]} for u in active_users]
    }
