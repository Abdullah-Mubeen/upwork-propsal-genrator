"""
Multi-Tenant Repositories

Organization and User repositories for multi-tenant RBAC system.
Replaces user_profile collection with proper user management.

Roles:
- super_admin: Platform-wide admin (from env var)
- admin: Organization admin (full org access)
- member: Regular user (generate proposals only)
"""
import logging
import uuid
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from app.infra.mongodb.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    SUPER_ADMIN = "super_admin"  # Platform admin
    ADMIN = "admin"             # Organization admin
    MEMBER = "member"           # Regular user

    @property
    def permissions(self) -> List[str]:
        """Get permissions for this role."""
        if self == UserRole.SUPER_ADMIN:
            return ["admin", "generate", "read", "write", "training", "manage_org"]
        elif self == UserRole.ADMIN:
            return ["admin", "generate", "read", "write", "training"]
        return ["generate"]


class OrganizationRepository(BaseRepository[Dict[str, Any]]):
    """Repository for organizations (tenants)."""
    
    collection_name = "organizations"
    
    def create(self, name: str, settings: Dict[str, Any] = None) -> Dict[str, str]:
        """Create a new organization."""
        org_id = f"org_{uuid.uuid4().hex[:12]}"
        doc = {
            "org_id": org_id,
            "name": name,
            "settings": settings or {},
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        db_id = self.insert_one(doc)
        logger.info(f"Created organization: {org_id} ({name})")
        return {"org_id": org_id, "db_id": db_id}
    
    def get_by_org_id(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get organization by org_id."""
        return self.find_one({"org_id": org_id})
    
    def update_settings(self, org_id: str, settings: Dict[str, Any]) -> bool:
        """Update organization settings."""
        result = self.collection.update_one(
            {"org_id": org_id},
            {"$set": {"settings": settings, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    def list_all(self, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List all organizations (super_admin only)."""
        results = list(self.collection.find({"is_active": True}).skip(skip).limit(limit))
        for r in results:
            r["_id"] = str(r["_id"])
        return results


class UserRepository(BaseRepository[Dict[str, Any]]):
    """Repository for users with RBAC."""
    
    collection_name = "users"
    
    @staticmethod
    def hash_key(key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def create(
        self,
        org_id: str,
        email: str,
        name: str,
        role: UserRole = UserRole.MEMBER
    ) -> Dict[str, Any]:
        """Create a new user with auto-generated API key."""
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        api_key = f"upk_{uuid.uuid4().hex}"  # upk = upwork proposal key
        
        doc = {
            "user_id": user_id,
            "org_id": org_id,
            "email": email.lower(),
            "name": name,
            "role": role.value if isinstance(role, UserRole) else role,
            "api_key_hash": self.hash_key(api_key),
            "api_key_prefix": api_key[:8],  # For identification
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        db_id = self.insert_one(doc)
        logger.info(f"Created user: {user_id} ({email}) in org {org_id}")
        
        # Return with unhashed key (only time it's visible)
        return {
            "user_id": user_id,
            "db_id": db_id,
            "api_key": api_key,  # Show once, never stored
            "org_id": org_id,
            "email": email,
            "name": name,
            "role": doc["role"]
        }
    
    def get_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by API key."""
        key_hash = self.hash_key(api_key)
        user = self.find_one({"api_key_hash": key_hash, "is_active": True})
        if user:
            # Update last login
            self.collection.update_one(
                {"user_id": user["user_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        return user
    
    def get_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id."""
        return self.find_one({"user_id": user_id})
    
    def get_by_email(self, email: str, org_id: str = None) -> Optional[Dict[str, Any]]:
        """Get user by email (optionally scoped to org)."""
        query = {"email": email.lower()}
        if org_id:
            query["org_id"] = org_id
        return self.find_one(query)
    
    def list_by_org(self, org_id: str) -> List[Dict[str, Any]]:
        """List all users in an organization."""
        results = list(self.collection.find({"org_id": org_id, "is_active": True}))
        for r in results:
            r["_id"] = str(r["_id"])
            r.pop("api_key_hash", None)  # Never expose
        return results
    
    def update_role(self, user_id: str, role: UserRole) -> bool:
        """Update user role (admin action)."""
        result = self.collection.update_one(
            {"user_id": user_id},
            {"$set": {"role": role.value if isinstance(role, UserRole) else role}}
        )
        return result.modified_count > 0
    
    def deactivate(self, user_id: str) -> bool:
        """Deactivate user (soft delete)."""
        result = self.collection.update_one(
            {"user_id": user_id},
            {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """Generate new API key for user."""
        new_key = f"upk_{uuid.uuid4().hex}"
        result = self.collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "api_key_hash": self.hash_key(new_key),
                "api_key_prefix": new_key[:8],
                "key_regenerated_at": datetime.utcnow()
            }}
        )
        return new_key if result.modified_count > 0 else None


# Singleton instances
_org_repo: Optional[OrganizationRepository] = None
_user_repo: Optional[UserRepository] = None


def get_org_repo() -> OrganizationRepository:
    """Get singleton OrganizationRepository."""
    global _org_repo
    if _org_repo is None:
        _org_repo = OrganizationRepository()
    return _org_repo


def get_user_repo() -> UserRepository:
    """Get singleton UserRepository."""
    global _user_repo
    if _user_repo is None:
        _user_repo = UserRepository()
    return _user_repo
