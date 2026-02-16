"""
Freelancer Profile Repository - For proposal personalization.

Supports:
- Manual profile data entry
- URL import (Upwork, LinkedIn, portfolio sites)
- Multiple profiles for agencies
- Single profile for individuals
"""
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from app.infra.mongodb.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ImportSource(str, Enum):
    """Where profile data was imported from."""
    MANUAL = "manual"
    UPWORK = "upwork"
    LINKEDIN = "linkedin"
    PORTFOLIO = "portfolio"  # Custom portfolio site


class FreelancerProfileRepository(BaseRepository[Dict[str, Any]]):
    """
    Freelancer profiles for proposal personalization.
    
    Individual org: 1 profile
    Agency org: Multiple profiles (one per bidder)
    """
    
    collection_name = "freelancer_profiles"
    
    def create(
        self,
        org_id: str,
        name: str,
        title: str,  # Professional title
        bio: str,
        skills: List[str],
        hourly_rate: float = None,
        years_experience: int = None,
        source: ImportSource = ImportSource.MANUAL,
        source_url: str = None
    ) -> Dict[str, Any]:
        """Create a freelancer profile."""
        profile_id = f"prof_{uuid.uuid4().hex[:12]}"
        
        doc = {
            "profile_id": profile_id,
            "org_id": org_id,
            
            # Core profile data
            "name": name.strip(),
            "title": title.strip(),
            "bio": bio.strip(),
            "skills": [s.strip() for s in skills if s],
            "hourly_rate": hourly_rate,
            "years_experience": years_experience,
            
            # Import tracking
            "source": source.value if isinstance(source, ImportSource) else source,
            "source_urls": [source_url] if source_url else [],
            "last_synced": datetime.utcnow() if source_url else None,
            
            # Embedding for bio-to-job matching
            "bio_embedding_id": None,
            
            # Activity
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        db_id = self.insert_one(doc)
        logger.info(f"Created profile: {profile_id} ({name}) in org {org_id}")
        return {"profile_id": profile_id, "db_id": db_id}
    
    def get_by_profile_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        return self.find_one({"profile_id": profile_id})
    
    def get_default_profile(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get the first/default profile for an org."""
        return self.find_one({"org_id": org_id, "is_active": True})
    
    def list_by_org(self, org_id: str) -> List[Dict[str, Any]]:
        """List all profiles for an organization."""
        results = list(self.collection.find({"org_id": org_id, "is_active": True}))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    
    def count_by_org(self, org_id: str) -> int:
        """Count profiles in an org (for individual limit check)."""
        return self.collection.count_documents({"org_id": org_id, "is_active": True})
    
    def update(self, profile_id: str, updates: Dict[str, Any]) -> bool:
        """Update profile data."""
        updates["updated_at"] = datetime.utcnow()
        result = self.collection.update_one(
            {"profile_id": profile_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    def add_source_url(self, profile_id: str, url: str) -> bool:
        """Add an imported source URL."""
        result = self.collection.update_one(
            {"profile_id": profile_id},
            {
                "$addToSet": {"source_urls": url},
                "$set": {"last_synced": datetime.utcnow()}
            }
        )
        return result.modified_count > 0
    
    def set_bio_embedding(self, profile_id: str, embedding_id: str) -> bool:
        """Set bio embedding ID after Pinecone upsert."""
        result = self.collection.update_one(
            {"profile_id": profile_id},
            {"$set": {"bio_embedding_id": embedding_id}}
        )
        return result.modified_count > 0
    
    def deactivate(self, profile_id: str) -> bool:
        """Soft delete profile."""
        result = self.collection.update_one(
            {"profile_id": profile_id},
            {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    def build_proposal_context(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Build context dict for proposal generation."""
        return {
            "freelancer_name": profile.get("name"),
            "professional_title": profile.get("title"),
            "bio_summary": profile.get("bio", "")[:500],  # First 500 chars
            "core_skills": profile.get("skills", [])[:10],  # Top 10 skills
            "hourly_rate": profile.get("hourly_rate"),
            "years_experience": profile.get("years_experience"),
        }


# Singleton
_freelancer_profile_repo: Optional[FreelancerProfileRepository] = None

def get_freelancer_profile_repo() -> FreelancerProfileRepository:
    global _freelancer_profile_repo
    if _freelancer_profile_repo is None:
        _freelancer_profile_repo = FreelancerProfileRepository()
    return _freelancer_profile_repo
