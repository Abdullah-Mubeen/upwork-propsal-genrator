"""
Portfolio Repository - Lean 8-field schema for RAG-optimized retrieval.

Each portfolio item = 1 vector in Pinecone.
No multi-chunk strategy - single embedding per project.
"""
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.infra.mongodb.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class PortfolioRepository(BaseRepository[Dict[str, Any]]):
    """
    Lean portfolio items for proposal generation.
    
    8 Core Fields:
    - project_title: What you built
    - deliverables: List of specific things delivered
    - skills: Tech stack used
    - outcome: Key result achieved
    - portfolio_url: Link to show work
    - industry: SaaS, FinTech, etc. (optional)
    - client_feedback: Testimonial text (optional)
    - duration_days: How long it took (optional)
    """
    
    collection_name = "portfolio_items"
    
    def create(
        self,
        org_id: str,
        profile_id: str,  # Which profile this belongs to
        project_title: str,
        deliverables: List[str],
        skills: List[str],
        outcome: str,
        portfolio_url: str = None,
        industry: str = None,
        client_feedback: str = None,
        duration_days: int = None
    ) -> Dict[str, Any]:
        """Create a lean portfolio entry."""
        item_id = f"port_{uuid.uuid4().hex[:12]}"
        
        doc = {
            "item_id": item_id,
            "org_id": org_id,
            "profile_id": profile_id,
            # Core 8 fields
            "project_title": project_title.strip(),
            "deliverables": [d.strip() for d in deliverables if d],
            "skills": [s.strip() for s in skills if s],
            "outcome": outcome.strip() if outcome else None,
            "portfolio_url": portfolio_url,
            "industry": industry,
            "client_feedback": client_feedback,
            "duration_days": duration_days,
            # Embedding tracking
            "embedding_id": None,  # Set after Pinecone upsert
            "is_embedded": False,
            # Timestamps
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        db_id = self.insert_one(doc)
        logger.info(f"Created portfolio item: {item_id} for profile {profile_id}")
        return {"item_id": item_id, "db_id": db_id}
    
    def get_by_item_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self.find_one({"item_id": item_id})
    
    def list_by_profile(self, profile_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all portfolio items for a profile."""
        results = list(self.collection.find({"profile_id": profile_id}).limit(limit))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    
    def list_by_org(self, org_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all portfolio items for an organization."""
        results = list(self.collection.find({"org_id": org_id}).limit(limit))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    
    def list_pending_embedding(self, org_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get items that need embedding."""
        query = {"is_embedded": False}
        if org_id:
            query["org_id"] = org_id
        results = list(self.collection.find(query).limit(limit))
        for r in results:
            r["_id"] = str(r["_id"])
        return results
    
    def mark_embedded(self, item_id: str, embedding_id: str) -> bool:
        """Mark item as embedded in Pinecone."""
        result = self.collection.update_one(
            {"item_id": item_id},
            {"$set": {
                "embedding_id": embedding_id,
                "is_embedded": True,
                "embedded_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0
    
    def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update portfolio item (re-embedding needed)."""
        updates["updated_at"] = datetime.utcnow()
        updates["is_embedded"] = False  # Needs re-embedding
        result = self.collection.update_one(
            {"item_id": item_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    def delete(self, item_id: str) -> bool:
        """Delete portfolio item."""
        result = self.collection.delete_one({"item_id": item_id})
        return result.deleted_count > 0
    
    def build_embedding_text(self, item: Dict[str, Any]) -> str:
        """Build text for embedding - single vector per project."""
        parts = [
            f"Project: {item.get('project_title', '')}",
            f"Deliverables: {', '.join(item.get('deliverables', []))}",
            f"Skills: {', '.join(item.get('skills', []))}",
        ]
        if item.get("outcome"):
            parts.append(f"Outcome: {item['outcome']}")
        if item.get("industry"):
            parts.append(f"Industry: {item['industry']}")
        if item.get("client_feedback"):
            parts.append(f"Feedback: {item['client_feedback']}")
        
        return "\n".join(parts)


# Singleton
_portfolio_repo: Optional[PortfolioRepository] = None

def get_portfolio_repo() -> PortfolioRepository:
    global _portfolio_repo
    if _portfolio_repo is None:
        _portfolio_repo = PortfolioRepository()
    return _portfolio_repo
