"""
MongoDB Repositories - Domain-specific data access.

Repository Pattern Implementation:
- TrainingDataRepository, ChunkRepository, EmbeddingRepository - Training data pipeline
- ProposalRepository, SentProposalRepository, FeedbackRepository - Proposal lifecycle
- AnalyticsRepository - Cross-collection statistics and metrics
- SkillRepository, ProfileRepository, AdminRepository - Supporting entities
"""

# Training Data repositories
from app.infra.mongodb.repositories.training_repo import (
    TrainingDataRepository,
    ChunkRepository,
    EmbeddingRepository,
    EmbeddingCacheRepository,
    get_training_repo,
    get_chunk_repo,
    get_embedding_repo,
    get_cache_repo,
)

# Proposal repositories
from app.infra.mongodb.repositories.proposal_repo import (
    ProposalRepository,
    SentProposalRepository,
    FeedbackRepository,
    get_proposal_repo,
    get_sent_proposal_repo,
    get_feedback_repo,
)

# Analytics and supporting repositories
from app.infra.mongodb.repositories.analytics_repo import (
    AnalyticsRepository,
    SkillRepository,
    SkillEmbeddingRepository,
    ProfileRepository,
    AdminRepository,
    get_analytics_repo,
    get_skill_repo,
    get_skill_embedding_repo,
    get_profile_repo,
    get_admin_repo,
)

# Multi-tenant repositories
from app.infra.mongodb.repositories.tenant_repo import (
    OrganizationRepository,
    UserRepository,
    UserRole,
    get_org_repo,
    get_user_repo,
)

__all__ = [
    # Training Data
    "TrainingDataRepository",
    "ChunkRepository",
    "EmbeddingRepository",
    "EmbeddingCacheRepository",
    "get_training_repo",
    "get_chunk_repo",
    "get_embedding_repo",
    "get_cache_repo",
    # Proposals
    "ProposalRepository",
    "SentProposalRepository",
    "FeedbackRepository",
    "get_proposal_repo",
    "get_sent_proposal_repo",
    "get_feedback_repo",
    # Analytics & Supporting
    "AnalyticsRepository",
    "SkillRepository",
    "SkillEmbeddingRepository",
    "ProfileRepository",
    "AdminRepository",
    "get_analytics_repo",
    "get_skill_repo",
    "get_skill_embedding_repo",
    "get_profile_repo",
    "get_admin_repo",
    # Multi-tenant
    "OrganizationRepository",
    "UserRepository",
    "UserRole",
    "get_org_repo",
    "get_user_repo",
]
