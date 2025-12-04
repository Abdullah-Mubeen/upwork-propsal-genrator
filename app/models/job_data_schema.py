"""
Pydantic schemas for job data ingestion and validation

Defines request/response models for:
- Job data upload
- Chunk retrieval
- Query responses
- Error handling
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# ===================== ENUMS =====================

class ProjectStatus(str, Enum):
    """Project status enumeration"""
    COMPLETED = "completed"
    ONGOING = "ongoing"
    PENDING = "pending"


class TaskType(str, Enum):
    """Task type enumeration for project classification"""
    WEBSITE = "website"
    MOBILE_APP = "mobile_app"
    BACKEND_API = "backend_api"
    FRONTEND = "frontend"
    FULL_STACK = "full_stack"
    CONSULTATION = "consultation"
    MAINTENANCE = "maintenance"
    CONTENT = "content"
    DESIGN = "design"
    OTHER = "other"


class ChunkType(str, Enum):
    """Chunk type enumeration"""
    METADATA = "metadata"
    PROPOSAL = "proposal"
    DESCRIPTION = "description"
    FEEDBACK = "feedback"
    SUMMARY = "summary"


# ===================== REQUEST SCHEMAS =====================

class SkillsValidator:
    """Validator for skills list"""
    
    @staticmethod
    def validate_skills(skills: List[str]) -> List[str]:
        """Ensure skills are non-empty strings"""
        if not skills:
            raise ValueError("Skills list cannot be empty")
        return [s.strip() for s in skills if s.strip()]


class JobDataUploadRequest(BaseModel):
    """Request model for uploading job training data"""
    
    contract_id: Optional[str] = Field(
        None,
        description="Unique contract ID. Auto-generated if not provided"
    )
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the company"
    )
    job_title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Job title/position"
    )
    job_description: str = Field(
        ...,
        min_length=10,
        description="Full job description"
    )
    your_proposal_text: str = Field(
        ...,
        min_length=10,
        description="Your proposal that was submitted"
    )
    skills_required: List[str] = Field(
        ...,
        min_items=1,
        description="List of required skills"
    )
    industry: Optional[str] = Field(
        "general",
        max_length=100,
        description="Industry of the job"
    )
    start_date: Optional[str] = Field(
        None,
        description="Project start date (ISO format)"
    )
    end_date: Optional[str] = Field(
        None,
        description="Project end date (ISO format)"
    )
    portfolio_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="List of portfolio or past work URLs"
    )
    client_feedback: Optional[str] = Field(
        None,
        description="Client feedback or review text"
    )
    feedback_type: Optional[str] = Field(
        "text",
        description="Type of feedback: 'text' or 'image'"
    )
    feedback_image_path: Optional[str] = Field(
        None,
        description="Path or URL to feedback image (if feedback_type is 'image')"
    )
    project_status: str = Field(
        "completed",
        description="Status of the project: completed, ongoing, cancelled"
    )
    task_type: str = Field(
        "other",
        description="Type of task/project"
    )
    other_task_type: Optional[str] = Field(
        None,
        description="Custom task type if task_type is 'other'"
    )
    urgent_adhoc: bool = Field(
        False,
        description="Whether this was an urgent/adhoc project"
    )
    has_feedback: bool = Field(
        False,
        description="Whether client feedback is provided (auto-set)"
    )
    
    @validator("skills_required", pre=True)
    def validate_skills(cls, v):
        """Validate and clean skills list"""
        return SkillsValidator.validate_skills(v)
    
    @validator("has_feedback", pre=True, always=True)
    def set_has_feedback(cls, v, values):
        """Auto-set has_feedback based on client_feedback presence"""
        if "client_feedback" in values:
            return bool(values.get("client_feedback") and values["client_feedback"].strip())
        return v or False
    
    @validator("company_name", "job_title", "industry")
    def strip_whitespace(cls, v):
        """Strip whitespace from string fields"""
        if isinstance(v, str):
            return v.strip()
        return v
    
    @validator("portfolio_urls", pre=True)
    def validate_portfolio_urls(cls, v):
        """Validate portfolio URLs"""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [url.strip() for url in v if url and url.strip()]
    
    @validator("task_type")
    def validate_task_type(cls, v):
        """Validate task type"""
        if isinstance(v, str):
            return v.strip().lower()
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "company_name": "TechCorp Inc",
                "job_title": "Senior Backend Engineer",
                "job_description": "We're looking for an experienced backend engineer...",
                "your_proposal_text": "I'm excited to work on this project...",
                "skills_required": ["Python", "FastAPI", "PostgreSQL"],
                "industry": "Technology",
                "task_type": "backend_api",
                "project_status": "completed",
                "urgent_adhoc": False,
                "has_feedback": True,
                "start_date": "2024-12-01",
                "end_date": "2024-12-15",
                "portfolio_urls": ["https://github.com/yourname", "https://yourportfolio.com"],
                "client_feedback": "Great work! Very responsive.",
                "feedback_type": "text"
            }
        }


class UpdateJobDataRequest(BaseModel):
    """Request model for updating job data"""
    
    company_name: Optional[str] = Field(None, min_length=1, max_length=255)
    job_title: Optional[str] = Field(None, min_length=1, max_length=255)
    job_description: Optional[str] = Field(None, min_length=10)
    your_proposal_text: Optional[str] = Field(None, min_length=10)
    skills_required: Optional[List[str]] = Field(None, min_items=1)
    industry: Optional[str] = Field(None, max_length=100)
    client_feedback: Optional[str] = Field(None)
    feedback_type: Optional[str] = Field(None)
    feedback_image_path: Optional[str] = Field(None)
    start_date: Optional[str] = Field(None, description="Project start date (ISO format)")
    end_date: Optional[str] = Field(None, description="Project end date (ISO format)")
    portfolio_urls: Optional[List[str]] = Field(None, description="List of portfolio URLs")
    project_status: Optional[str] = Field(None)
    task_type: Optional[str] = Field(None, description="Type of task/project")
    other_task_type: Optional[str] = Field(None, description="Custom task type")
    urgent_adhoc: Optional[bool] = Field(None)
    
    @validator("skills_required", pre=True, always=True)
    def validate_skills(cls, v):
        """Validate and clean skills list if provided"""
        if v is None:
            return None
        return SkillsValidator.validate_skills(v)
    
    class Config:
        use_enum_values = True


class DeleteJobsRequest(BaseModel):
    """Request model for bulk deletion"""
    
    contract_ids: List[str] = Field(
        ...,
        min_items=1,
        description="List of contract IDs to delete"
    )


# ===================== RESPONSE SCHEMAS =====================

class JobDataResponse(BaseModel):
    """Response model for job data - includes key fields for training"""
    
    db_id: str = Field(..., description="MongoDB document ID")
    contract_id: str = Field(..., description="Unique contract ID")
    company_name: str = Field(..., description="Company name")
    job_title: str = Field(..., description="Job title/position")
    industry: str = Field(..., description="Industry sector")
    skills_required: Optional[List[str]] = Field(None, description="Required skills")
    task_type: Optional[str] = Field(None, description="Type of task/project")
    project_status: str = Field(..., description="Project status")
    urgent_adhoc: Optional[bool] = Field(None, description="Is urgent/adhoc project")
    start_date: Optional[str] = Field(None, description="Project start date")
    end_date: Optional[str] = Field(None, description="Project end date")
    portfolio_url: Optional[str] = Field(None, description="Portfolio URL")
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: Optional[datetime] = Field(None, description="Updated timestamp")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "db_id": "507f1f77bcf86cd799439011",
                "contract_id": "550e8400-e29b-41d4-a716-446655440000",
                "company_name": "TechCorp Inc",
                "job_title": "Senior Backend Engineer",
                "industry": "Technology",
                "skills_required": ["Python", "FastAPI", "PostgreSQL"],
                "task_type": "backend_api",
                "project_status": "completed",
                "urgent_adhoc": False,
                "start_date": "2025-12-01",
                "end_date": "2025-12-15",
                "portfolio_url": "https://github.com/yourname",
                "created_at": "2025-12-03T10:30:00Z",
                "updated_at": "2025-12-03T10:30:00Z"
            }
        }


class JobDataDetailResponse(JobDataResponse):
    """Detailed response model for job data"""
    
    job_description: str
    your_proposal_text: str
    skills_required: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    portfolio_url: Optional[str] = None
    client_feedback: Optional[str] = None
    task_type: Optional[str] = Field(None, description="Type of task/project")
    urgent_adhoc: bool
    chunks_count: Optional[int] = Field(None, description="Number of chunks created")
    embedded_chunks_count: Optional[int] = Field(None, description="Number of embedded chunks")


class ChunkResponse(BaseModel):
    """Response model for chunk data"""
    
    chunk_id: str
    contract_id: str
    content: str
    chunk_type: str
    priority: float
    length: int
    industry: str
    skills_required: List[str]
    company_name: str
    project_status: str
    embedding_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response for successful upload"""
    
    status: str = Field("success", description="Operation status")
    db_id: str = Field(..., description="MongoDB document ID")
    contract_id: str = Field(..., description="Unique contract ID")
    message: str = Field(..., description="Success message")
    data: JobDataResponse
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "db_id": "507f1f77bcf86cd799439011",
                "contract_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Job data uploaded successfully",
                "data": {
                    "db_id": "507f1f77bcf86cd799439011",
                    "contract_id": "550e8400-e29b-41d4-a716-446655440000",
                    "company_name": "TechCorp Inc",
                    "job_title": "Senior Backend Engineer",
                    "industry": "Technology",
                    "project_status": "completed",
                    "created_at": "2025-12-03T10:30:00Z"
                }
            }
        }


class ListResponse(BaseModel):
    """Response for list operations"""
    
    status: str = Field("success")
    total: int = Field(..., description="Total count")
    count: int = Field(..., description="Items returned")
    skip: int = Field(..., description="Items skipped")
    limit: int = Field(..., description="Limit applied")
    items: List[Any] = Field(..., description="List of items")


class DeleteResponse(BaseModel):
    """Response for deletion"""
    
    status: str = Field("success")
    deleted_count: int = Field(..., description="Number of deleted items")
    chunks_deleted: int = Field(..., description="Number of associated chunks deleted")
    message: str = Field(..., description="Success message")


class ErrorResponse(BaseModel):
    """Response for errors"""
    
    status: str = Field("error")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional details")


# ===================== STATISTICS SCHEMAS =====================

class JobStatisticsResponse(BaseModel):
    """Statistics about job data"""
    
    total_jobs: int = Field(..., description="Total job records")
    total_chunks: int = Field(..., description="Total chunks created")
    chunks_embedded: int = Field(..., description="Chunks with embeddings")
    chunks_pending: int = Field(..., description="Chunks pending embedding")
    by_industry: Dict[str, int] = Field(..., description="Jobs by industry")
    by_status: Dict[str, int] = Field(..., description="Jobs by project status")
