"""
Job Data Ingestion Routes

API endpoints for:
- Upload job training data
- Retrieve job data
- Delete job data
- Get chunks
- Statistics
"""
import logging
from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional

from app.models.job_data_schema import (
    JobDataUploadRequest,
    UpdateJobDataRequest,
    DeleteJobsRequest,
    JobDataResponse,
    JobDataDetailResponse,
    ChunkResponse,
    UploadResponse,
    ListResponse,
    DeleteResponse,
    ErrorResponse,
    JobStatisticsResponse
)
from app.utils.job_data_processor import JobDataProcessor
from app.db import get_db
from app.utils.openai_service import OpenAIService
from app.utils.data_chunker import DataChunker
from app.utils.feedback_processor import FeedbackProcessor
from app.utils.pinecone_service import PineconeService
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Service instances (lazy loaded)
_processor: Optional[JobDataProcessor] = None
_pinecone_service: Optional[PineconeService] = None


def get_pinecone_service() -> Optional[PineconeService]:
    """Get or initialize Pinecone service"""
    global _pinecone_service
    if _pinecone_service is None:
        try:
            if settings.PINECONE_API_KEY:
                _pinecone_service = PineconeService(
                    api_key=settings.PINECONE_API_KEY,
                    environment=settings.PINECONE_ENVIRONMENT,
                    index_name=settings.PINECONE_INDEX_NAME,
                    namespace=settings.PINECONE_NAMESPACE,
                    dimension=settings.PINECONE_DIMENSION
                )
                logger.info("Pinecone service initialized successfully")
            else:
                logger.warning("PINECONE_API_KEY not configured, Pinecone features disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone service: {str(e)}")
    return _pinecone_service


def get_processor() -> JobDataProcessor:
    """Get or initialize job data processor"""
    global _processor
    if _processor is None:
        db = get_db()
        openai_service = OpenAIService(
            api_key=settings.OPENAI_API_KEY,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL,
            llm_model=settings.OPENAI_LLM_MODEL,
            vision_model=settings.OPENAI_VISION_MODEL
        )
        chunker = DataChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
            max_chunk_size=settings.MAX_CHUNK_SIZE
        )
        feedback_processor = FeedbackProcessor(openai_service)
        pinecone_service = get_pinecone_service()
        
        _processor = JobDataProcessor(
            db=db,
            openai_service=openai_service,
            chunker=chunker,
            feedback_processor=feedback_processor,
            pinecone_service=pinecone_service
        )
    return _processor


# ===================== UPLOAD ENDPOINTS =====================

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload job training data",
    responses={
        201: {"description": "Job data uploaded successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def upload_job_data(job_data: JobDataUploadRequest):
    """
    Upload new job training data for proposal generation training
    
    Complete end-to-end pipeline:
    1. Store job data in MongoDB
    2. Create smart chunks (metadata, proposal, description, feedback, summary)
    3. Generate embeddings for semantic search
    4. Save feedback to feedback_data collection
    5. Save vectors to Pinecone for AI retrieval
    
    This endpoint accepts complete job information:
    - Company and job details
    - Your proposal that was submitted
    - Skills required
    - Client feedback (optional - can be text or extracted from image)
    - Portfolio URL
    
    **Auto-generated fields:**
    - `contract_id` (if not provided, format: job_<short_id>)
    - `created_at`, `updated_at`
    
    **Returns:**
    - `contract_id`: Unique identifier for this job record
    - `db_id`: MongoDB document ID
    - Complete job data with statistics
    """
    try:
        processor = get_processor()
        
        # Convert request to dict
        job_dict = job_data.model_dump(exclude_unset=False)
        
        # Execute complete pipeline: store → chunks → embeddings → feedback → pinecone
        logger.info("🚀 Starting complete training pipeline...")
        pipeline_result = processor.process_complete_pipeline(job_dict, save_to_pinecone=True)
        
        contract_id = pipeline_result["contract_id"]
        
        # Get stored data for response
        stored_job = processor.get_job_with_chunks(contract_id)
        
        response_data = JobDataResponse(
            db_id=pipeline_result["db_id"],
            contract_id=contract_id,
            company_name=stored_job["company_name"],
            job_title=stored_job["job_title"],
            industry=stored_job["industry"],
            skills_required=stored_job.get("skills_required"),
            task_type=stored_job.get("task_type"),
            project_status=stored_job["project_status"],
            urgent_adhoc=stored_job.get("urgent_adhoc", False),
            start_date=stored_job.get("start_date"),
            end_date=stored_job.get("end_date"),
            portfolio_url=stored_job.get("portfolio_url"),
            created_at=stored_job["created_at"],
            updated_at=stored_job.get("updated_at")
        )
        
        # Log with complete pipeline statistics
        logger.info(f"✅ Complete Pipeline Success: {contract_id} | Chunks: {pipeline_result['chunks_created']} | Embeddings: {pipeline_result['embeddings_created']} | Pinecone: {pipeline_result['pinecone_vectors']} | Feedback: {pipeline_result['feedback_saved']}")
        
        return UploadResponse(
            status="success",
            db_id=pipeline_result["db_id"],
            contract_id=contract_id,
            message=f"Job data uploaded successfully! Processed: {pipeline_result['chunks_created']} chunks, {pipeline_result['embeddings_created']} embeddings, saved to {pipeline_result['pinecone_vectors']} Pinecone vectors",
            data=response_data
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading job data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload job data"
        )


# ===================== RETRIEVE ENDPOINTS =====================

@router.get(
    "/{contract_id}",
    response_model=JobDataDetailResponse,
    summary="Get job data by contract ID",
    responses={
        200: {"description": "Job data retrieved"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_job_data(contract_id: str):
    """
    Retrieve detailed job data by contract ID
    
    **Returns:**
    - Complete job information
    - Chunk statistics (total chunks and embedded chunks)
    - All metadata and feedback
    """
    try:
        processor = get_processor()
        
        job_data = processor.get_job_with_chunks(contract_id)
        
        return JobDataDetailResponse(
            db_id=str(job_data["_id"]),
            contract_id=job_data["contract_id"],
            company_name=job_data["company_name"],
            job_title=job_data["job_title"],
            job_description=job_data["job_description"],
            your_proposal_text=job_data["your_proposal_text"],
            skills_required=job_data["skills_required"],
            industry=job_data["industry"],
            project_status=job_data["project_status"],
            start_date=job_data.get("start_date"),
            end_date=job_data.get("end_date"),
            portfolio_url=job_data.get("portfolio_url"),
            client_feedback=job_data.get("client_feedback"),
            task_type=job_data.get("task_type"),
            urgent_adhoc=job_data.get("urgent_adhoc", False),
            created_at=job_data["created_at"],
            updated_at=job_data.get("updated_at"),
            chunks_count=job_data.get("chunks_count"),
            embedded_chunks_count=job_data.get("embedded_chunks_count")
        )
    
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job not found: {contract_id}")
    except Exception as e:
        logger.error(f"Error retrieving job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job data")


@router.get(
    "",
    response_model=ListResponse,
    summary="List all job data",
    responses={
        200: {"description": "Jobs retrieved"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def list_job_data(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    status: Optional[str] = Query(None, description="Filter by project status"),
    company: Optional[str] = Query(None, description="Search by company name")
):
    """
    List all job training data with optional filtering and pagination
    
    **Query Parameters:**
    - `skip`: Pagination offset (default: 0)
    - `limit`: Items per page (default: 50, max: 100)
    - `industry`: Filter by industry (optional)
    - `status`: Filter by project status (optional)
    - `company`: Search by company name (optional)
    
    **Returns:**
    - List of job summaries
    - Total count
    - Pagination info
    """
    try:
        processor = get_processor()
        db = get_db()
        
        # Build filter
        filters = {}
        if industry:
            filters["industry"] = industry
        if status:
            filters["project_status"] = status
        if company:
            filters["company_name"] = {"$regex": company, "$options": "i"}
        
        # Get jobs
        jobs = db.get_all_training_data(skip=skip, limit=limit, filters=filters)
        total = db.db["training_data"].count_documents(filters)
        
        items = [
            JobDataResponse(
                db_id=str(job["_id"]),
                contract_id=job["contract_id"],
                company_name=job["company_name"],
                job_title=job["job_title"],
                industry=job["industry"],
                skills_required=job.get("skills_required"),
                task_type=job.get("task_type"),
                project_status=job["project_status"],
                urgent_adhoc=job.get("urgent_adhoc", False),
                start_date=job.get("start_date"),
                end_date=job.get("end_date"),
                portfolio_url=job.get("portfolio_url"),
                created_at=job["created_at"],
                updated_at=job.get("updated_at")
            )
            for job in jobs
        ]
        
        return ListResponse(
            status="success",
            total=total,
            count=len(items),
            skip=skip,
            limit=limit,
            items=items
        )
    
    except Exception as e:
        logger.error(f"Error listing job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")


# ===================== CHUNKS ENDPOINTS =====================

@router.get(
    "/{contract_id}/chunks",
    response_model=ListResponse,
    summary="Get chunks for a contract",
    responses={
        200: {"description": "Chunks retrieved"},
        404: {"model": ErrorResponse, "description": "Contract not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_job_chunks(
    contract_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type")
):
    """
    Get chunks created from a job
    
    **Path Parameters:**
    - `contract_id`: Contract ID
    
    **Query Parameters:**
    - `skip`: Pagination offset
    - `limit`: Items per page
    - `chunk_type`: Filter by type (metadata, proposal, description, feedback, summary)
    
    **Returns:**
    - List of chunks with content and metadata
    """
    try:
        processor = get_processor()
        db = get_db()
        
        # Check if job exists
        job_data = db.get_training_data(contract_id)
        if not job_data:
            raise ValueError(f"Contract not found: {contract_id}")
        
        # Get chunks
        all_chunks = db.get_chunks_by_contract(contract_id)
        
        # Filter by type if provided
        if chunk_type:
            all_chunks = [c for c in all_chunks if c["chunk_type"] == chunk_type]
        
        # Paginate
        total = len(all_chunks)
        chunks = all_chunks[skip:skip + limit]
        
        items = [
            ChunkResponse(
                chunk_id=chunk["chunk_id"],
                contract_id=chunk["contract_id"],
                content=chunk["content"],
                chunk_type=chunk["chunk_type"],
                priority=chunk["priority"],
                length=chunk["length"],
                industry=chunk["industry"],
                skills_required=chunk["skills_required"],
                company_name=chunk["company_name"],
                project_status=chunk["project_status"],
                embedding_status=chunk["embedding_status"],
                created_at=chunk["created_at"]
            )
            for chunk in chunks
        ]
        
        return ListResponse(
            status="success",
            total=total,
            count=len(items),
            skip=skip,
            limit=limit,
            items=items
        )
    
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chunks")


# ===================== DELETE ENDPOINTS =====================

@router.delete(
    "/{contract_id}",
    response_model=DeleteResponse,
    summary="Delete job data by contract ID",
    responses={
        200: {"description": "Job deleted"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def delete_job_data(contract_id: str):
    """
    Delete job data and all associated chunks
    
    **Path Parameters:**
    - `contract_id`: Contract ID to delete
    
    **Cascading Deletes:**
    - Training data record
    - All chunks created from this job
    - All embeddings for those chunks
    
    **Returns:**
    - Count of deleted items
    """
    try:
        processor = get_processor()
        
        jobs_deleted, chunks_deleted = processor.delete_job_and_chunks(contract_id)
        
        return DeleteResponse(
            status="success",
            deleted_count=jobs_deleted,
            chunks_deleted=chunks_deleted,
            message=f"Deleted {jobs_deleted} job and {chunks_deleted} associated chunks"
        )
    
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Contract not found: {contract_id}")
    except Exception as e:
        logger.error(f"Error deleting job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete job data")


@router.post(
    "/bulk/delete",
    response_model=DeleteResponse,
    summary="Bulk delete jobs",
    responses={
        200: {"description": "Jobs deleted"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def bulk_delete_jobs(request: DeleteJobsRequest):
    """
    Delete multiple jobs by contract IDs
    
    **Body:**
    - `contract_ids`: List of contract IDs to delete
    
    **Returns:**
    - Total count of deleted items
    """
    try:
        processor = get_processor()
        
        total_jobs = 0
        total_chunks = 0
        
        for contract_id in request.contract_ids:
            try:
                jobs_deleted, chunks_deleted = processor.delete_job_and_chunks(contract_id)
                total_jobs += jobs_deleted
                total_chunks += chunks_deleted
            except ValueError:
                logger.warning(f"Contract not found: {contract_id}")
                continue
        
        return DeleteResponse(
            status="success",
            deleted_count=total_jobs,
            chunks_deleted=total_chunks,
            message=f"Deleted {total_jobs} jobs and {total_chunks} chunks"
        )
    
    except Exception as e:
        logger.error(f"Error bulk deleting jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Bulk deletion failed")


# ===================== STATISTICS ENDPOINTS =====================

@router.get(
    "/stats/overview",
    response_model=JobStatisticsResponse,
    summary="Get job data statistics",
    responses={
        200: {"description": "Statistics retrieved"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_statistics():
    """
    Get statistics about stored job data and chunks
    
    **Returns:**
    - Total job count
    - Chunk statistics
    - Breakdown by industry
    - Breakdown by project status
    """
    try:
        processor = get_processor()
        stats = processor.get_statistics()
        
        return JobStatisticsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
