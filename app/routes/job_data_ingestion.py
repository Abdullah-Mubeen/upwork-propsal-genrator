"""
Job Data Ingestion Routes

API endpoints for:
- Upload job training data
- Retrieve job data
- Delete job data
- Get chunks
- Statistics
- OCR text extraction

Authentication:
- All endpoints require API key via X-API-Key header
- Use verify_api_key dependency for protected routes
"""
import logging
import base64
from fastapi import APIRouter, HTTPException, Query, status, UploadFile, File, Depends
from typing import List, Optional
import io
from datetime import datetime
from PIL import Image

from app.middleware.auth import verify_api_key
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
from app.utils.advanced_chunker import AdvancedChunkProcessor
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
        chunker = AdvancedChunkProcessor()
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
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def upload_job_data(
    job_data: JobDataUploadRequest,
    api_key: str = Depends(verify_api_key)
):
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
        
        # Convert HttpUrl to string for MongoDB compatibility
        if "client_feedback_url" in job_dict and job_dict["client_feedback_url"] is not None:
            job_dict["client_feedback_url"] = str(job_dict["client_feedback_url"])
        
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
            platform=stored_job.get("platform"),
            project_status=stored_job.get("project_status", "completed"),
            urgent_adhoc=stored_job.get("urgent_adhoc", False),
            start_date=stored_job.get("start_date"),
            end_date=stored_job.get("end_date"),
            portfolio_url=stored_job.get("portfolio_url"),
            is_portfolio_entry=stored_job.get("is_portfolio_entry", False),
            created_at=stored_job.get("created_at", datetime.utcnow().isoformat()),
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
    "/list",
    response_model=ListResponse,
    summary="List all job data",
    responses={
        200: {"description": "Jobs retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def list_job_data(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    status: Optional[str] = Query(None, description="Filter by project status"),
    company: Optional[str] = Query(None, description="Search by company name"),
    api_key: str = Depends(verify_api_key)
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
                platform=job.get("platform"),
                project_status=job.get("project_status", "completed"),
                urgent_adhoc=job.get("urgent_adhoc", False),
                start_date=job.get("start_date"),
                end_date=job.get("end_date"),
                portfolio_url=job.get("portfolio_url"),
                is_portfolio_entry=job.get("is_portfolio_entry", False),
                created_at=job.get("created_at", datetime.utcnow().isoformat()),
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


@router.get(
    "/{contract_id}",
    response_model=JobDataDetailResponse,
    summary="Get job data by contract ID",
    responses={
        200: {"description": "Job data retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_job_data(
    contract_id: str,
    api_key: str = Depends(verify_api_key)
):
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
            project_status=job_data.get("project_status", "completed"),
            start_date=job_data.get("start_date"),
            end_date=job_data.get("end_date"),
            portfolio_url=job_data.get("portfolio_url"),
            portfolio_urls=job_data.get("portfolio_urls"),
            temporary_link=job_data.get("temporary_link"),
            client_feedback_url=job_data.get("client_feedback_url"),
            client_feedback_text=job_data.get("client_feedback_text"),
            task_type=job_data.get("task_type"),
            platform=job_data.get("platform"),
            urgent_adhoc=job_data.get("urgent_adhoc", False),
            is_portfolio_entry=job_data.get("is_portfolio_entry", False),
            deliverables=job_data.get("deliverables", []),
            outcomes=job_data.get("outcomes"),
            created_at=job_data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=job_data.get("updated_at"),
            chunks_count=job_data.get("chunks_count"),
            embedded_chunks_count=job_data.get("embedded_chunks_count")
        )
    
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job not found: {contract_id}")
    except Exception as e:
        logger.error(f"Error retrieving job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job data")


# ===================== CHUNKS ENDPOINTS =====================

@router.get(
    "/{contract_id}/chunks",
    response_model=ListResponse,
    summary="Get chunks for a contract",
    responses={
        200: {"description": "Chunks retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Contract not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_job_chunks(
    contract_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type"),
    api_key: str = Depends(verify_api_key)
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
                project_status=chunk.get("project_status", "completed"),
                embedding_status=chunk["embedding_status"],
                created_at=chunk.get("created_at", datetime.utcnow().isoformat())
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


# ===================== UPDATE ENDPOINTS =====================

@router.put(
    "/update/{contract_id}",
    response_model=UploadResponse,
    summary="Update job data by contract ID",
    responses={
        200: {"description": "Job updated successfully"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def update_job_data(
    contract_id: str,
    update_data: UpdateJobDataRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Update existing job data entry.
    
    **Path Parameters:**
    - `contract_id`: Contract ID to update
    
    **Updatable Fields:**
    - company_name, job_title, job_description
    - your_proposal_text, skills_required, industry
    - platform, task_type, project_status
    - portfolio_urls, client_feedback_text/url
    - And all other job metadata
    
    **Note:** Chunks and embeddings will be regenerated if content changes.
    """
    try:
        processor = get_processor()
        db = get_db()
        
        # Check if job exists
        existing = db.get_training_data(contract_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Job not found: {contract_id}")
        
        # Build update dict (only non-None values)
        update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
        
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Convert HttpUrl to string if present
        if "client_feedback_url" in update_dict and update_dict["client_feedback_url"] is not None:
            update_dict["client_feedback_url"] = str(update_dict["client_feedback_url"])
        
        # Add updated timestamp
        update_dict["updated_at"] = datetime.utcnow().isoformat()
        
        # Update in database
        db.db["training_data"].update_one(
            {"contract_id": contract_id},
            {"$set": update_dict}
        )
        
        # Check if content changed - if so, regenerate chunks/embeddings
        content_fields = ["job_description", "your_proposal_text", "skills_required", "deliverables", "outcomes", "client_feedback_text"]
        content_changed = any(f in update_dict for f in content_fields)
        
        if content_changed:
            logger.info(f"Content changed for {contract_id}, regenerating chunks...")
            
            # Delete old vectors from Pinecone FIRST
            pinecone_service = get_pinecone_service()
            if pinecone_service:
                try:
                    vectors_deleted = pinecone_service.delete_by_contract(contract_id)
                    logger.info(f"Deleted {vectors_deleted} old vectors from Pinecone for {contract_id}")
                except Exception as e:
                    logger.warning(f"Error deleting old Pinecone vectors: {str(e)}")
            
            # Delete old chunks and embeddings from MongoDB
            db.db["chunks"].delete_many({"contract_id": contract_id})
            db.db["embeddings"].delete_many({"contract_id": contract_id})
            
            # Regenerate chunks and embeddings (without re-storing the job)
            chunks_count, chunk_ids = processor.process_and_chunk_job(contract_id)
            embeddings_count, failed = processor.process_and_embed_chunks(contract_id)
            
            # Save to Pinecone
            if pinecone_service:
                try:
                    pinecone_count = processor.save_embeddings_to_pinecone(contract_id)
                    logger.info(f"Saved {pinecone_count} vectors to Pinecone for {contract_id}")
                except Exception as e:
                    logger.warning(f"Error saving to Pinecone: {str(e)}")
        
        # Get updated job for response
        stored_job = processor.get_job_with_chunks(contract_id)
        
        response_data = JobDataResponse(
            db_id=str(stored_job["_id"]),
            contract_id=contract_id,
            company_name=stored_job["company_name"],
            job_title=stored_job["job_title"],
            industry=stored_job.get("industry"),
            skills_required=stored_job.get("skills_required"),
            task_type=stored_job.get("task_type"),
            platform=stored_job.get("platform"),
            project_status=stored_job.get("project_status", "completed"),
            urgent_adhoc=stored_job.get("urgent_adhoc", False),
            start_date=stored_job.get("start_date"),
            end_date=stored_job.get("end_date"),
            portfolio_url=stored_job.get("portfolio_url"),
            is_portfolio_entry=stored_job.get("is_portfolio_entry", False),
            created_at=stored_job.get("created_at"),
            updated_at=stored_job.get("updated_at")
        )
        
        logger.info(f"✅ Updated job: {contract_id}")
        
        return UploadResponse(
            status="success",
            db_id=str(stored_job["_id"]),
            contract_id=contract_id,
            message=f"Job updated successfully" + (" (chunks regenerated)" if content_changed else ""),
            data=response_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update job data")


# ===================== DELETE ENDPOINTS =====================

@router.delete(
    "/delete/{contract_id}",
    response_model=DeleteResponse,
    summary="Delete job data by contract ID",
    responses={
        200: {"description": "Job deleted"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def delete_job_data(
    contract_id: str,
    api_key: str = Depends(verify_api_key)
):
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
    "/bulk-delete",
    response_model=DeleteResponse,
    summary="Bulk delete jobs",
    responses={
        200: {"description": "Jobs deleted"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def bulk_delete_jobs(
    request: DeleteJobsRequest,
    api_key: str = Depends(verify_api_key)
):
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


# ===================== OCR ENDPOINTS =====================

@router.post(
    "/extract-ocr",
    summary="Extract text from image using OCR (GPT-4 Vision)",
    responses={
        200: {"description": "Text extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid image or no image provided"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "OCR processing failed"}
    }
)
async def extract_text_from_feedback_image(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Extract text from feedback image using GPT-4 Vision OCR
    
    This endpoint accepts an image file and uses GPT-4 Vision to extract
    all text content. Perfect for processing client feedback screenshots,
    reviews, or handwritten notes.
    
    **Request:**
    - Multipart form with 'file' field containing image
    
    **Supported formats:** PNG, JPG, JPEG, GIF, WebP
    
    **Returns:**
    - extracted_text: The text content found in the image
    - success: Whether extraction was successful
    - message: Status message
    """
    try:
        # Validate file type
        allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPG, GIF, WebP"
            )
        
        # Validate file size (max 5MB)
        file_content = await file.read()
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 5MB limit"
            )
        
        # Validate image format
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        logger.info(f"🔍 Starting OCR text extraction for file: {file.filename}")
        
        # Get OpenAI service
        processor = get_processor()
        openai_service = processor.openai_service
        
        # Convert image to base64 for Vision API
        import base64
        image_data = base64.standard_b64encode(file_content).decode("utf-8")
        
        # Determine media type
        media_type = file.content_type if file.content_type in allowed_types else "image/jpeg"
        
        # Create data URL
        data_url = f"data:{media_type};base64,{image_data}"
        
        # Extract text using Vision API via base64 data URL
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": "high"
            }
        }
        
        response = openai_service.client.chat.completions.create(
            model=openai_service.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {
                            "type": "text",
                            "text": """Please extract ALL text from this image. 

Extract:
1. All written text
2. All typed text
3. All labels and headers
4. All numbers and dates
5. Any feedback or review content

Format the output clearly, maintaining the structure as much as possible. 
If this is a review/feedback screenshot, extract the complete feedback text."""
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        extracted_text = response.choices[0].message.content
        
        logger.info(f"✅ OCR extraction completed for {file.filename} - Extracted {len(extracted_text)} characters")
        
        return {
            "success": True,
            "message": "Text extracted successfully from image",
            "extracted_text": extracted_text,
            "filename": file.filename,
            "character_count": len(extracted_text)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during OCR extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {str(e)}"
        )


# ===================== STATISTICS ENDPOINTS =====================

@router.get(
    "/stats/overview",
    response_model=JobStatisticsResponse,
    summary="Get job data statistics",
    responses={
        200: {"description": "Statistics retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_statistics(
    api_key: str = Depends(verify_api_key)
):
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


@router.get(
    "/stats/filter-options",
    summary="Get unique filter options from database",
    responses={
        200: {"description": "Filter options retrieved"},
        401: {"description": "API key required"},
        403: {"description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def get_filter_options(
    api_key: str = Depends(verify_api_key)
):
    """
    Get unique task types and platforms from the database for filter dropdowns.
    
    **Returns:**
    - task_types: List of unique task types from database
    - platforms: List of unique platforms from database
    """
    try:
        db = get_db()
        
        # Get all jobs
        all_jobs = list(db.db["training_data"].find({}, {"task_type": 1, "platform": 1, "_id": 0}))
        
        # Extract unique task types (non-null)
        task_types = sorted(list(set(
            job.get("task_type") 
            for job in all_jobs 
            if job.get("task_type")
        )))
        
        # Extract unique platforms (non-null)
        platforms = sorted(list(set(
            job.get("platform") 
            for job in all_jobs 
            if job.get("platform")
        )))
        
        return {
            "task_types": task_types,
            "platforms": platforms
        }
    
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve filter options")


