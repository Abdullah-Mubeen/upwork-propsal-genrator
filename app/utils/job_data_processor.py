"""
Job Data Processor

Business logic for:
- Processing job data
- Creating chunks
- Managing embeddings
- Handling feedback
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from app.db import DatabaseManager
from app.utils.data_chunker import DataChunker
from app.utils.openai_service import OpenAIService
from app.utils.feedback_processor import FeedbackProcessor
from app.utils.pinecone_service import PineconeService
from app.config import settings

logger = logging.getLogger(__name__)


class JobDataProcessor:
    """
    Process job data and prepare for embedding and retrieval
    
    Handles:
    - Data validation and enrichment
    - Smart chunking
    - Embedding generation
    - Metadata extraction
    - Pinecone vector storage
    """
    
    def __init__(
        self,
        db: DatabaseManager,
        openai_service: OpenAIService,
        chunker: DataChunker,
        feedback_processor: FeedbackProcessor,
        pinecone_service: Optional[PineconeService] = None
    ):
        """
        Initialize processor with services
        
        Args:
            db: Database manager instance
            openai_service: OpenAI service for embeddings
            chunker: Data chunker instance
            feedback_processor: Feedback processor instance
            pinecone_service: Optional Pinecone service for vector storage
        """
        self.db = db
        self.openai_service = openai_service
        self.chunker = chunker
        self.feedback_processor = feedback_processor
        self.pinecone_service = pinecone_service
        self.openai_service = openai_service
        self.chunker = chunker
        self.feedback_processor = feedback_processor
    
    def process_and_store_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and store job data in database
        
        Steps:
        1. Validate and enrich job data
        2. Store in training_data collection
        3. Return with contract_id
        
        Args:
            job_data: Job data from request
            
        Returns:
            Dictionary with db_id and contract_id
            
        Raises:
            ValueError: If data is invalid
            Exception: If database operation fails
        """
        try:
            logger.info(f"Processing job data for {job_data.get('company_name')}")
            
            # Enrich data with defaults
            job_data.setdefault("industry", "general")
            job_data.setdefault("project_status", "completed")
            job_data.setdefault("urgent_adhoc", False)
            job_data.setdefault("has_feedback", False)
            
            # Store in database
            result = self.db.insert_training_data(job_data)
            
            logger.info(f"✓ Job data stored with contract_id: {result['contract_id']}")
            return result
        
        except Exception as e:
            logger.error(f"✗ Error processing job data: {str(e)}")
            raise
    
    def process_and_chunk_job(
        self,
        contract_id: str,
        force_rechunk: bool = False
    ) -> Tuple[int, List[str]]:
        """
        Retrieve job data and create chunks
        
        Steps:
        1. Get job data from database
        2. Check for existing chunks
        3. Create smart chunks if needed
        4. Store chunks in database
        5. Return chunk statistics
        
        Args:
            contract_id: Contract ID to chunk
            force_rechunk: Force re-chunking even if chunks exist
            
        Returns:
            Tuple of (chunks_created, chunk_ids)
            
        Raises:
            ValueError: If contract not found
            Exception: If chunking fails
        """
        try:
            logger.info(f"Chunking job data for contract: {contract_id}")
            
            # Get job data
            job_data = self.db.get_training_data(contract_id)
            if not job_data:
                raise ValueError(f"Contract not found: {contract_id}")
            
            # Check for existing chunks
            existing_chunks = self.db.get_chunks_by_contract(contract_id)
            if existing_chunks and not force_rechunk:
                logger.info(f"Chunks already exist for {contract_id}, skipping")
                return len(existing_chunks), []
            
            # Create chunks
            chunks = self.chunker.chunk_training_data(job_data)
            if not chunks:
                logger.warning(f"No chunks created for {contract_id}")
                return 0, []
            
            logger.info(f"Created {len(chunks)} chunks for {contract_id}")
            
            # Store chunks
            chunks_count, chunk_ids = self.db.insert_chunks(chunks, contract_id)
            
            logger.info(f"✓ Stored {chunks_count} chunks with {len(chunk_ids)} chunk IDs")
            return chunks_count, chunk_ids
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"✗ Error chunking job data: {str(e)}")
            raise
    
    def process_and_embed_chunks(
        self,
        contract_id: str,
        force_re_embed: bool = False
    ) -> Tuple[int, int]:
        """
        Generate embeddings for chunks
        
        Steps:
        1. Get pending chunks
        2. Extract text from chunks
        3. Generate embeddings in batches
        4. Save embeddings to database
        5. Update chunk status
        
        Args:
            contract_id: Contract ID to embed
            force_re_embed: Force re-embedding
            
        Returns:
            Tuple of (embedded_count, failed_count)
            
        Raises:
            Exception: If embedding fails
        """
        try:
            logger.info(f"Embedding chunks for contract: {contract_id}")
            
            # Get chunks pending embedding
            if force_re_embed:
                chunks = self.db.get_chunks_by_contract(contract_id)
            else:
                chunks = self.db.get_chunks_pending_embedding(limit=1000)
                chunks = [c for c in chunks if c["contract_id"] == contract_id]
            
            if not chunks:
                logger.info(f"No chunks to embed for {contract_id}")
                return 0, 0
            
            logger.info(f"Embedding {len(chunks)} chunks...")
            
            # Extract texts - support both old and new chunk formats
            # Keep track of which chunks have text
            chunks_with_text = []
            texts = []
            
            for chunk in chunks:
                text = chunk.get("content") or chunk.get("text", "")
                if text and text.strip():
                    chunks_with_text.append(chunk)
                    texts.append(text)
            
            if not texts:
                logger.warning(f"No valid chunk texts found for embedding for {contract_id}")
                return 0, 0
            
            logger.info(f"Found {len(texts)} valid texts to embed out of {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.openai_service.get_embeddings_batch(
                texts=texts,
                dimensions=settings.PINECONE_DIMENSION,
                batch_size=settings.EMBEDDING_BATCH_SIZE
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Prepare embedding records
            embedding_records = []
            successful_count = 0
            failed_count = 0
            
            for chunk, embedding in zip(chunks_with_text, embeddings):
                if embedding is None:
                    failed_count += 1
                    continue
                
                embedding_records.append({
                    "chunk_id": chunk["chunk_id"],
                    "contract_id": chunk["contract_id"],
                    "embedding": embedding,
                    "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                    "pinecone_vector_id": f"{chunk['contract_id']}_{chunk['chunk_id']}"
                })
                
                # Update chunk status
                self.db.update_chunk_embedding_status(
                    chunk["chunk_id"],
                    "completed",
                    f"{chunk['contract_id']}_{chunk['chunk_id']}"
                )
                
                successful_count += 1
            
            # Batch insert embeddings
            if embedding_records:
                self.db.insert_embeddings_batch(embedding_records)
                logger.info(f"✓ Stored {successful_count} embeddings")
            
            if failed_count > 0:
                logger.warning(f"Failed to embed {failed_count} chunks")
            
            return successful_count, failed_count
        
        except Exception as e:
            logger.error(f"✗ Error embedding chunks: {str(e)}")
            raise
    
    def save_feedback_to_collection(
        self,
        contract_id: str,
        feedback_text: str,
        feedback_type: str = "text",
        sentiment: str = "neutral"
    ) -> str:
        """
        Save feedback to feedback_data collection
        
        Args:
            contract_id: Contract ID for reference
            feedback_text: Feedback content
            feedback_type: Type of feedback (text, image, etc)
            sentiment: Sentiment analysis result (positive, negative, neutral)
            
        Returns:
            Feedback document ID
            
        Raises:
            Exception: If save fails
        """
        try:
            feedback_record = {
                "contract_id": contract_id,
                "feedback_text": feedback_text,
                "feedback_type": feedback_type,
                "sentiment": sentiment,
                "created_at": datetime.utcnow(),
                "length": len(feedback_text) if feedback_text else 0
            }
            
            feedback_id = self.db.insert_feedback(feedback_record)
            logger.info(f"✓ Feedback saved to collection for {contract_id}")
            return feedback_id
        
        except Exception as e:
            logger.error(f"✗ Error saving feedback to collection: {str(e)}")
            raise
    
    def process_feedback_text(
        self,
        contract_id: str,
        feedback_text: str,
        feedback_type: str = "text",
        image_path: Optional[str] = None,
        is_url: bool = False
    ) -> str:
        """
        Process feedback (text or image)
        
        Steps:
        1. If image: Extract text using OCR
        2. Update job data with feedback
        3. Mark job as having feedback
        4. Save to feedback_data collection
        
        Args:
            contract_id: Contract to update
            feedback_text: Direct text or image path
            feedback_type: "text" or "image"
            image_path: Path/URL to image if type is "image"
            is_url: If True, image_path is URL
            
        Returns:
            Extracted feedback text
            
        Raises:
            ValueError: If contract not found
            Exception: If processing fails
        """
        try:
            logger.info(f"Processing {feedback_type} feedback for {contract_id}")
            
            # Get job data
            job_data = self.db.get_training_data(contract_id)
            if not job_data:
                raise ValueError(f"Contract not found: {contract_id}")
            
            # Process feedback
            feedback_result = self.feedback_processor.process_feedback(
                feedback_content=feedback_text,
                feedback_type=feedback_type,
                image_path=image_path,
                is_url=is_url
            )
            
            if not feedback_result["success"]:
                logger.warning(f"Feedback processing returned error: {feedback_result.get('error')}")
            
            extracted_text = feedback_result["extracted_text"]
            
            # Update job data
            update_data = {
                "client_feedback": extracted_text,
                "has_feedback": True,
                "feedback_processed_at": datetime.utcnow(),
                "feedback_type": feedback_type
            }
            
            self.db.update_training_data(contract_id, update_data)
            
            # Save to feedback_data collection
            self.save_feedback_to_collection(
                contract_id,
                extracted_text,
                feedback_type=feedback_type,
                sentiment=feedback_result.get("sentiment", "neutral")
            )
            
            logger.info(f"✓ Feedback processed and stored for {contract_id}")
            return extracted_text
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"✗ Error processing feedback: {str(e)}")
            raise
    
    def get_job_with_chunks(self, contract_id: str) -> Dict[str, Any]:
        """
        Get job data with chunk statistics
        
        Args:
            contract_id: Contract ID to retrieve
            
        Returns:
            Job data with chunk information
            
        Raises:
            ValueError: If contract not found
        """
        try:
            job_data = self.db.get_training_data(contract_id)
            if not job_data:
                raise ValueError(f"Contract not found: {contract_id}")
            
            # Get chunk statistics
            chunks = self.db.get_chunks_by_contract(contract_id)
            embedded_chunks = [c for c in chunks if c["embedding_status"] == "completed"]
            
            job_data["chunks_count"] = len(chunks)
            job_data["embedded_chunks_count"] = len(embedded_chunks)
            
            return job_data
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving job data: {str(e)}")
            raise
    
    def delete_job_and_chunks(self, contract_id: str) -> Tuple[int, int]:
        """
        Delete job data and associated chunks
        
        Args:
            contract_id: Contract ID to delete
            
        Returns:
            Tuple of (jobs_deleted, chunks_deleted)
            
        Raises:
            ValueError: If contract not found
        """
        try:
            logger.info(f"Deleting job and chunks for {contract_id}")
            
            # Check if job exists
            job_data = self.db.get_training_data(contract_id)
            if not job_data:
                raise ValueError(f"Contract not found: {contract_id}")
            
            # Delete associated chunks
            chunks = self.db.get_chunks_by_contract(contract_id)
            chunks_count = len(chunks)
            
            for chunk in chunks:
                try:
                    self.db.db["chunks"].delete_one({"chunk_id": chunk["chunk_id"]})
                    self.db.db["embeddings"].delete_many({"chunk_id": chunk["chunk_id"]})
                except Exception as e:
                    logger.warning(f"Error deleting chunk {chunk['chunk_id']}: {str(e)}")
            
            # Delete associated proposals
            try:
                self.db.db["proposals"].delete_many({"contract_id": contract_id})
                logger.info(f"Deleted proposals for {contract_id}")
            except Exception as e:
                logger.warning(f"Error deleting proposals for {contract_id}: {str(e)}")
            
            # Delete associated feedback_data
            try:
                self.db.db["feedback_data"].delete_many({"contract_id": contract_id})
                logger.info(f"Deleted feedback_data for {contract_id}")
            except Exception as e:
                logger.warning(f"Error deleting feedback_data for {contract_id}: {str(e)}")
            
            # Delete from Pinecone vector database
            if self.pinecone_service:
                try:
                    vectors_deleted = self.pinecone_service.delete_by_contract(contract_id)
                    logger.info(f"Deleted {vectors_deleted} vectors from Pinecone for {contract_id}")
                except Exception as e:
                    logger.warning(f"Error deleting vectors from Pinecone for {contract_id}: {str(e)}")
            
            # Delete job data
            result = self.db.db["training_data"].delete_one({"contract_id": contract_id})
            jobs_deleted = result.deleted_count
            
            logger.info(f"✓ Deleted job (1), {chunks_count} chunks, and associated Pinecone vectors")
            return jobs_deleted, chunks_count
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"✗ Error deleting job: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored jobs and chunks"""
        try:
            stats = self.db.get_database_statistics()
            industry_stats = self.db.get_industry_statistics()
            
            # Add status breakdown
            status_stats = {}
            for status in ["completed", "ongoing", "pending"]:
                count = self.db.db["training_data"].count_documents({"project_status": status})
                status_stats[status] = count
            
            return {
                "total_jobs": stats["training_data_count"],
                "total_chunks": stats["total_chunks"],
                "chunks_embedded": stats["chunks_embedded"],
                "chunks_pending": stats["chunks_pending"],
                "by_industry": industry_stats,
                "by_status": status_stats
            }
        
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def save_embeddings_to_pinecone(
        self,
        contract_id: str,
        chunk_ids: List[str] = None
    ) -> int:
        """
        Save embeddings from MongoDB to Pinecone vector store
        
        Steps:
        1. Get embeddings from MongoDB
        2. Get corresponding chunks for metadata
        3. Format as vectors with rich metadata
        4. Upsert to Pinecone for semantic search
        
        Args:
            contract_id: Contract ID to save embeddings for
            chunk_ids: Optional list of specific chunk IDs to save (if None, saves all)
            
        Returns:
            Number of vectors saved to Pinecone
            
        Raises:
            Exception: If Pinecone save fails
        """
        if not self.pinecone_service:
            logger.warning(f"Pinecone service not initialized, skipping vector storage for {contract_id}")
            return 0
        
        try:
            logger.info(f"Saving embeddings to Pinecone for {contract_id}")
            
            # Get embeddings
            embeddings = self.db.get_embeddings_by_contract(contract_id)
            if not embeddings:
                logger.warning(f"No embeddings found for {contract_id}")
                return 0
            
            # Get job data and chunks for metadata
            job_data = self.db.get_training_data(contract_id)
            chunks = self.db.get_chunks_by_contract(contract_id)
            chunks_by_id = {c["chunk_id"]: c for c in chunks}
            
            # Filter by chunk_ids if specified
            if chunk_ids:
                embeddings = [e for e in embeddings if e["chunk_id"] in chunk_ids]
            
            if not embeddings:
                logger.warning(f"No matching embeddings found for chunks")
                return 0
            
            # Prepare vectors with rich metadata
            vectors_to_upsert = []
            
            for embedding in embeddings:
                chunk_id = embedding.get("chunk_id")
                chunk = chunks_by_id.get(chunk_id, {})
                embedding_vector = embedding.get("embedding", [])
                
                # Validate embedding
                if not embedding_vector or len(embedding_vector) == 0:
                    logger.warning(f"Skipping embedding for {chunk_id}: empty vector")
                    continue
                
                if len(embedding_vector) != settings.PINECONE_DIMENSION:
                    logger.warning(f"Skipping embedding for {chunk_id}: wrong dimension {len(embedding_vector)} vs {settings.PINECONE_DIMENSION}")
                    continue
                
                # Check for all-zero vector
                if all(v == 0.0 for v in embedding_vector):
                    logger.warning(f"Skipping embedding for {chunk_id}: all-zero vector (invalid embedding)")
                    continue
                
                # Create rich metadata for AI training
                metadata = {
                    "contract_id": contract_id,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "priority": chunk.get("priority", 1.0),
                    "company_name": job_data.get("company_name", ""),
                    "job_title": job_data.get("job_title", ""),
                    "industry": job_data.get("industry", "general"),
                    "skills": chunk.get("skills_required", []),
                    "task_type": job_data.get("task_type", "other"),
                    "urgent": job_data.get("urgent_adhoc", False),
                    "project_status": job_data.get("project_status", "completed"),
                    "has_feedback": job_data.get("has_feedback", False),
                    "portfolio_url": job_data.get("portfolio_url", ""),
                    "created_at": str(job_data.get("created_at", ""))
                }
                
                # Use embedding ID as vector ID
                vector_id = embedding.get("pinecone_vector_id", f"{contract_id}_{chunk_id}")
                
                vectors_to_upsert.append((
                    vector_id,
                    embedding_vector,
                    metadata
                ))
            
            if not vectors_to_upsert:
                logger.warning(f"No valid embeddings to upsert for {contract_id}")
                return 0
            
            # Upsert to Pinecone
            upserted_count = self.pinecone_service.upsert_vectors(vectors_to_upsert)
            logger.info(f"✓ Successfully saved {upserted_count} vectors to Pinecone for {contract_id}")
            
            return upserted_count
        
        except Exception as e:
            logger.error(f"✗ Error saving embeddings to Pinecone: {str(e)}")
            raise
    
    def save_proposal_record(
        self,
        contract_id: str,
        job_data: Dict[str, Any]
    ) -> str:
        """
        Save proposal record to proposals collection
        
        This stores the training proposal for reference and AI learning
        
        Args:
            contract_id: Contract ID reference
            job_data: Job data containing proposal text
            
        Returns:
            Proposal document ID
            
        Raises:
            Exception: If save fails
        """
        try:
            proposal_record = {
                "contract_id": contract_id,
                "company_name": job_data.get("company_name", ""),
                "job_title": job_data.get("job_title", ""),
                "proposal_text": job_data.get("your_proposal_text", ""),
                "industry": job_data.get("industry", "general"),
                "task_type": job_data.get("task_type", "other"),
                "skills_required": job_data.get("skills_required", []),
                "project_status": job_data.get("project_status", "completed"),
                "urgent_adhoc": job_data.get("urgent_adhoc", False),
                "has_feedback": job_data.get("has_feedback", False),
                "portfolio_url": job_data.get("portfolio_url", ""),
                "status": "completed",
                "created_at": datetime.utcnow(),
                "proposal_length": len(job_data.get("your_proposal_text", ""))
            }
            
            proposal_id = self.db.save_proposal(proposal_record)
            logger.info(f"✓ Proposal record saved for {contract_id}")
            return proposal_id
        
        except Exception as e:
            logger.error(f"✗ Error saving proposal record: {str(e)}")
            raise
    
    def process_complete_pipeline(
        self,
        job_data: Dict[str, Any],
        save_to_pinecone: bool = True
    ) -> Dict[str, Any]:
        """
        Complete end-to-end pipeline:
        1. Store job data in MongoDB
        2. Create smart chunks
        3. Generate embeddings
        4. Save to feedback_data collection if feedback provided
        5. Save proposal record
        6. Save embeddings to Pinecone for semantic search
        
        Args:
            job_data: Complete job data
            save_to_pinecone: Whether to save vectors to Pinecone
            
        Returns:
            Dictionary with statistics: contract_id, chunks_created, embeddings_created, pinecone_vectors
            
        Raises:
            Exception: If any step fails
        """
        try:
            logger.info("=" * 60)
            logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
            logger.info("=" * 60)
            
            # Step 1: Store job data
            logger.info("📝 Step 1: Storing job data in MongoDB...")
            store_result = self.process_and_store_job(job_data)
            contract_id = store_result["contract_id"]
            logger.info(f"✓ Job stored with contract_id: {contract_id}")
            
            # Step 2: Create chunks
            logger.info("📚 Step 2: Creating smart chunks...")
            chunks_count, chunk_ids = self.process_and_chunk_job(contract_id)
            logger.info(f"✓ Created {chunks_count} smart chunks")
            
            # Step 3: Generate embeddings
            logger.info("🔢 Step 3: Generating embeddings...")
            embeddings_count, failed_count = self.process_and_embed_chunks(contract_id)
            logger.info(f"✓ Generated {embeddings_count} embeddings ({failed_count} failed)")
            
            # Step 4: Handle feedback
            logger.info("💬 Step 4: Processing feedback...")
            if job_data.get("client_feedback"):
                try:
                    self.save_feedback_to_collection(
                        contract_id,
                        job_data["client_feedback"],
                        feedback_type="text",
                        sentiment="neutral"
                    )
                    logger.info(f"✓ Feedback saved to collection")
                except Exception as e:
                    logger.warning(f"Could not save feedback: {str(e)}")
            else:
                logger.info("⊘ No feedback provided")
            
            # Step 5: Save proposal reference
            logger.info("📄 Step 5: Saving proposal record...")
            try:
                self.save_proposal_record(contract_id, job_data)
                logger.info(f"✓ Proposal record saved")
            except Exception as e:
                logger.warning(f"Could not save proposal record: {str(e)}")
            
            # Step 6: Save to Pinecone
            pinecone_count = 0
            if save_to_pinecone and embeddings_count > 0:
                logger.info("📌 Step 6: Saving vectors to Pinecone...")
                try:
                    pinecone_count = self.save_embeddings_to_pinecone(contract_id, chunk_ids)
                    logger.info(f"✓ Saved {pinecone_count} vectors to Pinecone")
                except Exception as e:
                    logger.warning(f"Could not save to Pinecone: {str(e)}")
            else:
                logger.info("⊘ Pinecone save skipped")
            
            logger.info("=" * 60)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("=" * 60)
            
            return {
                "contract_id": contract_id,
                "db_id": store_result["db_id"],
                "chunks_created": chunks_count,
                "embeddings_created": embeddings_count,
                "embeddings_failed": failed_count,
                "pinecone_vectors": pinecone_count,
                "feedback_saved": bool(job_data.get("client_feedback"))
            }
        
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {str(e)}")
            raise

