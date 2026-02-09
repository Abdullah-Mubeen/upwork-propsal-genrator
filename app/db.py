import logging
from typing import Optional, List, Dict, Any, Tuple
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
from contextlib import contextmanager
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import uuid
import certifi
from app.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages MongoDB connections and data persistence with smart chunking strategy"""
    
    def __init__(self, connection_string: str = None, db_name: str = None):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection URI
            db_name: Database name
        """
        self.connection_string = connection_string or settings.MONGODB_URI
        self.db_name = db_name or settings.MONGODB_DB_NAME
        self.client: Optional[MongoClient] = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection with SSL configuration"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=15000,
                socketTimeoutMS=15000,
                retryWrites=True,
                maxPoolSize=50,
                minPoolSize=10,
                tls=True,
                tlsCAFile=certifi.where()
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            
            logger.info(f"Successfully connected to MongoDB: {self.db_name}")
            self._init_collections()
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def _init_collections(self):
        """Initialize collections and create indexes"""
        try:
            collections = {
                "training_data": "Raw job data for training the AI model",
                "chunks": "Smart chunks created from training data",
                "embeddings": "Embedding metadata and references",
                "proposals": "Generated proposals",
                "feedback_data": "Client feedback analysis",
                "embedding_cache": "Cache of embeddings for quick retrieval",
                "skills": "Unique skills with frequency and usage tracking",
                "skill_embeddings": "Embeddings for skill vectors",
                "sent_proposals": "Sent proposals with outcome tracking (viewed/hired)",
                "user_profile": "User profile data for dashboard"
            }
            
            for collection_name in collections.keys():
                if collection_name not in self.db.list_collection_names():
                    self. db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
            
            # Create indexes for better performance
            self._create_indexes()
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create indexes for collections"""
        try:
            # Training Data Collection Indexes
            self.db["training_data"].create_index([("contract_id", ASCENDING)], unique=True)
            self.db["training_data"].create_index([("company_name", ASCENDING)])
            self.db["training_data"].create_index([("industry", ASCENDING)])
            self. db["training_data"].create_index([("job_title", ASCENDING)])
            self.db["training_data"].create_index([("created_at", DESCENDING)])
            self.db["training_data"].create_index([("skills_required", ASCENDING)])
            self.db["training_data"].create_index([("project_status", ASCENDING)])
            self.db["training_data"].create_index([("urgent_adhoc", ASCENDING)])
            
            # Text index for full-text search - try to drop old index first
            try:
                # Drop old text index if it exists
                self.db["training_data"].drop_index("job_description_text_your_proposal_text_text_client_feedback.text_text")
                logger.info("Dropped old text index with client_feedback.text")
            except Exception as e:
                logger.debug(f"Old index not found or already dropped: {str(e)}")
            
            try:
                # Create new text index with new field name
                self.db["training_data"].create_index([
                    ("job_description", TEXT),
                    ("your_proposal_text", TEXT),
                    ("client_feedback_text", TEXT)
                ])
            except Exception as e:
                logger.warning(f"Could not create new text index: {str(e)}")
            
            # Chunks Collection Indexes - CRITICAL for retrieval
            self.db["chunks"].create_index([("contract_id", ASCENDING)])
            self.db["chunks"].create_index([("chunk_id", ASCENDING)], unique=True)
            self.db["chunks"].create_index([("chunk_type", ASCENDING)])
            self. db["chunks"].create_index([("priority", DESCENDING)])
            self.db["chunks"].create_index([("industry", ASCENDING)])
            self. db["chunks"].create_index([("skills_required", ASCENDING)])
            self.db["chunks"].create_index([("created_at", DESCENDING)])
            self.db["chunks"].create_index([("content", TEXT)])  # Text search on content
            
            # Embeddings Collection Indexes
            self. db["embeddings"].create_index([("chunk_id", ASCENDING)], unique=True)
            self.db["embeddings"].create_index([("contract_id", ASCENDING)])
            self.db["embeddings"].create_index([("embedding_model", ASCENDING)])
            self. db["embeddings"].create_index([("created_at", DESCENDING)])
            self.db["embeddings"].create_index([("pinecone_vector_id", ASCENDING)])
            
            # Proposals Collection Indexes
            self.db["proposals"].create_index([("contract_id", ASCENDING)])
            self.db["proposals"].create_index([("created_at", DESCENDING)])
            self.db["proposals"].create_index([("status", ASCENDING)])
            self.db["proposals"].create_index([("company_name", ASCENDING)])
            
            # Feedback Data Indexes
            self.db["feedback_data"].create_index([("contract_id", ASCENDING)])
            self.db["feedback_data"].create_index([("feedback_type", ASCENDING)])  # image or text
            self.db["feedback_data"].create_index([("sentiment", ASCENDING)])
            self.db["feedback_data"].create_index([("created_at", DESCENDING)])
            
            # Embedding Cache Indexes
            self.db["embedding_cache"]. create_index([("text_hash", ASCENDING)], unique=True)
            self.db["embedding_cache"].create_index([("model", ASCENDING)])
            self.db["embedding_cache"].create_index([("created_at", DESCENDING)])
            
            # Skills Collection Indexes
            self.db["skills"].create_index([("skill_name_lower", ASCENDING)], unique=True)
            self.db["skills"].create_index([("frequency", DESCENDING)])
            self.db["skills"].create_index([("contracts", ASCENDING)])
            self.db["skills"].create_index([("last_used", DESCENDING)])
            
            # Skill Embeddings Collection Indexes
            self.db["skill_embeddings"].create_index([("skill_name_lower", ASCENDING)], unique=True)
            self.db["skill_embeddings"].create_index([("created_at", DESCENDING)])
            
            # Sent Proposals Collection Indexes (Outcome Tracking)
            self.db["sent_proposals"].create_index([("proposal_id", ASCENDING)], unique=True)
            self.db["sent_proposals"].create_index([("outcome", ASCENDING)])
            self.db["sent_proposals"].create_index([("sent_at", DESCENDING)])
            self.db["sent_proposals"].create_index([("job_title", ASCENDING)])
            self.db["sent_proposals"].create_index([("skills_required", ASCENDING)])
            self.db["sent_proposals"].create_index([("industry", ASCENDING)])
            self.db["sent_proposals"].create_index([("source", ASCENDING)])  # ai_generated or manual
            
            # User Profile Collection (singleton)
            self.db["user_profile"].create_index([("user_id", ASCENDING)], unique=True)
            
            logger.info("Successfully created all database indexes")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def close(self):
        """Alias for disconnect - close MongoDB connection"""
        self.disconnect()
    
    def connect(self):
        """Alias for _connect - establish connection"""
        self._connect()
    
    def get_collection(self, collection_name: str):
        """Get a collection"""
        return self.db[collection_name]
    
    @contextmanager
    def session(self):
        """Context manager for database sessions"""
        session = self.client.start_session()
        try:
            yield session
        finally:
            session.end_session()
    
    # ===================== TRAINING DATA OPERATIONS =====================
    
    def insert_training_data(self, job_data: Dict[str, Any]) -> str:
        """
        Insert a new training data record (raw job data)
        Automatically generates contract_id if not provided
        Format: job_<short_unique_id> (e.g., job_a1b2c3d4)
        
        Args:
            job_data: Job information from user
            
        Returns:
            Inserted document ID and contract_id
        """
        try:
            # Generate unique contract_id if not provided
            if "contract_id" not in job_data or not job_data["contract_id"]:
                # Create short unique ID with job_ prefix
                short_id = uuid.uuid4().hex[:8]  # 8 chars hex
                job_data["contract_id"] = f"job_{short_id}"
            
            job_data["created_at"] = datetime.utcnow()
            job_data["updated_at"] = datetime.utcnow()
            
            # Ensure required fields
            job_data. setdefault("industry", "general")
            job_data.setdefault("project_status", "completed")
            job_data.setdefault("urgent_adhoc", False)
            
            result = self.db["training_data"].insert_one(job_data)
            logger.info(f"Inserted training data with contract_id: {job_data['contract_id']}, DB ID: {result.inserted_id}")
            
            return {
                "db_id": str(result.inserted_id),
                "contract_id": job_data["contract_id"]
            }
        except DuplicateKeyError:
            logger.error(f"Training data with contract_id already exists: {job_data. get('contract_id')}")
            raise
        except Exception as e:
            logger.error(f"Error inserting training data: {str(e)}")
            raise
    
    def get_training_data(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get training data by contract_id"""
        try:
            result = self.db["training_data"].find_one({"contract_id": contract_id})
            return result
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return None
    
    def get_all_training_data(self, skip: int = 0, limit: int = 50, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get all training data with pagination and filtering"""
        try:
            filter_query = filters or {}
            results = list(
                self.db["training_data"]
                .find(filter_query)
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return []
    
    def update_training_data(self, contract_id: str, update_data: Dict[str, Any]) -> bool:
        """Update training data"""
        try:
            update_data["updated_at"] = datetime. utcnow()
            
            result = self.db["training_data"].update_one(
                {"contract_id": contract_id},
                {"$set": update_data}
            )
            logger.info(f"Updated training data: {contract_id}, Modified: {result.modified_count}")
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating training data: {str(e)}")
            raise
    
    def get_training_data_count(self) -> int:
        """Get count of training data in database"""
        try:
            count = self.db["training_data"].count_documents({})
            return count
        except Exception as e:
            logger.error(f"Error counting training data: {str(e)}")
            return 0
    
    # ===================== SMART CHUNKS OPERATIONS =====================
    
    def insert_chunks(self, chunks: List[Dict[str, Any]], contract_id: str) -> Tuple[int, List[str]]:
        """
        Insert smart chunks created from training data
        
        Args:
            chunks: List of chunks from DataChunker
            contract_id: Contract ID for reference
            
        Returns:
            Tuple of (total_chunks_inserted, list_of_chunk_ids)
        """
        try:
            chunk_ids = []
            insert_docs = []
            
            for idx, chunk in enumerate(chunks):
                # Support both old ('content', 'type') and new ('text', 'chunk_type') formats
                chunk_text = chunk.get("text") or chunk.get("content", "")
                chunk_type = chunk.get("chunk_type") or chunk.get("type", "text")
                
                if not chunk_text or not chunk_text.strip():
                    logger.warning(f"Skipping empty chunk {idx} for {contract_id}")
                    continue
                
                chunk_id = f"{contract_id}_{chunk_type}_{idx}"
                
                # Extract metadata from chunk (metadata is nested)
                metadata = chunk.get("metadata", {})
                
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "contract_id": contract_id,
                    "content": chunk_text,
                    "chunk_type": chunk_type,
                    "priority": chunk.get("priority", 1.0),
                    "length": chunk.get("length", len(chunk_text)),
                    # CRITICAL: Extract from metadata for proper filtering
                    "task_type": metadata.get("task_type", "").lower(),  # Always lowercase for consistency
                    "industry": metadata.get("industry", "general").lower(),
                    "skills_required": metadata.get("skills", []),  # Note: in metadata it's "skills"
                    "company_name": metadata.get("company_name", ""),
                    "job_title": metadata.get("job_title", ""),
                    "urgency": metadata.get("urgency", "normal"),
                    "task_complexity": metadata.get("task_complexity", "medium"),
                    "is_completed": metadata.get("is_completed", False),
                    "duration_days": metadata.get("duration_days"),
                    "project_status": chunk.get("project_status", ""),
                    # NEW: Deliverables for matching what was actually built
                    "deliverables": chunk.get("deliverables", []),
                    "outcomes": chunk.get("outcomes", ""),
                    "created_at": datetime.utcnow(),
                    "embedding_status": "pending"
                }
                
                insert_docs.append(chunk_doc)
                chunk_ids.append(chunk_id)
            
            if insert_docs:
                result = self.db["chunks"].insert_many(insert_docs, ordered=False)
                logger. info(f"Inserted {len(insert_docs)} chunks for contract_id: {contract_id}")
            
            return len(insert_docs), chunk_ids
        except Exception as e:
            logger.error(f"Error inserting chunks: {str(e)}")
            raise
    
    def get_chunks_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a contract"""
        try:
            results = list(
                self.db["chunks"]
                .find({"contract_id": contract_id})
                . sort("priority", DESCENDING)
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def update_chunk_embedding_status(self, chunk_id: str, status: str, pinecone_vector_id: str = None) -> bool:
        """Update chunk's embedding status"""
        try:
            update_doc = {
                "embedding_status": status,
                "embedding_updated_at": datetime.utcnow()
            }
            
            if pinecone_vector_id:
                update_doc["pinecone_vector_id"] = pinecone_vector_id
            
            result = self. db["chunks"].update_one(
                {"chunk_id": chunk_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating chunk embedding status: {str(e)}")
            return False
    
    def get_chunks_pending_embedding(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chunks that still need to be embedded"""
        try:
            results = list(
                self.db["chunks"]
                . find({"embedding_status": "pending"})
                .sort("priority", DESCENDING)
                .limit(limit)
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving pending chunks: {str(e)}")
            return []
    
    # ===================== EMBEDDINGS OPERATIONS =====================
    
    def insert_embedding(self, embedding_data: Dict[str, Any]) -> str:
        """
        Insert embedding record
        
        Args:
            embedding_data: Embedding information
            
        Returns:
            Inserted document ID
        """
        try:
            embedding_data["created_at"] = datetime.utcnow()
            result = self.db["embeddings"]. insert_one(embedding_data)
            logger.info(f"Inserted embedding for chunk: {embedding_data. get('chunk_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting embedding: {str(e)}")
            raise
    
    def insert_embeddings_batch(self, embeddings_data: List[Dict[str, Any]]) -> int:
        """
        Batch insert embeddings
        
        Args:
            embeddings_data: List of embedding records
            
        Returns:
            Number of embeddings inserted
        """
        try:
            for emb in embeddings_data:
                emb["created_at"] = datetime.utcnow()
            
            result = self.db["embeddings"]. insert_many(embeddings_data, ordered=False)
            logger. info(f"Inserted {len(result.inserted_ids)} embeddings in batch")
            return len(result.inserted_ids)
        except Exception as e:
            logger. error(f"Error inserting embeddings batch: {str(e)}")
            raise
    
    def get_embedding_by_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding record for a chunk"""
        try:
            result = self.db["embeddings"]. find_one({"chunk_id": chunk_id})
            return result
        except Exception as e:
            logger.error(f"Error retrieving embedding: {str(e)}")
            return None
    
    def get_embeddings_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all embeddings for a contract"""
        try:
            results = list(self.db["embeddings"].find({"contract_id": contract_id}))
            return results
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {str(e)}")
            return []
    
    # ===================== FEEDBACK DATA OPERATIONS =====================
    
    def insert_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Insert client feedback (from image OCR or text)
        
        Args:
            feedback_data: Feedback information with extracted text
            
        Returns:
            Inserted document ID
        """
        try:
            feedback_data["created_at"] = datetime. utcnow()
            result = self.db["feedback_data"].insert_one(feedback_data)
            logger.info(f"Inserted feedback for contract: {feedback_data.get('contract_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting feedback: {str(e)}")
            raise
    
    def get_feedback_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a contract"""
        try:
            results = list(
                self.db["feedback_data"]
                .find({"contract_id": contract_id})
                .sort("created_at", DESCENDING)
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving feedback: {str(e)}")
            return []
    
    # ===================== PROPOSALS OPERATIONS =====================
    
    def save_proposal(self, proposal_data: Dict[str, Any]) -> str:
        """
        Save generated proposal
        
        Args:
            proposal_data: Proposal information
            
        Returns:
            Inserted document ID
        """
        try:
            proposal_data["created_at"] = datetime.utcnow()
            result = self.db["proposals"].insert_one(proposal_data)
            logger.info(f"Saved proposal for contract: {proposal_data.get('contract_id')}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving proposal: {str(e)}")
            raise
    
    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get a proposal by ID"""
        try:
            result = self.db["proposals"].find_one({"_id": ObjectId(proposal_id)})
            return result
        except Exception as e:
            logger.error(f"Error retrieving proposal: {str(e)}")
            return None
    
    def get_proposals_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all proposals for a contract"""
        try:
            results = list(
                self.db["proposals"]
                .find({"contract_id": contract_id})
                . sort("created_at", DESCENDING)
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving proposals: {str(e)}")
            return []
    
    # ===================== EMBEDDING CACHE OPERATIONS =====================
    
    def cache_embedding(self, text: str, embedding: List[float], model: str) -> bool:
        """
        Cache embedding for quick retrieval
        
        Args:
            text: Original text
            embedding: Embedding vector
            model: Embedding model used
            
        Returns:
            True if cached successfully
        """
        try:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            self.db["embedding_cache"].update_one(
                {"text_hash": text_hash, "model": model},
                {
                    "$set": {
                        "text_hash": text_hash,
                        "text": text[:500],  # Store first 500 chars
                        "embedding": embedding,
                        "model": model,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error caching embedding: {str(e)}")
            return False
    
    def get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        try:
            import hashlib
            text_hash = hashlib.md5(text. encode()).hexdigest()
            
            result = self.db["embedding_cache"].find_one({
                "text_hash": text_hash,
                "model": model
            })
            
            if result:
                logger.debug(f"Found cached embedding for text")
                return result. get("embedding")
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached embedding: {str(e)}")
            return None
    
    # ===================== SKILLS OPERATIONS =====================
    
    def save_skills(self, contract_id: str, skills: List[str]) -> List[str]:
        """
        Save skills from a job to skills collection
        
        Args:
            contract_id: Contract ID for reference
            skills: List of skill names
            
        Returns:
            List of skill IDs saved
        """
        try:
            skill_ids = []
            
            for skill in skills:
                if not skill or not isinstance(skill, str):
                    continue
                
                skill_lower = skill.strip().lower()
                
                # Update or insert skill
                skill_doc = {
                    "skill_name": skill,
                    "skill_name_lower": skill_lower,
                    "contracts": [contract_id],
                    "frequency": 1,
                    "last_used": datetime.utcnow(),
                    "created_at": datetime.utcnow()
                }
                
                # Check if skill already exists
                existing = self.db["skills"].find_one({"skill_name_lower": skill_lower})
                
                if existing:
                    # Update frequency and add contract if not already there
                    self.db["skills"].update_one(
                        {"_id": existing["_id"]},
                        {
                            "$inc": {"frequency": 1},
                            "$addToSet": {"contracts": contract_id},
                            "$set": {"last_used": datetime.utcnow()}
                        }
                    )
                    skill_ids.append(str(existing["_id"]))
                else:
                    # Insert new skill
                    result = self.db["skills"].insert_one(skill_doc)
                    skill_ids.append(str(result.inserted_id))
            
            if skill_ids:
                logger.info(f"✓ Saved {len(skill_ids)} skills for contract {contract_id}")
            
            return skill_ids
        
        except Exception as e:
            logger.error(f"Error saving skills: {str(e)}")
            return []
    
    def get_all_skills(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all unique skills with frequency and usage info
        
        Args:
            limit: Maximum number of skills to return
            
        Returns:
            List of skill documents sorted by frequency
        """
        try:
            skills = list(
                self.db["skills"]
                .find()
                .sort("frequency", DESCENDING)
                .limit(limit)
            )
            return skills
        except Exception as e:
            logger.error(f"Error retrieving skills: {str(e)}")
            return []
    
    def get_skills_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get skills associated with a specific contract"""
        try:
            skills = list(
                self.db["skills"]
                .find({"contracts": contract_id})
                .sort("frequency", DESCENDING)
            )
            return skills
        except Exception as e:
            logger.error(f"Error retrieving skills for contract: {str(e)}")
            return []
    
    def search_skills(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for skills by name (case-insensitive)
        
        Args:
            search_term: Skill name to search for
            
        Returns:
            List of matching skills
        """
        try:
            search_lower = search_term.strip().lower()
            skills = list(
                self.db["skills"]
                .find({"skill_name_lower": {"$regex": search_lower, "$options": "i"}})
                .sort("frequency", DESCENDING)
            )
            return skills
        except Exception as e:
            logger.error(f"Error searching skills: {str(e)}")
            return []
    
    def save_skill_embedding(self, skill_name: str, embedding: List[float]) -> str:
        """
        Save embedding for a skill
        
        Args:
            skill_name: Name of the skill
            embedding: Embedding vector
            
        Returns:
            Embedding document ID
        """
        try:
            skill_embedding = {
                "skill_name": skill_name,
                "skill_name_lower": skill_name.strip().lower(),
                "embedding": embedding,
                "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
                "created_at": datetime.utcnow()
            }
            
            result = self.db["skill_embeddings"].insert_one(skill_embedding)
            logger.info(f"✓ Saved embedding for skill: {skill_name}")
            return str(result.inserted_id)
        
        except Exception as e:
            logger.error(f"Error saving skill embedding: {str(e)}")
            return ""
    
    def get_skill_embedding(self, skill_name: str) -> Optional[List[float]]:
        """Get embedding for a skill"""
        try:
            skill_lower = skill_name.strip().lower()
            doc = self.db["skill_embeddings"].find_one({"skill_name_lower": skill_lower})
            
            if doc:
                return doc.get("embedding")
            return None
        except Exception as e:
            logger.error(f"Error retrieving skill embedding: {str(e)}")
            return None

    # ===================== ANALYTICS & STATISTICS =====================
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            stats = {
                "training_data_count": self.db["training_data"].count_documents({}),
                "total_chunks": self.db["chunks"].count_documents({}),
                "chunks_embedded": self.db["chunks"].count_documents({"embedding_status": "completed"}),
                "chunks_pending": self.db["chunks"].count_documents({"embedding_status": "pending"}),
                "total_embeddings": self.db["embeddings"].count_documents({}),
                "total_proposals": self.db["proposals"].count_documents({}),
                "total_feedback": self.db["feedback_data"].count_documents({}),
                "cached_embeddings": self.db["embedding_cache"].count_documents({})
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")
            return {}
    
    def get_industry_statistics(self) -> Dict[str, int]:
        """Get statistics by industry"""
        try:
            pipeline = [
                {"$group": {"_id": "$industry", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            results = list(self.db["training_data"].aggregate(pipeline))
            return {item["_id"]: item["count"] for item in results}
        except Exception as e:
            logger.error(f"Error getting industry statistics: {str(e)}")
            return {}

    # ===================== SENT PROPOSALS & OUTCOME TRACKING =====================
    
    def save_sent_proposal(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a generated proposal that was sent to a client.
        Tracks outcomes: sent → viewed → hired
        
        Args:
            proposal_data: Contains proposal text, job details, and metadata
            
        Returns:
            Inserted document with proposal_id
        """
        try:
            # Generate unique proposal_id
            proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
            
            document = {
                "proposal_id": proposal_id,
                "job_title": proposal_data.get("job_title", ""),
                "proposal_text": proposal_data.get("proposal_text", ""),
                "skills_required": proposal_data.get("skills_required", []),
                "word_count": proposal_data.get("word_count", 0),
                
                # Source: ai_generated or manual
                "source": proposal_data.get("source", "ai_generated"),
                
                # Outcome tracking
                "outcome": "sent",  # sent → viewed → hired | rejected
                "sent_at": datetime.utcnow(),
                "viewed_at": None,
                "hired_at": None,
                "outcome_updated_at": None,
                
                # For conversion rate calculation
                "discussion_initiated": False,
                "rejection_reason": None,
                
                # Metadata
                "created_at": datetime.utcnow()
            }
            
            result = self.db["sent_proposals"].insert_one(document)
            logger.info(f"Saved sent proposal: {proposal_id}")
            
            return {
                "proposal_id": proposal_id,
                "db_id": str(result.inserted_id),
                "outcome": "sent",
                "sent_at": document["sent_at"].isoformat()
            }
        except Exception as e:
            logger.error(f"Error saving sent proposal: {str(e)}")
            raise
    
    def update_proposal_outcome(
        self, 
        proposal_id: str, 
        outcome: str,
        discussion_initiated: bool = False,
        rejection_reason: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update the outcome of a sent proposal.
        
        Args:
            proposal_id: Unique proposal ID
            outcome: 'viewed', 'hired', or 'rejected'
            discussion_initiated: Whether client started a chat (validates message-market fit)
            rejection_reason: Optional reason if rejected
            
        Returns:
            Updated document or None
        """
        try:
            valid_outcomes = ["sent", "viewed", "hired", "rejected"]
            if outcome not in valid_outcomes:
                raise ValueError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
            
            update_data = {
                "outcome": outcome,
                "outcome_updated_at": datetime.utcnow()
            }
            
            # Set timestamp based on outcome
            if outcome == "viewed":
                update_data["viewed_at"] = datetime.utcnow()
                update_data["discussion_initiated"] = discussion_initiated
            elif outcome == "hired":
                update_data["hired_at"] = datetime.utcnow()
                # If hired, they definitely viewed and discussed
                update_data["viewed_at"] = update_data.get("viewed_at") or datetime.utcnow()
                update_data["discussion_initiated"] = True
            elif outcome == "rejected":
                update_data["rejection_reason"] = rejection_reason
            
            result = self.db["sent_proposals"].find_one_and_update(
                {"proposal_id": proposal_id},
                {"$set": update_data},
                return_document=True
            )
            
            if result:
                logger.info(f"Updated proposal {proposal_id} outcome to: {outcome}")
                return {
                    "proposal_id": proposal_id,
                    "outcome": outcome,
                    "discussion_initiated": result.get("discussion_initiated", False),
                    "rejection_reason": result.get("rejection_reason"),
                    "updated_at": update_data["outcome_updated_at"].isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error updating proposal outcome: {str(e)}")
            raise
    
    def get_sent_proposals(
        self, 
        skip: int = 0, 
        limit: int = 50,
        outcome_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get sent proposals with optional filtering by outcome.
        
        Args:
            skip: Number to skip (pagination)
            limit: Maximum to return
            outcome_filter: Filter by outcome (sent, viewed, hired, rejected)
            
        Returns:
            List of sent proposals
        """
        try:
            query = {}
            if outcome_filter:
                query["outcome"] = outcome_filter
            
            results = list(
                self.db["sent_proposals"]
                .find(query)
                .sort("sent_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for r in results:
                r["_id"] = str(r["_id"])
            
            return results
        except Exception as e:
            logger.error(f"Error getting sent proposals: {str(e)}")
            return []
    
    def get_sent_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get a single sent proposal by ID"""
        try:
            result = self.db["sent_proposals"].find_one({"proposal_id": proposal_id})
            if result:
                result["_id"] = str(result["_id"])
            return result
        except Exception as e:
            logger.error(f"Error getting sent proposal: {str(e)}")
            return None
    
    def get_proposal_conversion_stats(self) -> Dict[str, Any]:
        """
        Get conversion statistics for proposals.
        Used to calculate proposal effectiveness and Message-Market Fit.
        
        Returns:
            Statistics including conversion rates, view rates, etc.
        """
        try:
            total = self.db["sent_proposals"].count_documents({})
            viewed = self.db["sent_proposals"].count_documents({"outcome": {"$in": ["viewed", "hired"]}})
            hired = self.db["sent_proposals"].count_documents({"outcome": "hired"})
            discussions = self.db["sent_proposals"].count_documents({"discussion_initiated": True})
            rejected = self.db["sent_proposals"].count_documents({"outcome": "rejected"})
            
            return {
                "total_sent": total,
                "total_viewed": viewed,
                "total_hired": hired,
                "total_discussions": discussions,
                "total_rejected": rejected,
                "view_rate": round((viewed / total * 100), 1) if total > 0 else 0,
                "hire_rate": round((hired / total * 100), 1) if total > 0 else 0,
                "discussion_rate": round((discussions / total * 100), 1) if total > 0 else 0,
                # Message-Market Fit: Viewed + Discussion = Hook + Approach worked
                "message_market_fit": round((discussions / viewed * 100), 1) if viewed > 0 else 0,
                # Conversion from view to hire
                "view_to_hire_rate": round((hired / viewed * 100), 1) if viewed > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting conversion stats: {str(e)}")
            return {}
    
    def get_effective_proposals(self, min_outcome: str = "viewed", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get proposals that were at least viewed (validates Hook effectiveness).
        Used for AI learning - these proposals have proven message-market fit.
        
        Args:
            min_outcome: Minimum outcome to include ('viewed' or 'hired')
            limit: Maximum to return
            
        Returns:
            List of effective proposals
        """
        try:
            if min_outcome == "hired":
                query = {"outcome": "hired"}
            else:  # viewed or better
                query = {"outcome": {"$in": ["viewed", "hired"]}}
            
            results = list(
                self.db["sent_proposals"]
                .find(query)
                .sort([("outcome", DESCENDING), ("sent_at", DESCENDING)])
                .limit(limit)
            )
            
            for r in results:
                r["_id"] = str(r["_id"])
            
            return results
        except Exception as e:
            logger.error(f"Error getting effective proposals: {str(e)}")
            return []

    # ===================== USER PROFILE OPERATIONS =====================
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile (singleton - one profile per instance)"""
        try:
            profile = self.db["user_profile"].find_one({"user_id": "default"})
            if not profile:
                # Create default profile
                profile = {
                    "user_id": "default",
                    "name": "",
                    "email": "",
                    "upwork_url": "",
                    "bio": "",
                    "hourly_rate": 0,
                    "skills": [],
                    "timezone": "UTC",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                self.db["user_profile"].insert_one(profile)
            profile["_id"] = str(profile["_id"])
            return profile
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return {}
    
    def update_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile"""
        try:
            data["updated_at"] = datetime.utcnow()
            self.db["user_profile"].update_one(
                {"user_id": "default"},
                {"$set": data},
                upsert=True
            )
            return self.get_profile()
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            raise

    # ===================== ANALYTICS OPERATIONS =====================
    
    def _build_date_query(self, since: Optional[datetime], source: Optional[str] = None) -> Dict[str, Any]:
        """Build MongoDB query with optional date and source filters"""
        query = {}
        if since:
            query["sent_at"] = {"$gte": since}
        if source:
            query["source"] = source
        return query
    
    def get_analytics_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get dashboard summary statistics"""
        try:
            base_query = self._build_date_query(since)
            col = self.db["sent_proposals"]
            
            total = col.count_documents(base_query)
            ai_count = col.count_documents({**base_query, "source": "ai_generated"})
            manual_count = col.count_documents({**base_query, "source": "manual"})
            hired = col.count_documents({**base_query, "outcome": "hired"})
            viewed = col.count_documents({**base_query, "outcome": {"$in": ["viewed", "hired"]}})
            discussions = col.count_documents({**base_query, "discussion_initiated": True})
            
            ai_hired = col.count_documents({**base_query, "source": "ai_generated", "outcome": "hired"})
            manual_hired = col.count_documents({**base_query, "source": "manual", "outcome": "hired"})
            
            return {
                "total_proposals": total,
                "ai_generated_count": ai_count,
                "manual_count": manual_count,
                "total_hired": hired,
                "total_viewed": viewed,
                "overall_hire_rate": round(hired / total * 100, 1) if total else 0,
                "ai_hire_rate": round(ai_hired / ai_count * 100, 1) if ai_count else 0,
                "manual_hire_rate": round(manual_hired / manual_count * 100, 1) if manual_count else 0,
                "message_market_fit": round(discussions / viewed * 100, 1) if viewed else 0
            }
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def get_conversion_funnel(self, since: Optional[datetime] = None, source: Optional[str] = None) -> Dict[str, Any]:
        """Get conversion funnel: Sent → Viewed → Discussed → Hired"""
        try:
            query = self._build_date_query(since, source)
            col = self.db["sent_proposals"]
            
            sent = col.count_documents(query)
            viewed = col.count_documents({**query, "outcome": {"$in": ["viewed", "hired"]}})
            discussed = col.count_documents({**query, "discussion_initiated": True})
            hired = col.count_documents({**query, "outcome": "hired"})
            
            funnel = [
                {"stage": "Sent", "count": sent, "percentage": 100.0},
                {"stage": "Viewed", "count": viewed, "percentage": round(viewed / sent * 100, 1) if sent else 0},
                {"stage": "Discussed", "count": discussed, "percentage": round(discussed / sent * 100, 1) if sent else 0},
                {"stage": "Hired", "count": hired, "percentage": round(hired / sent * 100, 1) if sent else 0}
            ]
            return {"funnel": funnel, "total_sent": sent}
        except Exception as e:
            logger.error(f"Error getting funnel: {e}")
            return {"funnel": [], "total_sent": 0}
    
    def get_source_comparison(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Compare AI-generated vs manual proposals"""
        try:
            col = self.db["sent_proposals"]
            
            def get_stats(source: str) -> Dict[str, Any]:
                query = self._build_date_query(since, source)
                total = col.count_documents(query)
                hired = col.count_documents({**query, "outcome": "hired"})
                viewed = col.count_documents({**query, "outcome": {"$in": ["viewed", "hired"]}})
                return {
                    "total": total,
                    "hired": hired,
                    "viewed": viewed,
                    "hire_rate": round(hired / total * 100, 1) if total else 0,
                    "view_rate": round(viewed / total * 100, 1) if total else 0
                }
            
            ai = get_stats("ai_generated")
            manual = get_stats("manual")
            
            # Calculate effectiveness: how much better AI performs
            effectiveness = 1.0
            if manual["hire_rate"] > 0:
                effectiveness = round(ai["hire_rate"] / manual["hire_rate"], 2)
            elif ai["hire_rate"] > 0:
                effectiveness = float("inf")
            
            return {"ai_generated": ai, "manual": manual, "ai_effectiveness": effectiveness}
        except Exception as e:
            logger.error(f"Error getting comparison: {e}")
            return {"ai_generated": {}, "manual": {}, "ai_effectiveness": 1.0}
    
    def get_combined_funnel(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get combined funnel data for AI-generated and Manual proposals.
        Returns full flow: Generated → Sent → Viewed → Discussed → Hired
        """
        try:
            sent_col = self.db["sent_proposals"]
            proposals_col = self.db["proposals"]
            
            def get_funnel_for_source(source: str) -> Dict[str, Any]:
                """Get funnel data for a specific source (ai_generated or manual)"""
                query = self._build_date_query(since, source)
                
                # Count proposals in each stage
                sent = sent_col.count_documents(query)
                viewed = sent_col.count_documents({**query, "outcome": {"$in": ["viewed", "hired"]}})
                discussed = sent_col.count_documents({**query, "discussion_initiated": True})
                hired = sent_col.count_documents({**query, "outcome": "hired"})
                
                # Generated count from proposals collection (AI only)
                generated = 0
                if source == "ai_generated":
                    gen_query = {"created_at": {"$gte": since}} if since else {}
                    generated = proposals_col.count_documents(gen_query)
                else:
                    generated = sent  # Manual proposals are generated when sent
                
                # Calculate percentages relative to sent (funnel starts at sent)
                base = sent if sent > 0 else 1
                
                return {
                    "generated": generated,
                    "sent": sent,
                    "viewed": viewed,
                    "discussed": discussed,
                    "hired": hired,
                    "rates": {
                        "sent_rate": round(sent / generated * 100, 1) if generated > 0 else 0,
                        "view_rate": round(viewed / base * 100, 1),
                        "discuss_rate": round(discussed / base * 100, 1),
                        "hire_rate": round(hired / base * 100, 1)
                    }
                }
            
            ai_data = get_funnel_for_source("ai_generated")
            manual_data = get_funnel_for_source("manual")
            
            # Calculate totals
            total_generated = ai_data["generated"] + manual_data["generated"]
            total_sent = ai_data["sent"] + manual_data["sent"]
            total_viewed = ai_data["viewed"] + manual_data["viewed"]
            total_discussed = ai_data["discussed"] + manual_data["discussed"]
            total_hired = ai_data["hired"] + manual_data["hired"]
            
            return {
                "ai": ai_data,
                "manual": manual_data,
                "totals": {
                    "generated": total_generated,
                    "sent": total_sent,
                    "viewed": total_viewed,
                    "discussed": total_discussed,
                    "hired": total_hired
                },
                "ai_share": round(ai_data["sent"] / total_sent * 100, 1) if total_sent > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting combined funnel: {e}")
            return {
                "ai": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0, "rates": {}},
                "manual": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0, "rates": {}},
                "totals": {"generated": 0, "sent": 0, "viewed": 0, "discussed": 0, "hired": 0},
                "ai_share": 0
            }
    
    def get_proposal_trends(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get daily proposal trends for charts"""
        try:
            if not since:
                since = datetime.utcnow() - timedelta(days=30)
            
            pipeline = [
                {"$match": {"sent_at": {"$gte": since}}},
                {"$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$sent_at"}},
                    "sent": {"$sum": 1},
                    "hired": {"$sum": {"$cond": [{"$eq": ["$outcome", "hired"]}, 1, 0]}}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            results = list(self.db["sent_proposals"].aggregate(pipeline))
            return [
                {
                    "date": r["_id"],
                    "sent": r["sent"],
                    "hired": r["hired"],
                    "hire_rate": round(r["hired"] / r["sent"] * 100, 1) if r["sent"] else 0
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return []
    
    def get_skills_performance(self, since: Optional[datetime] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get performance breakdown by skill"""
        try:
            match_stage = {"$match": {"sent_at": {"$gte": since}}} if since else {"$match": {}}
            
            pipeline = [
                match_stage,
                {"$unwind": "$skills_required"},
                {"$group": {
                    "_id": "$skills_required",
                    "total": {"$sum": 1},
                    "hired": {"$sum": {"$cond": [{"$eq": ["$outcome", "hired"]}, 1, 0]}}
                }},
                {"$match": {"total": {"$gte": 2}}},  # At least 2 proposals
                {"$sort": {"hired": -1, "total": -1}},
                {"$limit": limit}
            ]
            
            results = list(self.db["sent_proposals"].aggregate(pipeline))
            return [
                {
                    "skill": r["_id"],
                    "total": r["total"],
                    "hired": r["hired"],
                    "hire_rate": round(r["hired"] / r["total"] * 100, 1) if r["total"] else 0
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting skills performance: {e}")
            return []
    
    def get_industry_performance(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get performance breakdown by industry"""
        try:
            match_stage = {"$match": {"sent_at": {"$gte": since}}} if since else {"$match": {}}
            
            pipeline = [
                match_stage,
                {"$group": {
                    "_id": "$industry",
                    "total": {"$sum": 1},
                    "hired": {"$sum": {"$cond": [{"$eq": ["$outcome", "hired"]}, 1, 0]}},
                    "viewed": {"$sum": {"$cond": [{"$in": ["$outcome", ["viewed", "hired"]]}, 1, 0]}}
                }},
                {"$sort": {"total": -1}}
            ]
            
            results = list(self.db["sent_proposals"].aggregate(pipeline))
            return [
                {
                    "industry": r["_id"] or "general",
                    "total": r["total"],
                    "hired": r["hired"],
                    "viewed": r["viewed"],
                    "hire_rate": round(r["hired"] / r["total"] * 100, 1) if r["total"] else 0
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error getting industry performance: {e}")
            return []
    
    # ===================== ADMIN: API KEY & ACTIVITY LOGGING =====================
    
    def log_activity(self, user_name: str, action: str, target: str = None, details: dict = None, ip: str = None):
        """
        Log an activity for audit trail.
        
        Args:
            user_name: Name of user who performed action (or 'Super Admin')
            action: Action type (create_user, revoke_user, generate, etc.)
            target: What was affected (user name, proposal title, etc.)
            details: Additional context
            ip: IP address (optional)
        """
        try:
            self.db["activity_log"].insert_one({
                "user_name": user_name,
                "action": action,
                "target": target,
                "details": details,
                "ip": ip,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
    
    def init_admin_collections(self):
        """Initialize admin collections and indexes"""
        try:
            # API Keys collection
            self.db["api_keys"].create_index([("key_hash", ASCENDING)], unique=True)
            self.db["api_keys"].create_index([("name", ASCENDING)])
            self.db["api_keys"].create_index([("is_active", ASCENDING)])
            self.db["api_keys"].create_index([("created_at", DESCENDING)])
            
            # Activity Log collection
            self.db["activity_log"].create_index([("timestamp", DESCENDING)])
            self.db["activity_log"].create_index([("key_prefix", ASCENDING)])
            self.db["activity_log"].create_index([("action", ASCENDING)])
            
            logger.info("Admin collections initialized")
        except Exception as e:
            logger.error(f"Error initializing admin collections: {e}")


# Singleton instance
_db_manager: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.init_admin_collections()
    return _db_manager

def close_db():
    """Close database connection"""
    global _db_manager
    if _db_manager:
        _db_manager.disconnect()
        _db_manager = None