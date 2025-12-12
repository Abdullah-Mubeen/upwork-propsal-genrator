import logging
from typing import Optional, List, Dict, Any, Tuple
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
from contextlib import contextmanager
from datetime import datetime
from bson.objectid import ObjectId
import uuid
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
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                retryWrites=True,
                maxPoolSize=50,
                minPoolSize=10
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
                "skill_embeddings": "Embeddings for skill vectors"
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
            
            logger. info("Successfully created all database indexes")
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


# Singleton instance
_db_manager: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def close_db():
    """Close database connection"""
    global _db_manager
    if _db_manager:
        _db_manager. disconnect()
        _db_manager = None