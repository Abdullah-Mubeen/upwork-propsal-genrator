"""
Advanced Data Chunking Strategy for RAG-Optimized Proposal Generation

Implements the 4-chunk strategy:
1. JOB_FACTS_CHUNK → Cleaned job description with strong metadata
2. PROPOSAL_CHUNK → Full, unbroken proposal text
3. FEEDBACK_CHUNK → Pure feedback (no ratings, dates, labels)
4. TEMPLATE_CHUNK → Writing templates and winning patterns

Each chunk is JSON-ready with structured metadata for filtering and retrieval.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkTypeEnum(str, Enum):
    """Chunk type classification"""
    JOB_FACTS = "job_facts"
    PROPOSAL = "proposal"
    FEEDBACK = "feedback"
    TEMPLATE = "template"


@dataclass
class ChunkMetadata:
    """Base metadata structure for all chunks"""
    job_id: str
    chunk_type: str
    created_at: str
    source: str = ""


@dataclass
class JobFactsMetadata(ChunkMetadata):
    """Metadata for JOB_FACTS_CHUNK"""
    title: str = ""
    company: str = ""
    skills: List[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget_usd: Optional[float] = None
    urgency: str = "normal"  # low, normal, high
    category: str = "general"  # website, mobile_app, backend_api, etc.
    industry: str = "general"


@dataclass
class ProposalMetadata(ChunkMetadata):
    """Metadata for PROPOSAL_CHUNK"""
    title: str = ""
    company: str = ""
    skills: List[str] = None
    did_win: bool = False
    price_usd: Optional[float] = None
    style: str = "formal"  # formal, casual, technical, sales
    tone: str = "professional"  # professional, friendly, technical
    industry: str = "general"


@dataclass
class FeedbackMetadata(ChunkMetadata):
    """Metadata for FEEDBACK_CHUNK"""
    title: str = ""
    company: str = ""
    rating: Optional[float] = None  # 1-5
    sentiment: str = "neutral"  # positive, negative, neutral, mixed
    industry: str = "general"


@dataclass
class TemplateMetadata(ChunkMetadata):
    """Metadata for TEMPLATE_CHUNK"""
    template_name: str = ""
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    use_cases: List[str] = None


class JobFactsChunker:
    """
    Creates JOB_FACTS_CHUNK from job descriptions.
    
    Extracts and cleans:
    - Job description (remove noise, formatting, links)
    - Requirements and responsibilities
    - Key qualifications
    - Does NOT include ratings, dates, or metadata in text
    """

    @staticmethod
    def extract_job_facts(job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract clean job facts chunk from raw job data.
        
        Args:
            job_data: Raw job data dictionary
            
        Returns:
            Chunk dictionary with text and metadata
        """
        # Get job description and clean it
        job_desc = job_data.get("job_description", "")
        if not job_desc:
            return None

        # Clean the text: remove formatting noise, extra whitespace, URLs, etc.
        cleaned_text = JobFactsChunker._clean_text(job_desc)

        # Extract metadata
        job_id = job_data.get("contract_id", "unknown")
        title = job_data.get("job_title", "")
        company = job_data.get("company_name", "")
        skills = job_data.get("skills_required", [])
        industry = job_data.get("industry", "general")
        
        # Parse dates if available
        start_date = job_data.get("start_date")
        end_date = job_data.get("end_date")
        
        # Determine urgency from project data
        urgency = "high" if job_data.get("urgent_adhoc") else "normal"
        
        # Map task_type to category
        task_type = job_data.get("task_type", "other")
        category = JobFactsChunker._map_task_to_category(task_type)

        metadata = JobFactsMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.JOB_FACTS.value,
            created_at=datetime.utcnow().isoformat(),
            source="job_description",
            title=title,
            company=company,
            skills=skills or [],
            start_date=start_date,
            end_date=end_date,
            budget_usd=None,  # Extract if available in future
            urgency=urgency,
            category=category,
            industry=industry
        )

        return {
            "chunk_type": ChunkTypeEnum.JOB_FACTS.value,
            "text": cleaned_text,
            "metadata": JobFactsChunker._metadata_to_dict(metadata),
            "length": len(cleaned_text),
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean job description text:
        - Remove URLs
        - Remove extra whitespace
        - Remove HTML-like tags
        - Normalize line breaks
        - Remove common noise patterns
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove common formatting noise like "***", "---", etc.
        text = re.sub(r'[\*\-_]{3,}', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def _map_task_to_category(task_type: str) -> str:
        """Map task_type to category"""
        mapping = {
            "website": "website",
            "mobile_app": "mobile_app",
            "backend_api": "backend_api",
            "frontend": "frontend",
            "full_stack": "full_stack",
            "consultation": "consultation",
            "maintenance": "maintenance",
            "content": "content",
            "design": "design",
        }
        return mapping.get(task_type, "general")

    @staticmethod
    def _metadata_to_dict(metadata: JobFactsMetadata) -> Dict[str, Any]:
        """Convert dataclass to dictionary, excluding None values"""
        data = asdict(metadata)
        return {k: v for k, v in data.items() if v is not None}


class ProposalChunker:
    """
    Creates PROPOSAL_CHUNK from proposals.
    
    Rules:
    - NEVER split the proposal text
    - Keep proposal as one complete chunk
    - Extract metadata for filtering
    - Mark whether it won or not
    """

    @staticmethod
    def extract_proposal(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract proposal as a single, unbroken chunk.
        
        Args:
            job_data: Raw job data dictionary
            
        Returns:
            Single proposal chunk or None if no proposal
        """
        proposal_text = job_data.get("your_proposal_text", "")
        if not proposal_text or len(proposal_text.strip()) < 50:
            return None

        job_id = job_data.get("contract_id", "unknown")
        title = job_data.get("job_title", "")
        company = job_data.get("company_name", "")
        skills = job_data.get("skills_required", [])
        industry = job_data.get("industry", "general")

        # Determine if proposal won based on project status
        # (In a real system, this would come from explicit tracking)
        did_win = job_data.get("project_status") == "completed"

        metadata = ProposalMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.PROPOSAL.value,
            created_at=datetime.utcnow().isoformat(),
            source="proposal_text",
            title=title,
            company=company,
            skills=skills or [],
            did_win=did_win,
            price_usd=None,  # Extract if available
            style="formal",  # Infer from content in future
            tone="professional",  # Infer from sentiment analysis
            industry=industry
        )

        return {
            "chunk_type": ChunkTypeEnum.PROPOSAL.value,
            "text": proposal_text.strip(),  # NO splitting
            "metadata": ProposalChunker._metadata_to_dict(metadata),
            "length": len(proposal_text),
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _metadata_to_dict(metadata: ProposalMetadata) -> Dict[str, Any]:
        """Convert dataclass to dictionary, excluding None values"""
        data = asdict(metadata)
        return {k: v for k, v in data.items() if v is not None}


class FeedbackChunker:
    """
    Creates FEEDBACK_CHUNK from client reviews.
    
    Rules:
    - Extract ONLY clean feedback text
    - Remove ratings, dates, labels, stars
    - Remove formatting noise
    - One chunk per review/feedback entry
    """

    @staticmethod
    def extract_feedback(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract clean feedback chunk.
        
        Args:
            job_data: Raw job data dictionary
            
        Returns:
            Feedback chunk or None if no feedback
        """
        feedback_text = job_data.get("client_feedback", "")
        if not feedback_text or len(feedback_text.strip()) < 20:
            return None

        # Clean the feedback: remove ratings, dates, labels
        cleaned_feedback = FeedbackChunker._clean_feedback(feedback_text)

        if not cleaned_feedback or len(cleaned_feedback) < 20:
            return None

        job_id = job_data.get("contract_id", "unknown")
        title = job_data.get("job_title", "")
        company = job_data.get("company_name", "")
        industry = job_data.get("industry", "general")

        # Determine sentiment (basic heuristic, could be improved with ML)
        sentiment = FeedbackChunker._determine_sentiment(cleaned_feedback)

        metadata = FeedbackMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.FEEDBACK.value,
            created_at=datetime.utcnow().isoformat(),
            source="client_feedback",
            title=title,
            company=company,
            rating=None,  # No ratings in text
            sentiment=sentiment,
            industry=industry
        )

        return {
            "chunk_type": ChunkTypeEnum.FEEDBACK.value,
            "text": cleaned_feedback,
            "metadata": FeedbackChunker._metadata_to_dict(metadata),
            "length": len(cleaned_feedback),
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _clean_feedback(text: str) -> str:
        """
        Clean feedback text:
        - Remove star ratings (★, ⭐, "5 stars", etc.)
        - Remove dates and timestamps
        - Remove labels ("Review:", "Feedback:", etc.)
        - Remove formatting noise
        """
        if not text:
            return ""

        # Remove star patterns
        text = re.sub(r'[★⭐✯\*]{1,5}', '', text)
        text = re.sub(r'(\d+)\s*(?:star|/5|out of 5)', '', text, flags=re.IGNORECASE)

        # Remove date labels (Date:, Posted:, etc.)
        text = re.sub(r'(?:Date|Posted|Review Date|Created):\s*', '', text, flags=re.IGNORECASE)
        
        # Remove common date patterns (e.g., "March 15, 2024", "15/03/2024", "03-15-2024")
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', text)
        text = re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', '', text, flags=re.IGNORECASE)

        # Remove common labels/prefixes
        text = re.sub(r'(?:^|\s)(Review|Feedback|Comment|Note|By):\s*', ' ', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    @staticmethod
    def _determine_sentiment(text: str) -> str:
        """
        Simple sentiment detection based on keywords.
        In production, use a proper NLP model.
        """
        positive_words = [
            'excellent', 'great', 'amazing', 'wonderful', 'fantastic',
            'love', 'awesome', 'perfect', 'professional', 'highly',
            'recommend', 'best', 'impressed', 'satisfied', 'happy'
        ]
        negative_words = [
            'poor', 'bad', 'terrible', 'awful', 'horrible',
            'hate', 'disappointed', 'issues', 'problem', 'broken',
            'not', 'never', 'worse', 'regret', 'unsatisfied'
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def _metadata_to_dict(metadata: FeedbackMetadata) -> Dict[str, Any]:
        """Convert dataclass to dictionary, excluding None values"""
        data = asdict(metadata)
        return {k: v for k, v in data.items() if v is not None}


class TemplateChunker:
    """
    Creates TEMPLATE_CHUNK from internal writing templates.
    
    These are extracted from high-performing proposals
    and can be used as reference templates for future generations.
    """

    @staticmethod
    def extract_template(
        template_name: str,
        template_text: str,
        category: str = "general",
        job_id: str = "template",
        use_cases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a template chunk from a writing template.
        
        Args:
            template_name: Name of the template
            template_text: Template content
            category: Template category/type
            job_id: Reference job ID (optional)
            use_cases: List of use cases for this template
            
        Returns:
            Template chunk
        """
        metadata = TemplateMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.TEMPLATE.value,
            created_at=datetime.utcnow().isoformat(),
            source="template_library",
            template_name=template_name,
            category=category,
            difficulty="medium",
            use_cases=use_cases or []
        )

        return {
            "chunk_type": ChunkTypeEnum.TEMPLATE.value,
            "text": template_text,
            "metadata": TemplateChunker._metadata_to_dict(metadata),
            "length": len(template_text),
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _metadata_to_dict(metadata: TemplateMetadata) -> Dict[str, Any]:
        """Convert dataclass to dictionary, excluding None values"""
        data = asdict(metadata)
        return {k: v for k, v in data.items() if v is not None}


class AdvancedChunkProcessor:
    """
    Main processor that orchestrates all chunk types.
    
    Returns JSON-ready chunks optimized for RAG embedding and retrieval.
    """

    def __init__(self):
        """Initialize chunk processor"""
        self.job_facts_chunker = JobFactsChunker()
        self.proposal_chunker = ProposalChunker()
        self.feedback_chunker = FeedbackChunker()
        self.template_chunker = TemplateChunker()
        logger.info("AdvancedChunkProcessor initialized")

    def process_job_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete job data into all chunk types.
        
        Args:
            job_data: Complete job data dictionary
            
        Returns:
            Dictionary with all chunks organized by type
        """
        job_id = job_data.get("contract_id", "unknown")
        logger.info(f"Processing job data for {job_id}")

        chunks = {
            "job_id": job_id,
            "processed_at": datetime.utcnow().isoformat(),
            "chunks": {
                "job_facts": None,
                "proposal": None,
                "feedback": None,
                "templates": []
            },
            "summary": {
                "total_chunks": 0,
                "total_text_length": 0,
                "chunk_types": []
            }
        }

        # Process JOB_FACTS
        job_facts_chunk = JobFactsChunker.extract_job_facts(job_data)
        if job_facts_chunk:
            chunks["chunks"]["job_facts"] = job_facts_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += job_facts_chunk["length"]
            chunks["summary"]["chunk_types"].append("job_facts")
            logger.debug(f"Created JOB_FACTS chunk for {job_id}")

        # Process PROPOSAL
        proposal_chunk = ProposalChunker.extract_proposal(job_data)
        if proposal_chunk:
            chunks["chunks"]["proposal"] = proposal_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += proposal_chunk["length"]
            chunks["summary"]["chunk_types"].append("proposal")
            logger.debug(f"Created PROPOSAL chunk for {job_id}")

        # Process FEEDBACK
        feedback_chunk = FeedbackChunker.extract_feedback(job_data)
        if feedback_chunk:
            chunks["chunks"]["feedback"] = feedback_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += feedback_chunk["length"]
            chunks["summary"]["chunk_types"].append("feedback")
            logger.debug(f"Created FEEDBACK chunk for {job_id}")

        logger.info(
            f"✓ Processed {job_id}: {chunks['summary']['total_chunks']} chunks, "
            f"{chunks['summary']['total_text_length']} total chars"
        )

        return chunks

    def process_batch(self, job_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple job data entries.
        
        Args:
            job_data_list: List of job data dictionaries
            
        Returns:
            List of processed chunk sets
        """
        processed_chunks = []
        for job_data in job_data_list:
            try:
                processed = self.process_job_data(job_data)
                processed_chunks.append(processed)
            except Exception as e:
                logger.error(f"Error processing job data: {str(e)}", exc_info=True)

        return processed_chunks

    def get_all_chunks_flat(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get all chunks in a flat list format (better for batch embedding).
        
        Args:
            job_data: Complete job data dictionary
            
        Returns:
            List of individual chunks, each JSON-ready
        """
        processed = self.process_job_data(job_data)
        flat_chunks = []

        for chunk_type, chunk_data in processed["chunks"].items():
            if chunk_data:
                flat_chunks.append(chunk_data)

        return flat_chunks

    def validate_chunk_json(self, chunk: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that a chunk is properly formatted and JSON-serializable.
        
        Args:
            chunk: Chunk dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to serialize to JSON
            json.dumps(chunk)
            
            # Check required fields
            required_fields = ["chunk_type", "text", "metadata", "created_at"]
            for field in required_fields:
                if field not in chunk:
                    return False, f"Missing required field: {field}"
            
            # Check metadata is dict
            if not isinstance(chunk["metadata"], dict):
                return False, "Metadata must be a dictionary"
            
            # Check chunk_type is valid
            valid_types = [e.value for e in ChunkTypeEnum]
            if chunk["chunk_type"] not in valid_types:
                return False, f"Invalid chunk_type: {chunk['chunk_type']}"
            
            return True, "Valid"
        
        except (TypeError, ValueError) as e:
            return False, f"JSON serialization error: {str(e)}"

    def export_chunks_json(self, chunks: Dict[str, Any]) -> str:
        """
        Export processed chunks as JSON string.
        
        Args:
            chunks: Processed chunks dictionary
            
        Returns:
            JSON string
        """
        return json.dumps(chunks, indent=2, default=str)


# Backward compatibility: Keep old interface working
def chunk_training_data_new(job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Wrapper function for backward compatibility with old DataChunker interface.
    
    Args:
        job_data: Complete job data dictionary
        
    Returns:
        List of chunks in flat format
    """
    processor = AdvancedChunkProcessor()
    return processor.get_all_chunks_flat(job_data)
