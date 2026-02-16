"""
Advanced Data Chunking Strategy for RAG-Optimized Proposal Generation

⚠️ DEPRECATED: This 5-chunk strategy is being replaced by single-vector-per-project.
Use app/services/embedding_service.py and app/infra/mongodb/repositories/portfolio_repo.py instead.
This file is kept for backward compatibility with existing training_data ingestion.
Will be removed in issue #22.

Implements the 5-layer SEMANTIC chunking strategy:
1. CONTEXT_SNAPSHOT → Company, Industry, Job Title, Task Type, Urgency
2. REQUIREMENTS_PROFILE → Skills Required, Job Description, Task Complexity
3. TIMELINE_SCOPE → Start Date, End Date, Status, Project Duration
4. DELIVERABLES_PORTFOLIO → Portfolio Links (for work reference)
5. FEEDBACK_OUTCOMES → Client Feedback, Proposal Analysis, Success Patterns

Each chunk is purpose-built for specific retrieval patterns and learning outcomes.
Metadata is multi-dimensional for optimal filtering and context assembly.
"""
import warnings
warnings.warn(
    "advanced_chunker is deprecated. Use embedding_service.py + portfolio_repo.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkTypeEnum(str, Enum):
    """Chunk type classification - purpose-built chunks"""
    CONTEXT_SNAPSHOT = "context_snapshot"  # Company + industry + job title + urgency
    REQUIREMENTS_PROFILE = "requirements_profile"  # Skills + job description + complexity
    TIMELINE_SCOPE = "timeline_scope"  # Duration + status + timeline analysis
    DELIVERABLES_PORTFOLIO = "deliverables_portfolio"  # Portfolio links for reference
    FEEDBACK_OUTCOMES = "feedback_outcomes"  # Client feedback + success patterns


@dataclass
class EnhancedMetadata:
    """Enhanced multi-dimensional metadata for optimal retrieval"""
    job_id: str
    chunk_type: str
    created_at: str
    
    # Retrieval dimensions
    industry: str = "general"
    task_type: str = "general"
    skills: List[str] = None  # For skill-based filtering
    urgency: str = "normal"  # low, normal, high
    
    # Context dimensions
    company_name: str = ""
    job_title: str = ""
    
    # Learning dimensions
    proposal_effectiveness: Optional[float] = None  # 0-1 score
    client_satisfaction: Optional[float] = None  # 1-5 scale
    
    # Timeline dimensions
    duration_days: Optional[int] = None
    is_completed: bool = False
    
    # Complexity dimensions
    task_complexity: str = "medium"  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class ContextSnapshotChunker:
    """
    Creates CONTEXT_SNAPSHOT chunk.
    
    Purpose: Fast metadata filtering for industry, task type, urgency matching
    Contains: Company Name, Industry, Job Title, Task Type, Urgency
    Use Case: "Find all React projects in SaaS that were urgent"
    """

    @staticmethod
    def extract(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract context snapshot chunk"""
        job_id = job_data.get("contract_id", "unknown")
        company = job_data.get("company_name", "")
        industry = job_data.get("industry", "general")
        title = job_data.get("job_title", "")
        task_type = job_data.get("task_type", "general")
        urgency = "high" if job_data.get("urgent_adhoc") else "normal"
        
        if not (company and title):
            return None

        # Create semantic text that captures context
        context_text = f"""Project Context: {company} ({industry})
Position: {title}
Type: {task_type}
Urgency: {urgency}
Nature: {'Ad-hoc/Emergency' if job_data.get('urgent_adhoc') else 'Planned Project'}"""

        metadata = EnhancedMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.CONTEXT_SNAPSHOT.value,
            created_at=datetime.utcnow().isoformat(),
            industry=industry,
            task_type=task_type,
            company_name=company,
            job_title=title,
            urgency=urgency,
            proposal_effectiveness=job_data.get("proposal_effectiveness_score")
        )

        return {
            "chunk_type": ChunkTypeEnum.CONTEXT_SNAPSHOT.value,
            "text": context_text,
            "metadata": metadata.to_dict(),
            "length": len(context_text),
            "created_at": datetime.utcnow().isoformat()
        }


class RequirementsProfileChunker:
    """
    Creates REQUIREMENTS_PROFILE chunk.
    
    Purpose: Skill matching and job complexity assessment
    Contains: Skills Required, Job Description, Task Complexity
    Use Case: "Find projects requiring React and PostgreSQL"
    """

    @staticmethod
    def extract(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract requirements profile chunk"""
        job_id = job_data.get("contract_id", "unknown")
        skills = job_data.get("skills_required", [])
        job_desc = job_data.get("job_description", "")
        complexity = job_data.get("task_complexity", "medium")
        industry = job_data.get("industry", "general")
        task_type = job_data.get("task_type", "general")
        
        if not (skills and job_desc):
            return None

        # Clean job description
        cleaned_desc = RequirementsProfileChunker._clean_text(job_desc)
        
        # Create requirements text
        skills_str = ", ".join(skills)
        requirements_text = f"""Required Skills: {skills_str}

Complexity Level: {complexity}

Job Description:
{cleaned_desc}

Task Classification: {task_type} in {industry}"""

        metadata = EnhancedMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.REQUIREMENTS_PROFILE.value,
            created_at=datetime.utcnow().isoformat(),
            skills=skills,
            task_complexity=complexity,
            industry=industry,
            task_type=task_type
        )

        return {
            "chunk_type": ChunkTypeEnum.REQUIREMENTS_PROFILE.value,
            "text": requirements_text,
            "metadata": metadata.to_dict(),
            "length": len(requirements_text),
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean job description"""
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class TimelineScopeChunker:
    """
    Creates TIMELINE_SCOPE chunk.
    
    Purpose: Project duration and schedule matching
    Contains: Start Date, End Date, Status, Duration Analysis
    Use Case: "Find similar quick projects (< 30 days)"
    """

    @staticmethod
    def extract(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract timeline scope chunk"""
        job_id = job_data.get("contract_id", "unknown")
        start_date = job_data.get("start_date")
        end_date = job_data.get("end_date")
        status = job_data.get("project_status", "unknown")
        
        # Calculate duration if dates available
        duration_days = None
        timeline_analysis = ""
        
        if start_date and end_date:
            try:
                from datetime import datetime as dt
                start = dt.fromisoformat(start_date.replace('Z', '+00:00'))
                end = dt.fromisoformat(end_date.replace('Z', '+00:00'))
                duration_days = (end - start).days
                duration_str = f"{duration_days} days"
                if duration_days <= 7:
                    speed = "Very Quick"
                elif duration_days <= 14:
                    speed = "Quick"
                elif duration_days <= 30:
                    speed = "Standard"
                else:
                    speed = "Extended"
                timeline_analysis = f"{speed} project ({duration_str})"
            except:
                timeline_analysis = f"Project from {start_date} to {end_date}"
        else:
            timeline_analysis = "Timeline: Not specified"

        timeline_text = f"""Timeline & Scope:
Project Status: {status.capitalize()}
Duration: {timeline_analysis}
Schedule: {start_date or 'Not specified'} → {end_date or 'Not specified'}
Completion: {'Completed/Successful' if status == 'completed' else 'In Progress/Planned'}"""

        # Store duration for metadata filtering
        if duration_days:
            job_data["project_duration_days"] = duration_days

        metadata = EnhancedMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.TIMELINE_SCOPE.value,
            created_at=datetime.utcnow().isoformat(),
            duration_days=duration_days,
            is_completed=status == "completed"
        )

        return {
            "chunk_type": ChunkTypeEnum.TIMELINE_SCOPE.value,
            "text": timeline_text,
            "metadata": metadata.to_dict(),
            "length": len(timeline_text),
            "created_at": datetime.utcnow().isoformat()
        }


class DeliverablesPortfolioChunker:
    """
    Creates DELIVERABLES_PORTFOLIO chunk.
    
    Purpose: Reference past work, deliverables, and portfolio examples
    Contains: Deliverables, Outcomes, Portfolio Links
    Use Case: "Find projects where I built membership systems with content migration"
    """

    @staticmethod
    def extract(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract deliverables portfolio chunk - CRITICAL for matching what was built"""
        job_id = job_data.get("contract_id", "unknown")
        portfolio_urls = job_data.get("portfolio_urls", [])
        deliverables = job_data.get("deliverables", [])
        outcomes = job_data.get("outcomes", "")
        
        # Create chunk if we have deliverables OR portfolio links
        if not deliverables and not portfolio_urls:
            return None

        # Build deliverables section
        deliverables_str = ""
        if deliverables:
            deliverables_str = "What Was Built:\n" + "\n".join([f"• {d}" for d in deliverables])
        
        # Build outcomes section
        outcomes_str = f"\nKey Outcome: {outcomes}" if outcomes else ""
        
        # Build portfolio section
        portfolio_str = ""
        if portfolio_urls:
            portfolio_str = "\n\nPortfolio References:\n" + "\n".join([f"• {url}" for url in portfolio_urls])
        
        portfolio_text = f"""Project Deliverables & Portfolio:

{deliverables_str}{outcomes_str}{portfolio_str}"""

        metadata = EnhancedMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.DELIVERABLES_PORTFOLIO.value,
            created_at=datetime.utcnow().isoformat()
        )

        return {
            "chunk_type": ChunkTypeEnum.DELIVERABLES_PORTFOLIO.value,
            "text": portfolio_text.strip(),
            "metadata": metadata.to_dict(),
            "deliverables": deliverables,  # Store for Pinecone metadata
            "outcomes": outcomes,
            "length": len(portfolio_text),
            "created_at": datetime.utcnow().isoformat()
        }


class FeedbackOutcomesChunker:
    """
    Creates FEEDBACK_OUTCOMES chunk.
    
    Purpose: Learning from past client feedback and success patterns
    Contains: Client Feedback, Satisfaction Score, Success Patterns
    Use Case: "What did clients praise in similar projects?"
    """

    @staticmethod
    def extract(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract feedback outcomes chunk"""
        job_id = job_data.get("contract_id", "unknown")
        feedback_text = job_data.get("client_feedback_text") or job_data.get("client_feedback", "")
        feedback_url = job_data.get("client_feedback_url", "")
        satisfaction = job_data.get("client_satisfaction")
        proposal_effectiveness = job_data.get("proposal_effectiveness_score")
        
        if not feedback_text or len(feedback_text.strip()) < 20:
            return None

        # Clean feedback (remove ratings, dates, etc.)
        cleaned_feedback = FeedbackOutcomesChunker._clean_feedback(feedback_text)
        
        # Determine sentiment
        sentiment = FeedbackOutcomesChunker._determine_sentiment(cleaned_feedback)
        
        # Build feedback text with context
        satisfaction_str = f"\nClient Satisfaction: {satisfaction}/5" if satisfaction else ""
        effectiveness_str = f"\nProposal Effectiveness: {proposal_effectiveness*100:.0f}%" if proposal_effectiveness else ""
        feedback_url_str = f"\n📎 Feedback URL: {feedback_url}" if feedback_url else ""
        
        feedback_outcomes_text = f"""Client Feedback & Outcomes:
Sentiment: {sentiment.capitalize()}
{satisfaction_str}{effectiveness_str}{feedback_url_str}

Client Comments:
{cleaned_feedback}"""

        metadata = EnhancedMetadata(
            job_id=job_id,
            chunk_type=ChunkTypeEnum.FEEDBACK_OUTCOMES.value,
            created_at=datetime.utcnow().isoformat(),
            proposal_effectiveness=proposal_effectiveness,
            client_satisfaction=satisfaction
        )

        return {
            "chunk_type": ChunkTypeEnum.FEEDBACK_OUTCOMES.value,
            "text": feedback_outcomes_text,
            "metadata": metadata.to_dict(),
            "length": len(feedback_outcomes_text),
            "created_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def _clean_feedback(text: str) -> str:
        """Remove ratings, dates, labels from feedback"""
        if not text:
            return ""
        # Remove star patterns
        text = re.sub(r'[★⭐✯\*]{1,5}', '', text)
        text = re.sub(r'(\d+)\s*(?:star|/5|out of 5)', '', text, flags=re.IGNORECASE)
        # Remove dates
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', text)
        # Remove labels
        text = re.sub(r'(?:Date|Posted|Review|Feedback|By):\s*', '', text, flags=re.IGNORECASE)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _determine_sentiment(text: str) -> str:
        """Determine sentiment from feedback"""
        positive_words = [
            'excellent', 'great', 'amazing', 'wonderful', 'fantastic',
            'love', 'awesome', 'perfect', 'professional', 'highly',
            'recommend', 'best', 'impressed', 'satisfied', 'happy'
        ]
        negative_words = [
            'poor', 'bad', 'terrible', 'awful', 'horrible',
            'hate', 'disappointed', 'issues', 'problem', 'broken'
        ]
        text_lower = text.lower()
        positive = sum(1 for w in positive_words if w in text_lower)
        negative = sum(1 for w in negative_words if w in text_lower)
        
        if positive > negative:
            return "positive"
        elif negative > positive:
            return "negative"
        return "neutral"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text"""
        if not text:
            return ""
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class AdvancedChunkProcessor:
    """
    Main processor using 5-layer SEMANTIC chunking strategy.
    
    Each chunk type serves a specific purpose in proposal generation:
    1. CONTEXT_SNAPSHOT: Fast metadata filtering (industry, task type, urgency)
    2. REQUIREMENTS_PROFILE: Skill matching and complexity
    3. TIMELINE_SCOPE: Duration matching
    4. DELIVERABLES_PORTFOLIO: Portfolio references
    5. FEEDBACK_OUTCOMES: Learning patterns from past projects
    
    Returns JSON-ready chunks optimized for hybrid retrieval.
    """

    def __init__(self):
        """Initialize chunk processor"""
        logger.info("AdvancedChunkProcessor initialized (5-layer semantic strategy)")

    def process_job_data(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete job data into 5 semantic chunks.
        
        Args:
            job_data: Complete job data dictionary
            
        Returns:
            Dictionary with all 5 chunk types
        """
        job_id = job_data.get("contract_id", "unknown")
        logger.info(f"[Chunking] Processing {job_id} using 5-layer semantic strategy")

        chunks = {
            "job_id": job_id,
            "processed_at": datetime.utcnow().isoformat(),
            "chunks": {
                "context_snapshot": None,
                "requirements_profile": None,
                "timeline_scope": None,
                "deliverables_portfolio": None,
                "feedback_outcomes": None
            },
            "summary": {
                "total_chunks": 0,
                "total_text_length": 0,
                "chunk_types": []
            }
        }

        # Process each chunk type
        
        # Layer 1: CONTEXT_SNAPSHOT
        context_chunk = ContextSnapshotChunker.extract(job_data)
        if context_chunk:
            chunks["chunks"]["context_snapshot"] = context_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += context_chunk["length"]
            chunks["summary"]["chunk_types"].append("context_snapshot")
            logger.debug(f"✓ Created CONTEXT_SNAPSHOT chunk for {job_id}")

        # Layer 2: REQUIREMENTS_PROFILE
        requirements_chunk = RequirementsProfileChunker.extract(job_data)
        if requirements_chunk:
            chunks["chunks"]["requirements_profile"] = requirements_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += requirements_chunk["length"]
            chunks["summary"]["chunk_types"].append("requirements_profile")
            logger.debug(f"✓ Created REQUIREMENTS_PROFILE chunk for {job_id}")

        # Layer 3: TIMELINE_SCOPE
        timeline_chunk = TimelineScopeChunker.extract(job_data)
        if timeline_chunk:
            chunks["chunks"]["timeline_scope"] = timeline_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += timeline_chunk["length"]
            chunks["summary"]["chunk_types"].append("timeline_scope")
            logger.debug(f"✓ Created TIMELINE_SCOPE chunk for {job_id}")

        # Layer 4: DELIVERABLES_PORTFOLIO
        portfolio_chunk = DeliverablesPortfolioChunker.extract(job_data)
        if portfolio_chunk:
            chunks["chunks"]["deliverables_portfolio"] = portfolio_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += portfolio_chunk["length"]
            chunks["summary"]["chunk_types"].append("deliverables_portfolio")
            logger.debug(f"✓ Created DELIVERABLES_PORTFOLIO chunk for {job_id}")

        # Layer 5: FEEDBACK_OUTCOMES
        feedback_chunk = FeedbackOutcomesChunker.extract(job_data)
        if feedback_chunk:
            chunks["chunks"]["feedback_outcomes"] = feedback_chunk
            chunks["summary"]["total_chunks"] += 1
            chunks["summary"]["total_text_length"] += feedback_chunk["length"]
            chunks["summary"]["chunk_types"].append("feedback_outcomes")
            logger.debug(f"✓ Created FEEDBACK_OUTCOMES chunk for {job_id}")

        logger.info(
            f"✓ [Chunking] {job_id}: {chunks['summary']['total_chunks']} chunks created, "
            f"{chunks['summary']['total_text_length']} chars total | "
            f"Types: {', '.join(chunks['summary']['chunk_types'])}"
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
        logger.info(f"[Chunking] Processing batch of {len(job_data_list)} jobs")
        processed_chunks = []
        
        for i, job_data in enumerate(job_data_list, 1):
            try:
                processed = self.process_job_data(job_data)
                processed_chunks.append(processed)
                logger.debug(f"[Batch {i}/{len(job_data_list)}] ✓ Processed successfully")
            except Exception as e:
                logger.error(f"[Batch {i}] ✗ Error processing: {str(e)}", exc_info=True)

        logger.info(f"✓ [Chunking] Batch complete: {len(processed_chunks)} successful")
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

    # Backward compatibility alias for DataChunker.chunk_training_data()
    chunk_training_data = get_all_chunks_flat

    def validate_chunk_json(self, chunk: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that a chunk is properly formatted and JSON-serializable.
        
        Args:
            chunk: Chunk dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            json.dumps(chunk)
            required_fields = ["chunk_type", "text", "metadata", "created_at"]
            
            for field in required_fields:
                if field not in chunk:
                    return False, f"Missing required field: {field}"
            
            if not isinstance(chunk["metadata"], dict):
                return False, "Metadata must be a dictionary"
            
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

