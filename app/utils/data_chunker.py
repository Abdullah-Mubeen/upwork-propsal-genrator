"""
Smart Data Chunking Strategy for AI Training (v2)

NEW STRATEGY:
- JOB_FACTS_CHUNK: One per job, cleaned description + metadata
- PROPOSAL_CHUNK: One per proposal, never split
- FEEDBACK_CHUNK: One per review, no ratings/dates/labels
- TEMPLATE_CHUNK: Internal writing templates

This module wraps the AdvancedChunker for optimal RAG performance.
Backward compatibility maintained with chunk_training_data() method.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Import the new advanced chunker (5-layer semantic strategy)
from app.utils.advanced_chunker import (
    AdvancedChunkProcessor,
    ChunkTypeEnum,
    ContextSnapshotChunker,
    RequirementsProfileChunker,
    TimelineScopeChunker,
    DeliverablesPortfolioChunker,
    FeedbackOutcomesChunker
)


class ChunkType:
    """Chunk type constants (legacy, kept for compatibility)"""
    METADATA = "job_facts"          # Now mapped to job_facts
    PROPOSAL = "proposal"           # Unchanged
    DESCRIPTION = "job_facts"       # Mapped to job_facts
    FEEDBACK = "feedback"           # Unchanged
    SUMMARY = "job_facts"           # Mapped to job_facts


class DataChunker:
    """
    Wrapper around AdvancedChunkProcessor for optimal RAG performance.
    
    Uses the 4-chunk strategy:
    1. JOB_FACTS_CHUNK - Clean job description with strong metadata
    2. PROPOSAL_CHUNK - Full, unbroken proposal text
    3. FEEDBACK_CHUNK - Pure feedback without noise
    4. TEMPLATE_CHUNK - Writing templates and patterns
    
    All chunks are JSON-ready with structured metadata for filtering and retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2048
    ):
        """
        Initialize chunker with size parameters (kept for backward compatibility).
        
        Args:
            chunk_size: Target size (legacy parameter, not used in new chunker)
            chunk_overlap: Overlap (legacy parameter, not used in new chunker)
            min_chunk_size: Minimum size (legacy parameter, not used in new chunker)
            max_chunk_size: Maximum size (legacy parameter, not used in new chunker)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Initialize the advanced processor
        self.processor = AdvancedChunkProcessor()
        logger.info("DataChunker initialized with AdvancedChunkProcessor (v2)")
    
    def chunk_training_data(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligently chunk complete training data using the 4-chunk strategy.
        
        This is the main entry point that maintains backward compatibility.
        
        Args:
            job_data: Complete job data containing all fields
            
        Returns:
            List of chunks, each JSON-ready with content, type, and metadata
        """
        return self.processor.get_all_chunks_flat(job_data)
    
    def _create_metadata_chunk(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility"""
        return ContextSnapshotChunker.extract(job_data)
    
    def _chunk_proposal_text(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy method - kept for backward compatibility"""
        # Not part of new 5-layer strategy, but provide fallback
        chunk = RequirementsProfileChunker.extract(job_data)
        return [chunk] if chunk else []
    
    def _chunk_job_description(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy method - kept for backward compatibility"""
        chunk = RequirementsProfileChunker.extract(job_data)
        return [chunk] if chunk else []
    
    def _chunk_feedback(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy method - kept for backward compatibility"""
        chunk = FeedbackOutcomesChunker.extract(job_data)
        return [chunk] if chunk else []
    
    def _create_summary_chunk(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility"""
        return ContextSnapshotChunker.extract(job_data)
    
    def _smart_chunk_text(self, text: str, overlap_ratio: float = 0.2) -> List[str]:
        """Legacy method - kept for backward compatibility"""
        # Simple fallback for legacy code
        if len(text) <= self.max_chunk_size:
            return [text]
        return [text]  # Return as-is, don't split
