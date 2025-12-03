"""
Smart Data Chunking Strategy for AI Training

Implements intelligent chunking for different data types:
- Metadata chunks (highest priority for filtering)
- Proposal text chunks (target similarity)
- Job description chunks (context)
- Feedback chunks (validation)
- Combined summary chunks
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ChunkType:
    """Chunk type constants with priority scores"""
    METADATA = "metadata"           # Priority: 2.0 - Structured info
    PROPOSAL = "proposal"           # Priority: 1.8 - What we generate
    DESCRIPTION = "description"     # Priority: 1.6 - Job context
    FEEDBACK = "feedback"           # Priority: 1.2 - Validation signals
    SUMMARY = "summary"             # Priority: 1.4 - Combined overview


class DataChunker:
    """
    Intelligent data chunker for proposal training data
    
    Chunking Strategy:
    1. Extract metadata (company, skills, title) - HIGHEST PRIORITY
    2. Chunk proposal text (previous proposals) - VERY HIGH
    3. Chunk job description (job context) - HIGH
    4. Chunk feedback (if available) - MEDIUM
    5. Create summary chunk (overview) - MEDIUM
    
    Each chunk includes:
    - Content text
    - Chunk type and priority score
    - Metadata for filtering (industry, skills, company)
    - Length information
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2048
    ):
        """
        Initialize chunker with size parameters
        
        Args:
            chunk_size: Target size in characters for chunks
            chunk_overlap: Overlap between consecutive chunks for context
            min_chunk_size: Minimum chunk size to keep
            max_chunk_size: Maximum chunk size allowed
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_training_data(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligently chunk complete training data into multiple chunk types
        
        Args:
            job_data: Complete job data containing all fields
            
        Returns:
            List of chunks, each with content, type, priority, and metadata
        """
        chunks = []
        contract_id = job_data.get("contract_id", "unknown")
        
        logger.info(f"Starting smart chunking for contract: {contract_id}")
        
        # 1. METADATA CHUNK - Highest Priority (2.0)
        # Structured info for exact matching and filtering
        metadata_chunk = self._create_metadata_chunk(job_data)
        if metadata_chunk:
            chunks.append(metadata_chunk)
            logger.debug(f"Created metadata chunk for {contract_id}")
        
        # 2. PROPOSAL TEXT CHUNKS - Very High Priority (1.8)
        # This is exactly what we're trying to generate
        proposal_chunks = self._chunk_proposal_text(job_data)
        chunks.extend(proposal_chunks)
        logger.debug(f"Created {len(proposal_chunks)} proposal chunks")
        
        # 3. JOB DESCRIPTION CHUNKS - High Priority (1.6)
        # Context of the original job
        description_chunks = self._chunk_job_description(job_data)
        chunks.extend(description_chunks)
        logger.debug(f"Created {len(description_chunks)} description chunks")
        
        # 4. FEEDBACK CHUNKS - Medium Priority (1.2)
        # Client feedback and validation signals
        feedback_chunks = self._chunk_feedback(job_data)
        chunks.extend(feedback_chunks)
        logger.debug(f"Created {len(feedback_chunks)} feedback chunks")
        
        # 5. COMBINED SUMMARY CHUNK - Medium Priority (1.4)
        # Quick reference combining key elements
        summary_chunk = self._create_summary_chunk(job_data)
        if summary_chunk:
            chunks.append(summary_chunk)
            logger.debug(f"Created summary chunk for {contract_id}")
        
        logger.info(f"✓ Created {len(chunks)} chunks for contract {contract_id}")
        return chunks
    
    def _create_metadata_chunk(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata chunk with structured information
        
        Highest priority for exact matching and filtering by skills, industry, etc.
        """
        company = job_data.get("company_name", "")
        title = job_data.get("job_title", "")
        industry = job_data.get("industry", "general")
        skills = job_data.get("skills_required", [])
        portfolio = job_data.get("portfolio_url", "")
        status = job_data.get("project_status", "completed")
        
        content = f"""
Company: {company}
Job Title: {title}
Industry: {industry}
Skills: {', '.join(skills) if skills else 'N/A'}
Portfolio: {portfolio if portfolio else 'N/A'}
Project Status: {status}
"""
        
        if len(content.strip()) >= self.min_chunk_size:
            return {
                "type": ChunkType.METADATA,
                "content": content.strip(),
                "priority": 2.0,  # HIGHEST - For structural matching
                "length": len(content),
                "company_name": company,
                "job_title": title,
                "industry": industry,
                "skills_required": skills,
                "project_status": status,
                "portfolio_url": portfolio
            }
        
        return None
    
    def _chunk_proposal_text(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk the proposal text - the target for similarity search
        
        Since proposals are exactly what we generate, they're most relevant
        for semantic similarity matching.
        """
        proposal_text = job_data.get("your_proposal_text", "")
        
        if not proposal_text or len(proposal_text) < self.min_chunk_size:
            return []
        
        chunks = []
        
        # If proposal is short, keep as single chunk
        if len(proposal_text) <= self.max_chunk_size:
            chunks.append({
                "type": ChunkType.PROPOSAL,
                "content": proposal_text,
                "priority": 1.8,  # VERY HIGH
                "length": len(proposal_text),
                "company_name": job_data.get("company_name", ""),
                "industry": job_data.get("industry", "general"),
                "skills_required": job_data.get("skills_required", []),
                "project_status": job_data.get("project_status", "completed"),
                "portfolio_url": job_data.get("portfolio_url", "")
            })
        else:
            # Break into sentence-aware chunks with overlap
            text_chunks = self._smart_chunk_text(proposal_text, overlap_ratio=0.2)
            for i, chunk_content in enumerate(text_chunks):
                chunks.append({
                    "type": ChunkType.PROPOSAL,
                    "content": chunk_content,
                    "priority": 1.8 - (i * 0.05),  # Slightly lower for later chunks
                    "length": len(chunk_content),
                    "company_name": job_data.get("company_name", ""),
                    "industry": job_data.get("industry", "general"),
                    "skills_required": job_data.get("skills_required", []),
                    "project_status": job_data.get("project_status", "completed"),
                    "portfolio_url": job_data.get("portfolio_url", "")
                })
        
        return chunks
    
    def _chunk_job_description(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk job description - provides context for the opportunity
        """
        job_desc = job_data.get("job_description", "")
        
        if not job_desc or len(job_desc) < self.min_chunk_size:
            return []
        
        chunks = []
        
        if len(job_desc) <= self.max_chunk_size:
            chunks.append({
                "type": ChunkType.DESCRIPTION,
                "content": job_desc,
                "priority": 1.6,  # HIGH
                "length": len(job_desc),
                "company_name": job_data.get("company_name", ""),
                "industry": job_data.get("industry", "general"),
                "skills_required": job_data.get("skills_required", []),
                "project_status": job_data.get("project_status", "completed"),
                "portfolio_url": job_data.get("portfolio_url", "")
            })
        else:
            text_chunks = self._smart_chunk_text(job_desc, overlap_ratio=0.15)
            for i, chunk_content in enumerate(text_chunks):
                chunks.append({
                    "type": ChunkType.DESCRIPTION,
                    "content": chunk_content,
                    "priority": 1.6 - (i * 0.03),
                    "length": len(chunk_content),
                    "company_name": job_data.get("company_name", ""),
                    "industry": job_data.get("industry", "general"),
                    "skills_required": job_data.get("skills_required", []),
                    "project_status": job_data.get("project_status", "completed"),
                    "portfolio_url": job_data.get("portfolio_url", "")
                })
        
        return chunks
    
    def _chunk_feedback(self, job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk client feedback - validation and quality indicators
        
        Feedback can be:
        - Text feedback (direct user input)
        - Extracted text from image OCR (by OpenAI Vision)
        """
        feedback = job_data.get("client_feedback", "")
        
        if not feedback or len(feedback) < self.min_chunk_size:
            return []
        
        chunks = []
        
        if len(feedback) <= self.max_chunk_size:
            chunks.append({
                "type": ChunkType.FEEDBACK,
                "content": f"Client Feedback:\n{feedback}",
                "priority": 1.2,  # MEDIUM
                "length": len(feedback),
                "company_name": job_data.get("company_name", ""),
                "industry": job_data.get("industry", "general"),
                "skills_required": job_data.get("skills_required", []),
                "project_status": job_data.get("project_status", "completed"),
                "portfolio_url": job_data.get("portfolio_url", "")
            })
        else:
            text_chunks = self._smart_chunk_text(feedback, overlap_ratio=0.1)
            for i, chunk_content in enumerate(text_chunks):
                chunks.append({
                    "type": ChunkType.FEEDBACK,
                    "content": chunk_content,
                    "priority": 1.2 - (i * 0.02),
                    "length": len(chunk_content),
                    "company_name": job_data.get("company_name", ""),
                    "industry": job_data.get("industry", "general"),
                    "skills_required": job_data.get("skills_required", []),
                    "project_status": job_data.get("project_status", "completed"),
                    "portfolio_url": job_data.get("portfolio_url", "")
                })
        
        return chunks
    
    def _create_summary_chunk(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a combined summary chunk from key fields
        
        This provides a high-level overview for quick semantic matching
        """
        company = job_data.get("company_name", "")
        title = job_data.get("job_title", "")
        desc_snippet = job_data.get("job_description", "")[:300]
        proposal_snippet = job_data.get("your_proposal_text", "")[:300]
        skills = ", ".join(job_data.get("skills_required", []))
        
        summary = f"""
Project Summary: {company} - {title}

Industry: {job_data.get('industry', 'general')}
Skills: {skills}

Job Overview:
{desc_snippet}

Proposal Highlights:
{proposal_snippet}
"""
        
        if len(summary.strip()) >= self.min_chunk_size:
            return {
                "type": ChunkType.SUMMARY,
                "content": summary.strip(),
                "priority": 1.4,  # MEDIUM
                "length": len(summary),
                "company_name": company,
                "industry": job_data.get("industry", "general"),
                "skills_required": job_data.get("skills_required", []),
                "project_status": job_data.get("project_status", "completed"),
                "portfolio_url": job_data.get("portfolio_url", "")
            }
        
        return None
    
    def _smart_chunk_text(self, text: str, overlap_ratio: float = 0.2) -> List[str]:
        """
        Intelligently chunk text while preserving meaning
        
        Strategy:
        1. Break on sentence boundaries when possible
        2. Maintain context overlap between chunks
        3. Respect min/max chunk size constraints
        
        Args:
            text: Text to chunk
            overlap_ratio: Ratio of overlap between chunks (0.0-1.0)
            
        Returns:
            List of chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        overlap_chars = int(self.chunk_size * overlap_ratio)
        
        # Split by sentences (handle various sentence endings)
        sentences = text.replace("!\n", ".|").replace("?\n", ".|").split(". ")
        
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Add period back if needed
            sentence_with_period = sentence if sentence.endswith((".", "!", "?")) else sentence + ". "
            
            # If adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence_with_period) > self.chunk_size:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    # Create overlap from end of previous chunk
                    current_chunk = current_chunk[-overlap_chars:] if overlap_chars > 0 else ""
            
            current_chunk += sentence_with_period
        
        # Add remaining chunk if valid
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Fallback: return original text if nothing worked
        return chunks if chunks else [text]
