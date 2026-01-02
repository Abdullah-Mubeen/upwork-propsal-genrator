"""
Proposal Generation Routes

Single endpoint for intelligent proposal generation based on:
- New job description and skills
- Historical project data
- Portfolio links and feedback
- Previous successful proposals
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from app.utils.retrieval_pipeline import RetrievalPipeline
from app.utils.openai_service import OpenAIService
from app.utils.prompt_engine import PromptEngine
from app.db import get_db
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/proposals", tags=["proposals"])


# ===================== REQUEST/RESPONSE MODELS =====================

class GenerateProposalRequest(BaseModel):
    """Request to generate a proposal based on new job details"""
    # Required job information
    job_title: str = Field(..., description="Job position title")
    company_name: str = Field(..., description="Client company name")
    job_description: str = Field(..., description="Full job description")
    skills_required: List[str] = Field(..., description="Required skills")
    
    # Optional job details
    industry: Optional[str] = Field(None, description="Industry category (e.g., SaaS, FinTech, HealthTech)")
    task_type: Optional[str] = Field(None, description="Type of task (e.g., full_stack, backend, frontend, mobile)")
    estimated_budget: Optional[float] = Field(None, description="Estimated budget in USD")
    project_duration_days: Optional[int] = Field(None, description="Project duration in days")
    urgent_adhoc: Optional[bool] = Field(False, description="Is this urgent/adhoc project?")
    
    # Proposal customization
    proposal_style: str = Field("professional", description="Style: professional, casual, technical, creative, data_driven")
    tone: str = Field("confident", description="Tone: confident, humble, enthusiastic, analytical, friendly")
    max_word_count: int = Field(300, ge=100, le=1500, description="Target proposal length in words (default 300, recommended 200-350)")
    
    # Timeline options
    include_timeline: bool = Field(False, description="Include timeline in proposal? If false, no timeline section added")
    timeline_duration: Optional[str] = Field(None, description="Custom timeline duration (e.g., '2-3 weeks', '1 month'). Only used if include_timeline is True")
    
    # Historical data options
    similar_projects_count: int = Field(3, ge=1, le=10, description="Number of similar past projects to reference")
    include_previous_proposals: bool = Field(True, description="Include analysis of previous proposals for similar jobs?")
    include_portfolio: bool = Field(True, description="Include portfolio links from similar projects?")
    include_feedback: bool = Field(True, description="Include client feedback/testimonials from similar projects?")


class ProposalResponse(BaseModel):
    """Response with intelligently generated proposal"""
    success: bool
    job_title: str
    company_name: str
    generated_proposal: str
    word_count: int
    proposal_style: str
    proposal_tone: str
    
    # Historical context used
    similar_projects: List[Dict[str, Any]] = Field(description="Similar projects from history used for reference")
    previous_proposals_insights: Optional[Dict[str, Any]] = Field(None, description="Insights from previous proposals")
    portfolio_links_used: List[str] = Field(description="Portfolio links included in proposal")
    feedback_urls_used: List[str] = Field(description="Client feedback URLs included")
    
    # Retrieval insights
    insights: Optional[Dict[str, Any]] = Field(None, description="Success patterns and client values extracted from retrieval")
    
    # Quality metrics
    confidence_score: float = Field(description="Confidence score (0-1) for proposal quality")
    improvement_suggestions: Optional[List[str]] = Field(None, description="Suggestions for better proposals")
    
    metadata: Dict[str, Any] = Field(description="Additional metadata")


# ===================== MAIN ENDPOINT =====================

@router.post(
    "/generate",
    response_model=ProposalResponse,
    status_code=200,
    summary="Generate intelligent proposal from job details and history",
    responses={
        200: {"description": "Proposal generated successfully"},
        400: {"description": "Invalid job data"},
        500: {"description": "Generation error"}
    }
)
async def generate_proposal(request: GenerateProposalRequest):
    """
    Generate an intelligent AI proposal based on:
    
    1. **New Job Details** - Job title, description, skills, company
    2. **Historical Data** - Similar past projects from database
    3. **Previous Proposals** - What worked in similar situations
    4. **Portfolio & Feedback** - Links from successful similar projects
    5. **Style & Tone** - Customizable writing style and tone
    
    The AI will:
    - Find similar past projects
    - Analyze what made those proposals successful
    - Extract portfolio links and feedback URLs
    - Generate a new proposal leveraging all historical insights
    - Score proposal quality and suggest improvements
    
    **Request Example:**
    ```json
    {
        "job_title": "Full Stack Developer",
        "company_name": "TechStartup Inc",
        "job_description": "Build a modern web app with React and Node.js...",
        "skills_required": ["React", "Node.js", "MongoDB", "Docker"],
        "industry": "SaaS",
        "task_type": "full_stack",
        "proposal_style": "professional",
        "tone": "confident",
        "max_word_count": 500,
        "similar_projects_count": 3,
        "include_previous_proposals": true,
        "include_portfolio": true,
        "include_feedback": true
    }
    ```
    
    **Response Example:**
    ```json
    {
        "success": true,
        "job_title": "Full Stack Developer",
        "company_name": "TechStartup Inc",
        "generated_proposal": "Dear TechStartup Inc...",
        "word_count": 487,
        "proposal_style": "professional",
        "proposal_tone": "confident",
        "similar_projects": [...],
        "previous_proposals_insights": {...},
        "portfolio_links_used": ["https://..."],
        "feedback_urls_used": ["https://..."],
        "confidence_score": 0.92,
        "improvement_suggestions": [...],
        "metadata": {...}
    }
    ```
    """
    try:
        logger.info(f"[ProposalAPI] Generating proposal for {request.company_name} - {request.job_title}")
        
        # Initialize services
        db = get_db()
        openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
        retrieval_pipeline = RetrievalPipeline(db, openai_service)
        prompt_engine = PromptEngine()
        
        # Prepare job data
        job_data = {
            "job_title": request.job_title,
            "company_name": request.company_name,
            "job_description": request.job_description,
            "skills_required": request.skills_required,
            "industry": request.industry or "general",
            "task_type": request.task_type or "other"
        }
        
        # Step 1: Get historical jobs
        logger.info(f"[ProposalAPI] Step 1: Fetching historical job data...")
        all_jobs = list(db.db.training_data.find({}))
        
        if not all_jobs:
            logger.warning("No historical job data available - generating proposal with basic context")
        else:
            logger.info(f"[ProposalAPI] Found {len(all_jobs)} historical jobs")
        
        # Step 2: Find similar projects
        logger.info(f"[ProposalAPI] Step 2: Finding similar past projects...")
        retrieval_result = retrieval_pipeline.retrieve_for_proposal(
            job_data,
            all_jobs,
            top_k=request.similar_projects_count
        )
        
        similar_projects = retrieval_result.get("similar_projects", [])
        logger.info(f"[ProposalAPI] Found {len(similar_projects)} similar projects")
        
        # CRITICAL: Filter to only projects with actual portfolio URLs
        # Don't mention projects without portfolio proof to avoid suggesting fake credentials
        projects_with_portfolio = []
        for project in similar_projects:
            portfolio_urls = project.get("portfolio_urls", [])
            # Ensure portfolio_urls is a list
            if isinstance(portfolio_urls, str):
                portfolio_urls = [portfolio_urls] if portfolio_urls else []
            if isinstance(portfolio_urls, list):
                portfolio_urls = [url for url in portfolio_urls if url]  # Filter empty strings
            project["portfolio_urls"] = portfolio_urls
            
            # Only include projects with actual portfolio links
            if portfolio_urls:
                projects_with_portfolio.append(project)
                logger.debug(f"  ✓ Project {project.get('company')} has {len(portfolio_urls)} portfolio links")
            else:
                logger.debug(f"  ✗ Skipping project {project.get('company')} - no portfolio URLs")
        
        # Use filtered projects (with portfolio) for proposal generation
        if projects_with_portfolio:
            similar_projects = projects_with_portfolio
            logger.info(f"[ProposalAPI] Filtered to {len(similar_projects)} projects with portfolio links")
        else:
            logger.warning(f"[ProposalAPI] No projects with portfolio links found - using all similar projects")
        
        # Step 3: Analyze previous proposals for similar jobs
        logger.info(f"[ProposalAPI] Step 3: Analyzing previous proposals...")
        previous_proposals_insights = None
        portfolio_links_used = []
        feedback_urls_used = []
        
        if request.include_previous_proposals and similar_projects:
            previous_proposals = []
            for project in similar_projects[:request.similar_projects_count]:
                contract_id = project.get("contract_id")
                stored_proposal = db.db.proposals.find_one({
                    "contract_id": contract_id,
                    "generated_proposal": {"$exists": True}
                })
                
                if stored_proposal:
                    previous_proposals.append({
                        "contract_id": contract_id,
                        "company": project.get("company"),
                        "proposal_text": stored_proposal.get("generated_proposal"),
                        "style": stored_proposal.get("proposal_style"),
                        "tone": stored_proposal.get("proposal_tone"),
                        "word_count": stored_proposal.get("word_count")
                    })
            
            if previous_proposals:
                previous_proposals_insights = {
                    "total_previous_proposals": len(previous_proposals),
                    "common_patterns": _extract_proposal_patterns(previous_proposals),
                    "effective_phrases": _extract_effective_phrases(previous_proposals),
                    "average_word_count": sum(p["word_count"] for p in previous_proposals) / len(previous_proposals)
                }
                logger.info(f"[ProposalAPI] Analyzed {len(previous_proposals)} previous proposals")
        
        # Step 4: Collect portfolio and feedback URLs
        if request.include_portfolio or request.include_feedback:
            logger.info(f"[ProposalAPI] Step 4: Collecting portfolio and feedback URLs...")
            
            for project in similar_projects[:request.similar_projects_count]:
                # Extract portfolio URLs (plural - can be a list)
                if request.include_portfolio:
                    portfolio_urls = project.get("portfolio_urls", [])
                    if isinstance(portfolio_urls, list):
                        portfolio_links_used.extend(portfolio_urls)
                    elif isinstance(portfolio_urls, str):
                        portfolio_links_used.append(portfolio_urls)
                
                # Extract feedback URL (singular)
                if request.include_feedback and project.get("client_feedback_url"):
                    feedback_urls_used.append(project.get("client_feedback_url"))
            
            logger.info(f"[ProposalAPI] Collected {len(portfolio_links_used)} portfolio links, {len(feedback_urls_used)} feedback URLs")
        
        # Step 5: Build prompt with all context
        logger.info(f"[ProposalAPI] Step 5: Building enhanced prompt with historical context...")
        
        prompt = prompt_engine.build_proposal_prompt(
            job_data=job_data,
            similar_projects=similar_projects[:request.similar_projects_count],
            success_patterns=previous_proposals_insights.get("common_patterns", []) if previous_proposals_insights else [],
            style=request.proposal_style,
            tone=request.tone,
            max_words=request.max_word_count,
            include_portfolio=request.include_portfolio and bool(portfolio_links_used),
            include_feedback=request.include_feedback and bool(feedback_urls_used),
            include_timeline=request.include_timeline,
            timeline_duration=request.timeline_duration
        )
        
        # Step 6: Generate proposal
        logger.info(f"[ProposalAPI] Step 6: Generating proposal with AI...")
        
        # Calculate max_tokens based on target word count
        # Average: 1.33 tokens per word, add 20% buffer for complete output
        # For 300 words target: 300 * 1.33 * 1.2 ≈ 480 tokens, round up to 600 for safety
        max_tokens = int(request.max_word_count * 1.5) + 100  # Dynamic based on request
        
        proposal_text = openai_service.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        word_count = len(proposal_text.split())
        
        logger.info(f"[ProposalAPI] Generated {word_count} word proposal (target ~{request.max_word_count})")
        
        # Step 7: Score quality and suggest improvements
        logger.info(f"[ProposalAPI] Step 7: Scoring proposal quality...")
        
        references = {
            "portfolio_links_used": portfolio_links_used,
            "feedback_urls_used": feedback_urls_used,
            "projects_referenced": similar_projects[:request.similar_projects_count]
        }
        
        quality_score = prompt_engine.score_proposal_quality(proposal_text, job_data, references)
        
        improvement_suggestions = None
        if quality_score.get("overall_score", 1.0) < 0.85:
            improvement_suggestions = _generate_improvement_suggestions(quality_score)
        
        # NOTE: Generated proposals are only returned in response.
        # Only historical job data is saved to MongoDB and Pinecone.
        logger.info(f"✓ [ProposalAPI] Proposal generation complete!")
        
        # Return response
        return ProposalResponse(
            success=True,
            job_title=request.job_title,
            company_name=request.company_name,
            generated_proposal=proposal_text,
            word_count=word_count,
            proposal_style=request.proposal_style,
            proposal_tone=request.tone,
            similar_projects=similar_projects[:request.similar_projects_count],
            previous_proposals_insights=previous_proposals_insights,
            portfolio_links_used=portfolio_links_used,
            feedback_urls_used=feedback_urls_used,
            insights=retrieval_result.get("insights"),
            confidence_score=quality_score.get("overall_score", 0.85),
            improvement_suggestions=improvement_suggestions,
            metadata={
                "similar_projects_used": len(similar_projects[:request.similar_projects_count]),
                "previous_proposals_analyzed": len(previous_proposals_insights.get("common_patterns", [])) if previous_proposals_insights else 0,
                "quality_score_details": quality_score
            }
        )
        
    except Exception as e:
        logger.error(f"[ProposalAPI] Error generating proposal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate proposal: {str(e)}")


# ===================== HELPER FUNCTIONS =====================

def _extract_proposal_patterns(proposals: List[Dict[str, Any]]) -> List[str]:
    """Extract common patterns from previous proposals"""
    patterns = []
    
    for proposal in proposals:
        text = proposal.get("proposal_text", "").lower()
        
        if "understanding your needs" in text:
            patterns.append("Understanding client needs section")
        if "relevant experience" in text or "my experience" in text:
            patterns.append("Relevant experience section")
        if "approach" in text or "methodology" in text:
            patterns.append("Approach/methodology section")
        if "timeline" in text or "deliverables" in text:
            patterns.append("Timeline and deliverables section")
        if "investment" in text or "pricing" in text or "cost" in text:
            patterns.append("Investment/pricing discussion")
    
    return list(set(patterns))


def _extract_effective_phrases(proposals: List[Dict[str, Any]]) -> List[str]:
    """Extract effective phrases from previous proposals"""
    return [
        "I am confident in my ability to",
        "Based on my experience",
        "I understand the importance of",
        "With my expertise in",
        "I have successfully delivered"
    ]


def _generate_improvement_suggestions(quality_score: Dict[str, Any]) -> List[str]:
    """Generate improvement suggestions based on quality score"""
    suggestions = []
    
    if quality_score.get("length_score", 0) < 0.5:
        suggestions.append("Increase proposal length to better showcase your experience and approach")
    
    if quality_score.get("structure_score", 0) < 0.5:
        suggestions.append("Add more structure with clear sections: needs analysis, approach, timeline, deliverables")
    
    if quality_score.get("credibility_score", 0) < 0.5:
        suggestions.append("Include more specific examples, case studies, and portfolio references")
    
    if quality_score.get("reference_score", 0) < 0.5:
        suggestions.append("Add references to 2-3 similar successful projects")
    
    return suggestions
