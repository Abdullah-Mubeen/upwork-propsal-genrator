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


# ===================== ANALYSIS REQUEST/RESPONSE MODELS =====================

class AnalyzeJobRequest(BaseModel):
    """Request to analyze job and retrieve similar projects for selection"""
    job_title: str = Field(..., description="Job position title")
    company_name: str = Field(..., description="Client company name")
    job_description: str = Field(..., description="Full job description")
    skills_required: List[str] = Field(..., description="Required skills")
    industry: Optional[str] = Field(None, description="Industry category")
    task_type: Optional[str] = Field(None, description="Type of task")
    similar_projects_count: int = Field(5, ge=1, le=10, description="Number of similar projects to retrieve")


class PortfolioItem(BaseModel):
    """Individual portfolio item for selection"""
    url: str = Field(..., description="Portfolio URL")
    project_company: str = Field(..., description="Company name from source project")
    project_title: str = Field(..., description="Project title from source project")
    contract_id: str = Field(..., description="Source contract ID")


class FeedbackItem(BaseModel):
    """Individual feedback item for selection"""
    url: str = Field(..., description="Feedback URL")
    text: Optional[str] = Field(None, description="Feedback text preview")
    project_company: str = Field(..., description="Company name from source project")
    project_title: str = Field(..., description="Project title from source project")
    contract_id: str = Field(..., description="Source contract ID")
    satisfaction_score: Optional[float] = Field(None, description="Client satisfaction score (1-5)")


class ProposalTextItem(BaseModel):
    """Original winning proposal text from past projects"""
    proposal_text: str = Field(..., description="The original proposal text that won the job")
    project_company: str = Field(..., description="Company name from source project")
    project_title: str = Field(..., description="Project title from source project")
    contract_id: str = Field(..., description="Source contract ID")
    skills: List[str] = Field(default_factory=list, description="Skills used in this project")


class AnalyzeJobResponse(BaseModel):
    """Response with similar projects and selectable items"""
    success: bool
    job_title: str
    company_name: str
    
    # Similar projects found
    similar_projects: List[Dict[str, Any]] = Field(description="Similar projects with details")
    
    # Selectable items
    available_portfolio_items: List[PortfolioItem] = Field(description="Available portfolio URLs to select from")
    available_feedback_items: List[FeedbackItem] = Field(description="Available client feedback to select from")
    available_proposal_texts: List[ProposalTextItem] = Field(description="Available winning proposal texts to select from")
    
    # Insights
    insights: Optional[Dict[str, Any]] = Field(None, description="Success patterns extracted")
    
    metadata: Dict[str, Any] = Field(description="Analysis metadata")


class GenerateWithSelectionsRequest(BaseModel):
    """Request to generate proposal with user-selected portfolio and feedback"""
    # Required job information
    job_title: str = Field(..., description="Job position title")
    company_name: str = Field(..., description="Client company name")
    job_description: str = Field(..., description="Full job description")
    skills_required: List[str] = Field(..., description="Required skills")
    
    # Optional job details
    industry: Optional[str] = Field(None, description="Industry category")
    task_type: Optional[str] = Field(None, description="Type of task")
    
    # User-selected items (REQUIRED - user must select these)
    selected_portfolio_urls: List[str] = Field(default_factory=list, description="User-selected portfolio URLs to include exactly as-is")
    selected_feedback_items: List[Dict[str, Any]] = Field(default_factory=list, description="User-selected feedback items with url and text")
    selected_proposal_text: Optional[str] = Field(None, description="User-selected winning proposal text to use as base - will be included EXACTLY as-is")
    
    # Similar projects context (from analyze step)
    similar_projects: List[Dict[str, Any]] = Field(default_factory=list, description="Similar projects from analysis step")
    
    # Proposal customization
    proposal_style: str = Field("professional", description="Style: professional, casual, technical, creative")
    tone: str = Field("confident", description="Tone: confident, humble, enthusiastic, analytical")
    max_word_count: int = Field(300, ge=100, le=1500, description="Target proposal length")
    
    # Timeline options
    include_timeline: bool = Field(False, description="Include timeline in proposal?")
    timeline_duration: Optional[str] = Field(None, description="Custom timeline duration")


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
        
        # Initialize Pinecone service for semantic search
        from app.utils.pinecone_service import PineconeService
        pinecone_service = PineconeService(api_key=settings.PINECONE_API_KEY)
        
        retrieval_pipeline = RetrievalPipeline(db, pinecone_service)
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


# ===================== ANALYZE ENDPOINT =====================

@router.post(
    "/analyze",
    response_model=AnalyzeJobResponse,
    status_code=200,
    summary="Analyze job and retrieve similar projects for portfolio/feedback selection",
    responses={
        200: {"description": "Analysis completed with selectable items"},
        400: {"description": "Invalid job data"},
        500: {"description": "Analysis error"}
    }
)
async def analyze_job(request: AnalyzeJobRequest):
    """
    Analyze a job description and retrieve similar past projects.
    
    Returns portfolio URLs and client feedback from similar projects,
    allowing the user to SELECT which ones to include in the final proposal.
    
    **Use Case:**
    1. User provides job details
    2. System finds similar past projects
    3. Returns available portfolio links and feedback for user selection
    4. User selects desired items
    5. User calls /generate-with-selections with chosen items
    """
    try:
        logger.info(f"[AnalyzeAPI] Analyzing job: {request.company_name} - {request.job_title}")
        
        # Initialize services
        db = get_db()
        
        # Initialize Pinecone service for semantic search
        from app.utils.pinecone_service import PineconeService
        pinecone_service = PineconeService(api_key=settings.PINECONE_API_KEY)
        
        retrieval_pipeline = RetrievalPipeline(db, pinecone_service)
        
        # Prepare job data
        job_data = {
            "job_title": request.job_title,
            "company_name": request.company_name,
            "job_description": request.job_description,
            "skills_required": request.skills_required,
            "industry": request.industry or "general",
            "task_type": request.task_type or "other"
        }
        
        # Get historical jobs
        all_jobs = list(db.db.training_data.find({}))
        
        if not all_jobs:
            logger.warning("No historical job data available")
            return AnalyzeJobResponse(
                success=True,
                job_title=request.job_title,
                company_name=request.company_name,
                similar_projects=[],
                available_portfolio_items=[],
                available_feedback_items=[],
                insights=None,
                metadata={"message": "No historical data available"}
            )
        
        logger.info(f"[AnalyzeAPI] Found {len(all_jobs)} historical jobs")
        
        # Find similar projects
        retrieval_result = retrieval_pipeline.retrieve_for_proposal(
            job_data,
            all_jobs,
            top_k=request.similar_projects_count
        )
        
        similar_projects = retrieval_result.get("similar_projects", [])
        logger.info(f"[AnalyzeAPI] Found {len(similar_projects)} similar projects")
        
        # Extract available portfolio items
        available_portfolio_items = []
        available_feedback_items = []
        seen_portfolio_urls = set()
        seen_feedback_urls = set()
        
        for project in similar_projects:
            contract_id = project.get("contract_id", "")
            company = project.get("company", "Unknown")
            title = project.get("title", "Unknown Project")
            
            # Get full project data from training_data for additional details
            full_project = db.db.training_data.find_one({"contract_id": contract_id}) or {}
            
            # Extract portfolio URLs
            portfolio_urls = project.get("portfolio_urls", [])
            if isinstance(portfolio_urls, str):
                portfolio_urls = [portfolio_urls] if portfolio_urls else []
            
            for url in portfolio_urls:
                if url and url not in seen_portfolio_urls:
                    seen_portfolio_urls.add(url)
                    available_portfolio_items.append(PortfolioItem(
                        url=url,
                        project_company=company,
                        project_title=title,
                        contract_id=contract_id
                    ))
            
            # Extract feedback URL
            feedback_url = project.get("client_feedback_url")
            if feedback_url and feedback_url not in seen_feedback_urls:
                seen_feedback_urls.add(feedback_url)
                feedback_text = full_project.get("client_feedback_text", "")
                satisfaction = project.get("satisfaction") or full_project.get("client_satisfaction")
                
                available_feedback_items.append(FeedbackItem(
                    url=feedback_url,
                    text=feedback_text[:200] + "..." if feedback_text and len(feedback_text) > 200 else feedback_text,
                    project_company=company,
                    project_title=title,
                    contract_id=contract_id,
                    satisfaction_score=satisfaction
                ))
        
        # Extract available proposal texts from similar projects
        available_proposal_texts = []
        seen_contract_ids = set()
        
        for project in similar_projects:
            contract_id = project.get("contract_id", "")
            if contract_id in seen_contract_ids:
                continue
            seen_contract_ids.add(contract_id)
            
            company = project.get("company", "Unknown")
            title = project.get("title", "Unknown Project")
            skills = project.get("skills", [])
            
            # Get full project data from training_data for the proposal text
            full_project = db.db.training_data.find_one({"contract_id": contract_id}) or {}
            proposal_text = full_project.get("your_proposal_text", "")
            
            if proposal_text and proposal_text.strip():
                available_proposal_texts.append(ProposalTextItem(
                    proposal_text=proposal_text,
                    project_company=company,
                    project_title=title,
                    contract_id=contract_id,
                    skills=skills if isinstance(skills, list) else []
                ))
        
        logger.info(f"[AnalyzeAPI] Extracted {len(available_portfolio_items)} portfolio items, {len(available_feedback_items)} feedback items, {len(available_proposal_texts)} proposal texts")
        
        return AnalyzeJobResponse(
            success=True,
            job_title=request.job_title,
            company_name=request.company_name,
            similar_projects=similar_projects,
            available_portfolio_items=available_portfolio_items,
            available_feedback_items=available_feedback_items,
            available_proposal_texts=available_proposal_texts,
            insights=retrieval_result.get("insights"),
            metadata={
                "total_historical_jobs": len(all_jobs),
                "similar_projects_found": len(similar_projects),
                "portfolio_items_available": len(available_portfolio_items),
                "feedback_items_available": len(available_feedback_items),
                "proposal_texts_available": len(available_proposal_texts)
            }
        )
        
    except Exception as e:
        logger.error(f"[AnalyzeAPI] Error analyzing job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze job: {str(e)}")


# ===================== GENERATE WITH SELECTIONS ENDPOINT =====================

@router.post(
    "/generate-with-selections",
    response_model=ProposalResponse,
    status_code=200,
    summary="Generate proposal with user-selected portfolio and feedback",
    responses={
        200: {"description": "Proposal generated with selected items"},
        400: {"description": "Invalid request"},
        500: {"description": "Generation error"}
    }
)
async def generate_proposal_with_selections(request: GenerateWithSelectionsRequest):
    """
    Generate/compile a proposal using USER-SELECTED portfolio URLs, feedback, and proposal text.
    
    The selected items will be included EXACTLY as provided - no modifications.
    
    If a proposal text is selected, it will be used AS-IS with portfolio/feedback appended.
    
    **Workflow:**
    1. First call /analyze to get available items
    2. User selects which portfolio URLs, feedback, and proposal text to include
    3. Call this endpoint with selections
    4. If proposal text selected: Use it EXACTLY + append portfolio/feedback
    5. If no proposal text: Generate new proposal with selected items
    """
    try:
        logger.info(f"[GenerateWithSelectionsAPI] Generating proposal for {request.company_name} - {request.job_title}")
        logger.info(f"[GenerateWithSelectionsAPI] User selected {len(request.selected_portfolio_urls)} portfolio URLs, {len(request.selected_feedback_items)} feedback items")
        logger.info(f"[GenerateWithSelectionsAPI] Selected proposal text: {'Yes' if request.selected_proposal_text else 'No'}")
        
        # Initialize services
        db = get_db()
        openai_service = OpenAIService(api_key=settings.OPENAI_API_KEY)
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
        
        # Use the similar projects from the analysis step (if provided)
        similar_projects = request.similar_projects if request.similar_projects else []
        
        # User-selected items - use EXACTLY as provided
        portfolio_links_used = request.selected_portfolio_urls
        feedback_urls_used = [item.get("url", "") for item in request.selected_feedback_items if item.get("url")]
        
        # Check if user selected an existing proposal text
        if request.selected_proposal_text:
            # USE THE SELECTED PROPOSAL EXACTLY AS-IS
            # Compile: proposal text + portfolio URLs + feedback URLs
            logger.info(f"[GenerateWithSelectionsAPI] Using user-selected proposal text EXACTLY as-is")
            
            proposal_text = request.selected_proposal_text.strip()
            
            # Append portfolio URLs if selected
            if portfolio_links_used:
                proposal_text += "\n\n---\nPortfolio:"
                for url in portfolio_links_used:
                    proposal_text += f"\n{url}"
            
            # Append feedback URLs if selected
            if feedback_urls_used:
                proposal_text += "\n\nClient Feedback:"
                for url in feedback_urls_used:
                    proposal_text += f"\n{url}"
        else:
            # Generate new proposal with AI using selected items
            # Extract feedback texts for prompt context
            selected_feedback_texts = []
            for item in request.selected_feedback_items:
                if item.get("text"):
                    selected_feedback_texts.append({
                        "url": item.get("url", ""),
                        "text": item.get("text", ""),
                        "company": item.get("project_company", "")
                    })
            
            # Build prompt with EXACT user selections
            prompt = prompt_engine.build_proposal_prompt_with_selections(
                job_data=job_data,
                similar_projects=similar_projects,
                selected_portfolio_urls=portfolio_links_used,
                selected_feedback_items=selected_feedback_texts,
                style=request.proposal_style,
                tone=request.tone,
                max_words=request.max_word_count,
                include_timeline=request.include_timeline,
                timeline_duration=request.timeline_duration
            )
            
            # Generate proposal
            max_tokens = int(request.max_word_count * 1.5) + 100
            
            proposal_text = openai_service.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
        
        word_count = len(proposal_text.split())
        
        logger.info(f"[GenerateWithSelectionsAPI] Final proposal has {word_count} words")
        
        # Score quality
        references = {
            "portfolio_links_used": portfolio_links_used,
            "feedback_urls_used": feedback_urls_used,
            "projects_referenced": similar_projects
        }
        
        quality_score = prompt_engine.score_proposal_quality(proposal_text, job_data, references)
        
        improvement_suggestions = None
        if quality_score.get("overall_score", 1.0) < 0.85:
            improvement_suggestions = _generate_improvement_suggestions(quality_score)
        
        logger.info(f"✓ [GenerateWithSelectionsAPI] Proposal generation complete!")
        
        return ProposalResponse(
            success=True,
            job_title=request.job_title,
            company_name=request.company_name,
            generated_proposal=proposal_text,
            word_count=word_count,
            proposal_style=request.proposal_style,
            proposal_tone=request.tone,
            similar_projects=similar_projects,
            previous_proposals_insights=None,
            portfolio_links_used=portfolio_links_used,
            feedback_urls_used=feedback_urls_used,
            insights=None,
            confidence_score=quality_score.get("overall_score", 0.85),
            improvement_suggestions=improvement_suggestions,
            metadata={
                "similar_projects_used": len(similar_projects),
                "user_selected_portfolio_count": len(portfolio_links_used),
                "user_selected_feedback_count": len(feedback_urls_used),
                "quality_score_details": quality_score
            }
        )
        
    except Exception as e:
        logger.error(f"[GenerateWithSelectionsAPI] Error generating proposal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate proposal: {str(e)}")
