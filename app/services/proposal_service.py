"""
Proposal Service

Business logic for proposal generation, including:
- Retrieval of similar projects
- Analysis of previous proposals
- Proposal generation with AI
- Quality scoring and suggestions
"""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProposalRequest:
    """Input parameters for proposal generation."""
    job_title: str
    company_name: str
    job_description: str
    skills_required: List[str]
    industry: Optional[str] = "general"
    task_type: Optional[str] = "other"
    estimated_budget: Optional[float] = None
    project_duration_days: Optional[int] = None
    urgent_adhoc: bool = False
    proposal_style: str = "professional"
    tone: str = "confident"
    max_word_count: int = 300
    include_timeline: bool = False
    timeline_duration: Optional[str] = None
    similar_projects_count: int = 3
    include_previous_proposals: bool = True
    include_portfolio: bool = True
    include_feedback: bool = True
    # Multi-tenant fields (optional - for org-scoped queries)
    org_id: Optional[str] = None
    profile_id: Optional[str] = None


@dataclass
class ProposalResult:
    """Result of proposal generation."""
    success: bool
    generated_proposal: str = ""
    word_count: int = 0
    similar_projects: List[Dict[str, Any]] = field(default_factory=list)
    previous_proposals_insights: Optional[Dict[str, Any]] = None
    portfolio_links_used: List[str] = field(default_factory=list)
    feedback_urls_used: List[str] = field(default_factory=list)
    insights: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    improvement_suggestions: Optional[List[str]] = None
    error_message: Optional[str] = None


class ProposalService:
    """
    Service for intelligent proposal generation.
    
    Orchestrates:
    - JobRequirementsService for deep job understanding (NEW)
    - RetrievalPipeline for finding similar projects
    - PromptEngine for building prompts
    - OpenAI for text generation
    """
    
    def __init__(
        self,
        db_manager=None,
        openai_service=None,
        retrieval_pipeline=None,
        prompt_engine=None,
        job_requirements_service=None
    ):
        """
        Initialize with dependencies.
        
        Args:
            db_manager: Database manager instance
            openai_service: OpenAI service for text generation
            retrieval_pipeline: Pipeline for finding similar projects
            prompt_engine: Engine for building prompts
            job_requirements_service: Service for extracting structured job requirements (NEW)
        """
        self.db = db_manager
        self.openai_service = openai_service
        self.retrieval_pipeline = retrieval_pipeline
        self.prompt_engine = prompt_engine
        self.job_requirements_service = job_requirements_service
    
    def generate_proposal(self, request: ProposalRequest) -> ProposalResult:
        """
        Generate an intelligent proposal based on job details and history.
        
        Steps:
        1. Get historical jobs
        2. Find similar projects via semantic search
        3. Analyze previous proposals
        4. Collect portfolio and feedback URLs
        5. Build enhanced prompt
        6. Generate proposal with AI
        7. Score quality and suggest improvements
        
        Args:
            request: ProposalRequest with job details and options
            
        Returns:
            ProposalResult with generated proposal and metadata
        """
        try:
            logger.info(f"[ProposalService] Generating proposal for {request.company_name} - {request.job_title}")
            
            # Prepare job data
            job_data = self._prepare_job_data(request)
            
            # Step 0: Extract structured job requirements (NEW - Issue #23)
            job_requirements = self._extract_job_requirements(request)
            if job_requirements:
                logger.info(f"[ProposalService] Extracted requirements: task='{job_requirements.exact_task[:50]}...' "
                           f"tone={job_requirements.client_tone}, confidence={job_requirements.extraction_confidence:.2f}")
                # Enhance job_data with extracted requirements
                job_data = self._enhance_job_data_with_requirements(job_data, job_requirements)
            
            # Step 1: Get historical jobs (org-scoped if org_id provided)
            all_jobs = self._get_historical_jobs(request.org_id)
            
            # Step 2: Find similar projects (ENHANCED with job_requirements - Issue #24)
            retrieval_result = self._find_similar_projects(
                job_data, all_jobs, request.similar_projects_count,
                job_requirements=job_requirements
            )
            similar_projects = retrieval_result.get("similar_projects", [])
            
            # Filter to projects with portfolio
            similar_projects = self._filter_projects_with_portfolio(similar_projects)
            
            # Step 3: Analyze previous proposals
            previous_insights = None
            if request.include_previous_proposals and similar_projects:
                previous_insights = self._analyze_previous_proposals(
                    similar_projects[:request.similar_projects_count]
                )
            
            # Step 4: Collect portfolio and feedback URLs
            portfolio_links, feedback_urls = self._collect_urls(
                similar_projects[:request.similar_projects_count],
                request.include_portfolio,
                request.include_feedback
            )
            
            # Step 4.5: Get freelancer profile context if available
            profile_context = None
            if request.profile_id and request.org_id:
                profile_context = self._get_freelancer_profile(request.org_id, request.profile_id)
            
            # Step 5: Build prompt (now with job requirements context)
            prompt = self._build_prompt(
                job_data=job_data,
                similar_projects=similar_projects[:request.similar_projects_count],
                previous_insights=previous_insights,
                portfolio_links=portfolio_links,
                feedback_urls=feedback_urls,
                request=request,
                profile_context=profile_context,
                job_requirements=job_requirements if 'job_requirements' in dir() else None
            )
            
            # Step 6: Generate proposal
            proposal_text, word_count = self._generate_with_ai(prompt, request.max_word_count)
            
            # Step 7: Score quality
            confidence_score, suggestions = self._score_proposal(
                proposal_text=proposal_text,
                portfolio_links=portfolio_links,
                feedback_urls=feedback_urls,
                similar_projects=similar_projects,
                request=request
            )
            
            return ProposalResult(
                success=True,
                generated_proposal=proposal_text,
                word_count=word_count,
                similar_projects=similar_projects[:request.similar_projects_count],
                previous_proposals_insights=previous_insights,
                portfolio_links_used=portfolio_links,
                feedback_urls_used=feedback_urls,
                insights=retrieval_result.get("insights"),
                confidence_score=confidence_score,
                improvement_suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"[ProposalService] Error generating proposal: {e}")
            return ProposalResult(
                success=False,
                error_message=str(e)
            )
    
    def _prepare_job_data(self, request: ProposalRequest) -> Dict[str, Any]:
        """Prepare job data dict from request."""
        job_data = {
            "job_title": request.job_title,
            "company_name": request.company_name,
            "job_description": request.job_description,
            "skills_required": request.skills_required,
            "industry": request.industry or "general",
            "task_type": request.task_type or "other"
        }
        
        # Enhance industry detection if needed
        if not request.industry or request.industry == "general":
            job_data["industry"] = self._detect_industry(job_data)
        
        return job_data
    
    def _extract_job_requirements(self, request: ProposalRequest) -> Optional[Any]:
        """
        Extract structured job requirements using JobRequirementsService.
        
        This is the NEW job understanding layer that deeply analyzes the job
        description BEFORE retrieval and generation.
        
        Args:
            request: ProposalRequest with job details
            
        Returns:
            JobRequirements dataclass or None if service unavailable
        """
        if not self.job_requirements_service:
            logger.debug("[ProposalService] JobRequirementsService not available, skipping extraction")
            return None
        
        try:
            from app.services.job_requirements_service import JobRequirements
            
            requirements = self.job_requirements_service.extract_job_requirements(
                job_description=request.job_description,
                job_title=request.job_title,
                skills_required=request.skills_required
            )
            
            return requirements
            
        except Exception as e:
            logger.warning(f"[ProposalService] Job requirements extraction failed: {e}")
            return None
    
    def _enhance_job_data_with_requirements(
        self,
        job_data: Dict[str, Any],
        requirements: Any
    ) -> Dict[str, Any]:
        """
        Enhance job_data with extracted requirements for better retrieval.
        
        This merges the LLM-extracted understanding into the job_data dict
        so the retrieval pipeline can find truly relevant projects.
        
        Args:
            job_data: Original job data dict
            requirements: JobRequirements dataclass
            
        Returns:
            Enhanced job_data dict
        """
        # Add extracted understanding to job_data
        job_data["exact_task"] = requirements.exact_task
        job_data["client_tone"] = requirements.client_tone
        job_data["client_problems"] = requirements.problems_mentioned
        job_data["client_constraints"] = requirements.constraints
        job_data["tech_stack"] = requirements.tech_stack_mentioned
        job_data["do_not_assume"] = requirements.do_not_assume
        
        # Override industry if extracted with high confidence
        if requirements.inferred_industry and requirements.inferred_industry != "general":
            if requirements.extraction_confidence >= 0.6:
                job_data["industry"] = requirements.inferred_industry
                logger.debug(f"[ProposalService] Industry overridden to: {requirements.inferred_industry}")
        
        # Add complexity for retrieval filtering
        job_data["task_complexity"] = requirements.complexity_level
        
        # Add deliverables for better semantic matching
        if requirements.specific_deliverables:
            job_data["expected_deliverables"] = requirements.specific_deliverables
        
        return job_data
    
    def _detect_industry(self, job_data: Dict[str, Any]) -> str:
        """Detect industry from job data context."""
        try:
            from app.utils.metadata_extractor import MetadataExtractor
            result = MetadataExtractor.detect_industry_with_context(job_data)
            if result.get("industry") != "general" and result.get("confidence", 0) >= 0.5:
                logger.info(f"[ProposalService] Industry detected: {result['industry']}")
                return result["industry"]
        except Exception as e:
            logger.debug(f"Industry detection failed: {e}")
        return job_data.get("industry", "general")
    
    def _get_historical_jobs(self, org_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical jobs - from portfolio_items if org_id, else training_data."""
        if not self.db:
            return []
        try:
            # Try new portfolio_items if org_id provided
            if org_id:
                from app.infra.mongodb.repositories import get_portfolio_repo
                portfolio_repo = get_portfolio_repo()
                items = portfolio_repo.list_by_org(org_id)
                if items:
                    # Convert portfolio items to job-like format for retrieval
                    return [self._portfolio_to_job_format(item) for item in items]
            
            # Fall back to training_data (backward compatible)
            return list(self.db.db.training_data.find({}))
        except Exception as e:
            logger.error(f"Error fetching historical jobs: {e}")
            return []
    
    def _portfolio_to_job_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert lean portfolio item to job format for retrieval compatibility."""
        return {
            "contract_id": item.get("item_id"),
            "job_title": item.get("project_title", ""),
            "skills_required": item.get("skills", []),
            "industry": item.get("industry", "general"),
            "deliverables": item.get("deliverables", []),
            "outcomes": item.get("outcome", ""),
            "portfolio_urls": [item.get("portfolio_url")] if item.get("portfolio_url") else [],
            "client_feedback_text": item.get("client_feedback", ""),
            "duration_days": item.get("duration_days"),
        }
    
    def _get_freelancer_profile(self, org_id: str, profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get freelancer profile for proposal personalization."""
        try:
            from app.infra.mongodb.repositories import get_freelancer_profile_repo
            profile_repo = get_freelancer_profile_repo()
            
            if profile_id:
                return profile_repo.get_by_profile_id(profile_id)
            return profile_repo.get_default_profile(org_id)
        except Exception as e:
            logger.debug(f"Profile fetch failed: {e}")
            return None
    
    def _find_similar_projects(
        self,
        job_data: Dict[str, Any],
        all_jobs: List[Dict[str, Any]],
        top_k: int,
        job_requirements: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Find similar projects using retrieval pipeline.
        
        ENHANCED (Issue #24): Now passes JobRequirements to retrieval for:
        - Better semantic search queries using specific_deliverables
        - Problem-based matching using problems_mentioned
        - More accurate filtering using extracted tech stack and complexity
        """
        if not self.retrieval_pipeline:
            return {"similar_projects": [], "insights": None}
        
        try:
            return self.retrieval_pipeline.retrieve_for_proposal(
                job_data, all_jobs, top_k=top_k,
                job_requirements=job_requirements
            )
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return {"similar_projects": [], "insights": None}
    
    def _filter_projects_with_portfolio(
        self,
        projects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter to projects with actual portfolio URLs."""
        filtered = []
        for project in projects:
            portfolio_urls = project.get("portfolio_urls", [])
            if isinstance(portfolio_urls, str):
                portfolio_urls = [portfolio_urls] if portfolio_urls else []
            portfolio_urls = [url for url in portfolio_urls if url]
            project["portfolio_urls"] = portfolio_urls
            
            if portfolio_urls:
                filtered.append(project)
        
        return filtered if filtered else projects
    
    def _analyze_previous_proposals(
        self,
        similar_projects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze previous proposals for similar jobs."""
        if not self.db:
            return None
        
        previous_proposals = []
        for project in similar_projects:
            contract_id = project.get("contract_id")
            if not contract_id:
                continue
            
            stored = self.db.db.proposals.find_one({
                "contract_id": contract_id,
                "generated_proposal": {"$exists": True}
            })
            
            if stored:
                previous_proposals.append({
                    "contract_id": contract_id,
                    "company": project.get("company"),
                    "proposal_text": stored.get("generated_proposal"),
                    "style": stored.get("proposal_style"),
                    "tone": stored.get("proposal_tone"),
                    "word_count": stored.get("word_count", 0)
                })
        
        if not previous_proposals:
            return None
        
        return {
            "total_previous_proposals": len(previous_proposals),
            "common_patterns": self._extract_proposal_patterns(previous_proposals),
            "effective_phrases": self._extract_effective_phrases(previous_proposals),
            "average_word_count": sum(p["word_count"] for p in previous_proposals) / len(previous_proposals)
        }
    
    def _extract_proposal_patterns(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Extract common patterns from previous proposals."""
        patterns = []
        for p in proposals:
            text = p.get("proposal_text", "")
            if "specific experience" in text.lower():
                patterns.append("References specific past experience")
            if "similar project" in text.lower():
                patterns.append("Mentions similar past projects")
            if any(word in text.lower() for word in ["timeline", "milestone", "deadline"]):
                patterns.append("Includes timeline/milestones")
            if any(word in text.lower() for word in ["portfolio", "work sample", "example"]):
                patterns.append("References portfolio/samples")
        return list(set(patterns))
    
    def _extract_effective_phrases(self, proposals: List[Dict[str, Any]]) -> List[str]:
        """Extract effective phrases from proposals."""
        phrases = []
        for p in proposals:
            text = p.get("proposal_text", "")
            if "I'd love to" in text:
                phrases.append("I'd love to")
            if "Based on my experience" in text:
                phrases.append("Based on my experience")
            if "I've successfully" in text:
                phrases.append("I've successfully")
        return list(set(phrases))[:5]
    
    def _collect_urls(
        self,
        projects: List[Dict[str, Any]],
        include_portfolio: bool,
        include_feedback: bool
    ) -> tuple:
        """Collect portfolio and feedback URLs from projects."""
        portfolio_links = []
        feedback_urls = []
        
        for project in projects:
            if include_portfolio:
                urls = project.get("portfolio_urls", [])
                if isinstance(urls, list):
                    portfolio_links.extend(urls)
                elif isinstance(urls, str) and urls:
                    portfolio_links.append(urls)
            
            if include_feedback:
                feedback_url = project.get("client_feedback_url")
                if feedback_url:
                    feedback_urls.append(feedback_url)
        
        return portfolio_links, feedback_urls
    
    def _build_prompt(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        previous_insights: Optional[Dict[str, Any]],
        portfolio_links: List[str],
        feedback_urls: List[str],
        request: ProposalRequest,
        profile_context: Optional[Dict[str, Any]] = None,
        job_requirements: Optional[Any] = None
    ) -> str:
        """Build the proposal generation prompt with enhanced job understanding."""
        if not self.prompt_engine:
            # Fallback to basic prompt
            return self._build_basic_prompt(job_data, request, profile_context, job_requirements)
        
        success_patterns = []
        if previous_insights:
            success_patterns = previous_insights.get("common_patterns", [])
        
        # Build job requirements context for the prompt engine
        requirements_context = None
        if job_requirements:
            requirements_context = {
                "exact_task": job_requirements.exact_task,
                "deliverables": job_requirements.specific_deliverables,
                "client_tone": job_requirements.client_tone,
                "problems": job_requirements.problems_mentioned,
                "constraints": job_requirements.constraints,
                "do_not_assume": job_requirements.do_not_assume,  # CRITICAL: Anti-hallucination
                "resources_provided": job_requirements.resources_provided,
            }
        
        return self.prompt_engine.build_proposal_prompt(
            job_data=job_data,
            similar_projects=similar_projects,
            success_patterns=success_patterns,
            style=request.proposal_style,
            tone=request.tone,
            max_words=request.max_word_count,
            include_portfolio=request.include_portfolio and bool(portfolio_links),
            include_feedback=request.include_feedback and bool(feedback_urls),
            include_timeline=request.include_timeline,
            timeline_duration=request.timeline_duration,
            profile_context=profile_context,
            requirements_context=requirements_context  # NEW: Pass extracted requirements
        )
    
    def _build_basic_prompt(
        self, 
        job_data: Dict[str, Any], 
        request: ProposalRequest,
        profile_context: Optional[Dict[str, Any]] = None,
        job_requirements: Optional[Any] = None
    ) -> str:
        """Build a basic prompt when prompt engine is not available."""
        profile_section = ""
        if profile_context:
            profile_section = f"""
Freelancer Profile:
- Name: {profile_context.get('display_name', 'Freelancer')}
- Title: {profile_context.get('headline', '')}
- Skills: {', '.join(profile_context.get('skills', [])[:10])}
- Bio: {profile_context.get('bio', '')[:300]}
"""
        
        # Add requirements section if available (NEW - Issue #23)
        requirements_section = ""
        if job_requirements:
            requirements_section = f"""
EXTRACTED JOB UNDERSTANDING:
- Exact Task: {job_requirements.exact_task}
- Client Tone: {job_requirements.client_tone}
- Problems Mentioned: {', '.join(job_requirements.problems_mentioned) if job_requirements.problems_mentioned else 'None stated'}
- DO NOT ASSUME (avoid mentioning these unless client specified): {', '.join(job_requirements.do_not_assume) if job_requirements.do_not_assume else 'N/A'}
"""
        
        return f"""Generate a {request.proposal_style} proposal for:

Job Title: {job_data['job_title']}
Company: {job_data['company_name']}
Description: {job_data['job_description']}
Skills Required: {', '.join(job_data['skills_required'])}
{profile_section}{requirements_section}
Write a {request.tone} proposal in approximately {request.max_word_count} words."""
    
    def _generate_with_ai(self, prompt: str, max_word_count: int) -> tuple:
        """Generate proposal text using AI."""
        if not self.openai_service:
            raise ValueError("OpenAI service not initialized")
        
        max_tokens = int(max_word_count * 1.5) + 100
        
        proposal_text = self.openai_service.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        word_count = len(proposal_text.split())
        logger.info(f"[ProposalService] Generated {word_count} words (target ~{max_word_count})")
        
        return proposal_text, word_count
    
    def _score_proposal(
        self,
        proposal_text: str,
        portfolio_links: List[str],
        feedback_urls: List[str],
        similar_projects: List[Dict[str, Any]],
        request: ProposalRequest
    ) -> tuple:
        """Score proposal quality and generate suggestions."""
        score = 0.5  # Base score
        suggestions = []
        
        # Check for portfolio references
        if portfolio_links and any(link in proposal_text for link in portfolio_links):
            score += 0.15
        elif request.include_portfolio:
            suggestions.append("Consider adding portfolio links to strengthen credibility")
        
        # Check for specific project references
        for project in similar_projects:
            if project.get("company", "").lower() in proposal_text.lower():
                score += 0.1
                break
        
        # Check word count adherence
        word_count = len(proposal_text.split())
        if abs(word_count - request.max_word_count) < request.max_word_count * 0.2:
            score += 0.1
        else:
            suggestions.append(f"Adjust length closer to {request.max_word_count} words")
        
        # Check for personalization
        if request.company_name.lower() in proposal_text.lower():
            score += 0.1
        else:
            suggestions.append("Add more company-specific personalization")
        
        # Check for skill mentions
        skills_mentioned = sum(1 for skill in request.skills_required if skill.lower() in proposal_text.lower())
        if skills_mentioned >= len(request.skills_required) * 0.5:
            score += 0.05
        
        return min(score, 1.0), suggestions if suggestions else None
    
    def save_generated_proposal(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        word_count: int,
        style: str,
        tone: str
    ) -> Optional[str]:
        """Save generated proposal to database."""
        if not self.db:
            return None
        
        try:
            from app.infra.mongodb.repositories import get_proposal_repo
            repo = get_proposal_repo()
            
            return repo.save_proposal({
                "job_title": job_data.get("job_title"),
                "company_name": job_data.get("company_name"),
                "generated_proposal": proposal_text,
                "word_count": word_count,
                "proposal_style": style,
                "proposal_tone": tone,
                "job_data": job_data
            })
        except Exception as e:
            logger.error(f"Error saving proposal: {e}")
            return None
