"""
Multi-Layer Retrieval Pipeline for Proposal Generation

Implements 3-stage intelligent retrieval:
1. METADATA FILTERING (Fast) - Filter by industry, skills, urgency, etc.
2. SEMANTIC SEARCH (Smart) - Find similar projects using embeddings
3. FEEDBACK ANALYSIS (Learning) - Extract patterns from top projects

Returns ranked similar projects with actionable insights for proposal generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Metadata filter criteria for Stage 1"""
    industries: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    task_type: Optional[str] = None
    min_complexity: Optional[str] = None  # low, medium, high
    max_complexity: Optional[str] = None
    min_duration_days: Optional[int] = None
    max_duration_days: Optional[int] = None
    min_effectiveness: Optional[float] = None  # 0-1
    min_satisfaction: Optional[float] = None  # 1-5
    completed_only: bool = False
    with_feedback: bool = False


class RetrievalPipeline:
    """Multi-stage retrieval for proposal generation"""

    COMPLEXITY_LEVELS = {"low": 1, "medium": 2, "high": 3}

    def __init__(self, db=None, pinecone_service=None):
        """
        Initialize retrieval pipeline.
        
        Args:
            db: Database instance for job data access
            pinecone_service: Pinecone service for semantic search
        """
        self.db = db
        self.pinecone_service = pinecone_service
        logger.info("RetrievalPipeline initialized")

    def retrieve_for_proposal(
        self,
        new_job_requirements: Dict[str, Any],
        all_jobs: List[Dict[str, Any]],
        top_k: int = 5,
        use_semantic_search: bool = True
    ) -> Dict[str, Any]:
        """
        Complete retrieval pipeline for a new job requiring a proposal.
        
        Args:
            new_job_requirements: New job data to generate proposal for
            all_jobs: All historical jobs in the system
            top_k: Number of similar projects to return
            use_semantic_search: Whether to use semantic search (requires embeddings)
            
        Returns:
            Comprehensive retrieval result with context for proposal generation
        """
        job_id = new_job_requirements.get("contract_id", "unknown")
        logger.info(f"[Retrieval] Starting pipeline for {job_id}")

        # STAGE 1: METADATA FILTERING
        logger.debug(f"[Stage 1] Filtering by metadata...")
        filter_criteria = self._build_filter_criteria(new_job_requirements)
        filtered_jobs = self._metadata_filter(all_jobs, filter_criteria)
        logger.info(f"  → Filtered: {len(all_jobs)} jobs → {len(filtered_jobs)} matches")

        if not filtered_jobs:
            logger.warning(f"  ⚠ No jobs match filter criteria, using all jobs")
            filtered_jobs = all_jobs[:20]  # Fallback to top 20

        # STAGE 2: SEMANTIC SEARCH
        logger.debug(f"[Stage 2] Semantic similarity search...")
        ranked_projects = self._semantic_rank(
            new_job_requirements,
            filtered_jobs,
            use_semantic_search=use_semantic_search
        )
        ranked_projects = ranked_projects[:top_k]
        logger.info(f"  → Ranked: {len(ranked_projects)} top similar projects")

        # STAGE 3: FEEDBACK ANALYSIS
        logger.debug(f"[Stage 3] Analyzing feedback patterns...")
        insights = self._extract_insights(ranked_projects, new_job_requirements)
        logger.info(f"  → Extracted {len(insights['success_patterns'])} success patterns")

        result = {
            "query_job": {
                "contract_id": job_id,
                "title": new_job_requirements.get("job_title"),
                "industry": new_job_requirements.get("industry"),
                "skills": new_job_requirements.get("skills_required", []),
            },
            "stage1_filtered_count": len(filtered_jobs),
            "similar_projects": [
                {
                    "contract_id": p.get("contract_id"),
                    "company": p.get("company_name"),
                    "title": p.get("job_title"),
                    "industry": p.get("industry"),
                    "skills": p.get("skills_required", []),
                    "similarity_score": score,
                    "effectiveness": p.get("proposal_effectiveness_score"),
                    "satisfaction": p.get("client_satisfaction"),
                    "portfolio_urls": p.get("portfolio_urls", []),
                    "client_feedback_url": p.get("client_feedback_url"),
                }
                for p, score in ranked_projects
            ],
            "insights": insights,
            "proposal_context": self._build_proposal_context(ranked_projects, new_job_requirements)
        }

        logger.info(f"✓ [Retrieval] Complete: {len(ranked_projects)} projects, "
                   f"{len(insights['success_patterns'])} patterns, "
                   f"{len(insights['winning_sections'])} sections")

        return result

    def _build_filter_criteria(self, job_data: Dict[str, Any]) -> FilterCriteria:
        """
        Build intelligent filter criteria from new job requirements.
        
        Prioritizes projects with feedback and proven effectiveness while
        keeping filters flexible to find relevant matches.
        
        NOTE: Training data has client_feedback_text but NOT proposal_effectiveness_score
        or client_satisfaction fields, so we only filter by feedback presence and status.
        """
        skills = job_data.get("skills_required", [])
        industry = job_data.get("industry", "")
        task_type = job_data.get("task_type", "")
        complexity = job_data.get("task_complexity", "medium")
        urgency = job_data.get("urgent_adhoc", False)

        # Build flexible criteria (not overly restrictive)
        # We want to find projects with feedback first, then broaden if needed
        # NOTE: min_effectiveness and min_satisfaction are disabled since those fields don't exist
        return FilterCriteria(
            industries=[industry] if industry else None,
            skills=skills if skills else None,
            task_type=task_type if task_type else None,
            with_feedback=True,  # MUST have feedback to learn from
            completed_only=True  # Only completed projects (proven success)
        )

    def _metadata_filter(
        self,
        jobs: List[Dict[str, Any]],
        criteria: FilterCriteria
    ) -> List[Dict[str, Any]]:
        """
        STAGE 1: Fast metadata filtering.
        
        Applies smart filters to reduce search space while keeping good candidates.
        Prioritizes task_type and completed projects with feedback.
        """
        filtered = []

        for job in jobs:
            # Completed projects preferred if specified
            if criteria.completed_only and job.get("project_status") != "completed":
                continue

            # Must have feedback if specified
            feedback = job.get("client_feedback_text") or job.get("client_feedback")
            if criteria.with_feedback and not feedback:
                continue

            # Task type filter - SOFT (if specified, try to match but not exclusive)
            if criteria.task_type:
                if job.get("task_type") == criteria.task_type:
                    filtered.append(job)
                    continue
                # If task type doesn't match, might skip but keep looking for other good matches
                # Don't skip yet - will use skills/industry as fallback

            # Skills filter (soft - at least some overlap)
            if criteria.skills:
                job_skills = set(s.lower() for s in job.get("skills_required", []))
                query_skills = set(s.lower() for s in criteria.skills)
                if not (job_skills & query_skills):  # No overlap
                    continue

            # If we got here and task_type was specified but didn't match, still add it as fallback
            if criteria.task_type and job.get("task_type") != criteria.task_type:
                filtered.append(job)
            elif not criteria.task_type:
                filtered.append(job)

        # If we filtered to 0, return all completed projects with feedback
        if not filtered:
            for job in jobs:
                if job.get("project_status") == "completed":
                    feedback = job.get("client_feedback_text") or job.get("client_feedback")
                    if feedback:
                        filtered.append(job)

        return filtered

    def _semantic_rank(
        self,
        new_job: Dict[str, Any],
        candidate_jobs: List[Dict[str, Any]],
        use_semantic_search: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        STAGE 2: Semantic ranking using embeddings + metadata.
        
        Combines semantic similarity with metadata similarity.
        """
        from app.utils.metadata_extractor import MetadataExtractor

        ranked = []

        for job in candidate_jobs:
            # Use metadata similarity
            similarity_scores = MetadataExtractor.compare_projects(new_job, job)
            overall_sim = MetadataExtractor.calculate_overall_similarity(similarity_scores)

            # Could enhance with semantic search if available
            # For now, use metadata-based ranking
            ranked.append((job, overall_sim))

        # Sort by similarity descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def _extract_insights(
        self,
        similar_projects: List[Tuple[Dict[str, Any], float]],
        new_job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        STAGE 3: Analyze feedback from top projects for winning patterns.
        
        Extracts:
        - Success patterns (what worked in proposals that won)
        - Common elements (skills, approaches, messaging)
        - Winning sections (how to structure proposals)
        - Client values (what clients praised)
        """
        insights = {
            "success_patterns": [],
            "common_elements": {},
            "winning_sections": [],
            "client_values": [],
            "feedback_sentiment": "positive",
            "success_rate": 0.0
        }

        success_count = 0
        feedback_samples = []
        proposals_samples = []

        for job, similarity in similar_projects:
            if job.get("project_status") != "completed":
                continue

            success_count += 1
            
            # Collect feedback
            feedback = job.get("client_feedback_text") or job.get("client_feedback", "")
            if feedback:
                feedback_samples.append(feedback)

            # Collect proposal text for pattern analysis
            proposal = job.get("your_proposal_text", "")
            if proposal:
                proposals_samples.append(proposal)

        # Extract success patterns from proposals and feedback
        if proposals_samples:
            insights["success_patterns"] = self._extract_winning_patterns(proposals_samples, feedback_samples)

        # Extract what clients valued
        if feedback_samples:
            insights["client_values"] = self._extract_client_values(feedback_samples)
            insights["feedback_sentiment"] = self._analyze_feedback_sentiment(feedback_samples)

        # Calculate success rate
        insights["success_rate"] = success_count / len(similar_projects) if similar_projects else 0
        insights["successful_project_count"] = success_count

        # Add winning sections template
        insights["winning_sections"] = self._build_winning_sections_template(new_job)

        logger.debug(f"  Extracted patterns: {len(insights['success_patterns'])} patterns, "
                    f"{len(insights['client_values'])} client values")

        return insights

    def _extract_winning_patterns(self, proposals: List[str], feedbacks: List[str]) -> List[str]:
        """Extract patterns that made proposals win"""
        patterns = []
        
        if not proposals:
            logger.debug("  No proposals to analyze for patterns")
            return patterns
        
        logger.debug(f"  Analyzing {len(proposals)} proposals for winning patterns...")
        
        # Analyze what made proposals successful
        pattern_indicators = {
            "Reference specific past projects": ["completed", "worked on", "delivered", "built", "designed", "developed"],
            "Show understanding of their problem": ["understand", "see you", "dealing with", "challenge", "realize", "know"],
            "Include portfolio proof": ["portfolio", "github", "live site", "project link", "attachment", "check"],
            "Be specific about approach": ["approach", "methodology", "process", "steps", "strategy", "method"],
            "Include realistic timeline": ["timeline", "phases", "weeks", "months", "days", "schedule"],
            "Personal and conversational tone": ["i've", "you're", "let's", "excited", "hey", "thanks", "thank"],
            "Show past results/metrics": ["improved", "increased", "reduced", "%", "faster", "faster", "better"],
            "Address client's specific tech stack": ["wordpress", "shopify", "react", "python", "nodejs", "aws", "html", "css"]
        }
        
        for pattern, indicators in pattern_indicators.items():
            # Check if pattern appears in successful proposals
            matches = 0
            for proposal in proposals[:10]:  # Check top 10 proposals
                if any(indicator in proposal.lower() for indicator in indicators):
                    matches += 1
            
            if matches >= 1:  # Pattern appears in at least 1 winning proposal (lowered threshold)
                patterns.append(pattern)
                logger.debug(f"    ✓ Pattern: {pattern} (found in {matches} proposals)")
        
        logger.debug(f"  Extracted {len(patterns)} patterns total")
        return patterns[:8]  # Return up to 8 patterns

    def _extract_client_values(self, feedbacks: List[str]) -> List[str]:
        """Extract what clients valued from feedback"""
        values = []
        value_keywords = {
            "Responsive communication": ["responsive", "communication", "quick reply", "available"],
            "Professional quality": ["professional", "quality", "polished", "attention to detail"],
            "Timely delivery": ["on time", "deadline", "fast", "quick turnaround"],
            "Problem solving": ["challenges", "obstacles", "creative", "solutions"],
            "Exceeded expectations": ["exceeded", "amazing", "great", "fantastic"],
            "Technical expertise": ["expertise", "knowledge", "skilled", "experienced"],
            "Attention to detail": ["detail", "thorough", "precise", "careful"]
        }
        
        for value, keywords in value_keywords.items():
            for feedback in feedbacks:
                if any(kw in feedback.lower() for kw in keywords):
                    if value not in values:
                        values.append(value)
                    break
        
        return values[:5]

    def _analyze_feedback_sentiment(self, feedbacks: List[str]) -> str:
        """Simple sentiment analysis of feedback"""
        if not feedbacks:
            return "neutral"
        
        positive_words = ["great", "excellent", "amazing", "fantastic", "love", "perfect", "best", "wonderful"]
        negative_words = ["bad", "poor", "terrible", "awful", "hate", "worst", "disappointing"]
        
        positive_count = sum(1 for f in feedbacks if any(word in f.lower() for word in positive_words))
        negative_count = sum(1 for f in feedbacks if any(word in f.lower() for word in negative_words))
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _build_winning_sections_template(self, new_job: Dict[str, Any]) -> List[str]:
        """Build template of sections that should be in winning proposal"""
        return [
            "HOOK: Acknowledge their specific problem",
            "PROOF: 2-3 past similar projects with outcomes",
            "PROOF: Portfolio links for credibility",
            "APPROACH: Specific solution for their tech stack",
            "TIMELINE: Realistic phases and duration",
            "CTA: Clear, friendly call-to-action"
        ]

    def _extract_phrases(self, texts: List[str], num_phrases: int = 5) -> List[str]:
        """Extract common positive phrases and winning patterns from feedback texts"""
        phrases = []
        
        # Winning patterns that appear in successful proposals
        positive_patterns = {
            "Reference specific past projects": "Mentioning company names and past work",
            "Show understanding of their problem": "Acknowledge their specific challenge",
            "Include portfolio proof": "Add links to live projects and work samples",
            "Be conversational and human": "Use casual, direct language (not AI tone)",
            "Include metrics and results": "Show past performance numbers and outcomes",
            "Specific to their tech stack": "Address their specific technologies",
            "Clear timeline included": "Provide realistic project phases and duration"
        }

        for pattern, description in positive_patterns.items():
            # Check if pattern-related keywords appear in feedback
            pattern_keywords = pattern.lower().split() + description.lower().split()
            
            for text in texts:
                text_lower = text.lower()
                # If multiple keywords match, consider pattern found
                if sum(1 for kw in pattern_keywords if kw in text_lower) >= 2:
                    if pattern not in phrases:
                        phrases.append(pattern)
                    break

        return phrases[:num_phrases]

    def _build_proposal_context(
        self,
        similar_projects: List[Tuple[Dict[str, Any], float]],
        new_job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build rich context for AI proposal generation.
        
        Combines similar project examples with patterns and insights.
        """
        context = {
            "reference_projects": [],
            "patterns_to_follow": [],
            "sections_to_include": [],
            "tone_style": "professional",
            "estimated_success_rate": 0.0
        }

        # Extract top 3 successful projects as references
        success_count = 0
        for job, similarity in similar_projects:
            if job.get("project_status") == "completed" and success_count < 3:
                context["reference_projects"].append({
                    "company": job.get("company_name"),
                    "industry": job.get("industry"),
                    "skills": job.get("skills_required", []),
                    "feedback": (job.get("client_feedback_text") or 
                               job.get("client_feedback", ""))[:200],  # First 200 chars
                    "feedback_url": job.get("client_feedback_url"),
                    "effectiveness": job.get("proposal_effectiveness_score")
                })
                success_count += 1

        # Calculate estimated success rate
        completed = sum(1 for j, _ in similar_projects if j.get("project_status") == "completed")
        context["estimated_success_rate"] = completed / len(similar_projects) if similar_projects else 0.5

        return context
