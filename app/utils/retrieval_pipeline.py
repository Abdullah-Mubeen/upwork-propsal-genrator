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
                    "feedback_url": p.get("client_feedback_url"),
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
        """Build filter criteria from new job requirements"""
        skills = job_data.get("skills_required", [])
        industry = job_data.get("industry", "")
        task_type = job_data.get("task_type", "")
        complexity = job_data.get("task_complexity", "medium")
        urgency = job_data.get("urgent_adhoc", False)

        # Build flexible criteria (not overly restrictive)
        return FilterCriteria(
            industries=[industry] if industry else None,
            skills=skills if skills else None,
            task_type=task_type if task_type else None,
            min_effectiveness=0.3,  # Only consider reasonably successful projects
            with_feedback=True,  # Prefer projects with feedback
            completed_only=False  # Include ongoing but require feedback
        )

    def _metadata_filter(
        self,
        jobs: List[Dict[str, Any]],
        criteria: FilterCriteria
    ) -> List[Dict[str, Any]]:
        """
        STAGE 1: Fast metadata filtering.
        
        Applies hard filters to reduce search space.
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

            # Effectiveness threshold
            if criteria.min_effectiveness:
                effectiveness = job.get("proposal_effectiveness_score") or 0
                if effectiveness < criteria.min_effectiveness:
                    continue

            # Satisfaction threshold
            if criteria.min_satisfaction:
                satisfaction = job.get("client_satisfaction") or 0
                if satisfaction < criteria.min_satisfaction:
                    continue

            # Industry filter (soft - at least one match)
            if criteria.industries:
                job_industries = set(job.get("industry_tags", []))
                if not any(ind in job_industries for ind in criteria.industries):
                    if job.get("industry") not in criteria.industries:
                        continue

            # Skills filter (soft - at least some overlap)
            if criteria.skills:
                job_skills = set(s.lower() for s in job.get("skills_required", []))
                query_skills = set(s.lower() for s in criteria.skills)
                if not (job_skills & query_skills):  # No overlap
                    continue

            # Complexity filter
            if criteria.min_complexity or criteria.max_complexity:
                job_complexity = job.get("task_complexity", "medium")
                min_level = self.COMPLEXITY_LEVELS.get(criteria.min_complexity or "low", 1)
                max_level = self.COMPLEXITY_LEVELS.get(criteria.max_complexity or "high", 3)
                job_level = self.COMPLEXITY_LEVELS.get(job_complexity, 2)
                if not (min_level <= job_level <= max_level):
                    continue

            # Duration filter
            if criteria.min_duration_days or criteria.max_duration_days:
                duration = job.get("project_duration_days")
                if duration:
                    min_dur = criteria.min_duration_days or 0
                    max_dur = criteria.max_duration_days or float('inf')
                    if not (min_dur <= duration <= max_dur):
                        continue

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
        STAGE 3: Analyze feedback from top projects for patterns.
        
        Extracts:
        - Success patterns (what worked)
        - Common objections/solutions
        - Winning sections
        - Writing style patterns
        """
        insights = {
            "success_patterns": [],
            "common_elements": {},
            "winning_sections": [],
            "feedback_sentiment": {},
            "writing_patterns": []
        }

        success_count = 0
        feedback_samples = []

        for job, similarity in similar_projects:
            if job.get("project_status") != "completed":
                continue

            success_count += 1
            
            # Collect feedback
            feedback = job.get("client_feedback_text") or job.get("client_feedback", "")
            if feedback:
                feedback_samples.append(feedback)

            # Collect reusable sections
            sections = job.get("reusable_sections", [])
            for section in sections:
                insights["winning_sections"].append(section)
                insights["common_elements"][section] = insights["common_elements"].get(section, 0) + 1

        # Analyze feedback samples
        if feedback_samples:
            # Extract common positive phrases
            phrases = self._extract_phrases(feedback_samples)
            insights["success_patterns"] = phrases[:5]  # Top 5 patterns

        # Deduplicate and rank winning sections
        section_counts = insights["common_elements"]
        insights["winning_sections"] = sorted(
            section_counts.keys(),
            key=lambda x: section_counts[x],
            reverse=True
        )

        # Estimate effectiveness
        insights["successful_project_count"] = success_count
        insights["success_rate"] = success_count / len(similar_projects) if similar_projects else 0

        return insights

    def _extract_phrases(self, texts: List[str], num_phrases: int = 5) -> List[str]:
        """Extract common positive phrases from feedback texts"""
        phrases = []
        
        # Simple phrase extraction based on positive keywords
        positive_contexts = {
            "Great communication": ["communication", "responsive", "quick", "reply"],
            "Technical excellence": ["excellent code", "well-structured", "professional", "architecture"],
            "Timely delivery": ["on time", "deadline", "fast", "quick turnaround"],
            "Problem solving": ["challenges", "obstacles", "solutions", "adapt"],
            "Attention to detail": ["detail", "quality", "polish", "refinement"],
        }

        for context, keywords in positive_contexts.items():
            for text in texts:
                text_lower = text.lower()
                if any(kw in text_lower for kw in keywords):
                    if context not in phrases:
                        phrases.append(context)
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
