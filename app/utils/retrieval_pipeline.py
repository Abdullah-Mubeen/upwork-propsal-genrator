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
    platform: Optional[str] = None  # wordpress, shopify, woocommerce, etc.
    urgent_adhoc: Optional[bool] = None  # Prioritize urgent/adhoc projects when job is urgent


class RetrievalPipeline:
    """Multi-stage retrieval for proposal generation"""

    COMPLEXITY_LEVELS = {"low": 1, "medium": 2, "high": 3}
    
    # Platform keywords for smart detection
    # CRITICAL: More specific keywords should come first in each list
    PLATFORM_KEYWORDS = {
        "wordpress": ["wordpress", "wp-admin", "elementor", "divi", "theme", "plugin", "gutenberg", "acf", "wp theme", "wp plugin", "geo directory", "geodirectory"],
        "shopify": ["shopify", "shopify theme", "shopify app", "shopify store", "liquid", "shopify plus"],
        "woocommerce": ["woocommerce", "woo commerce", "woo membership", "woomembership", "woo subscription", "woo-"],
        "wix": ["wix", "wix site", "wix website", "wix editor"],
        "webflow": ["webflow"],
        "squarespace": ["squarespace"],
        "magento": ["magento", "adobe commerce"],
        "drupal": ["drupal"],
        "joomla": ["joomla"],
        "react": ["react", "reactjs", "react.js", "next.js", "nextjs", "react native"],
        "vue": ["vue", "vuejs", "vue.js", "nuxt"],
        "angular": ["angular", "angularjs"],
        "html_css": ["static site", "landing page"],  # Removed html/css as they're too generic
    }

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
                    "task_type": p.get("task_type"),  # Include task type for matching validation
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
        
        CRITICAL: Now uses LLM-based semantic industry detection when needed,
        and extracts CLIENT INTENTS (what they actually want done).
        """
        from app.utils.metadata_extractor import MetadataExtractor
        
        skills = job_data.get("skills_required", [])
        industry = job_data.get("industry", "")
        task_type = job_data.get("task_type", "")
        complexity = job_data.get("task_complexity", "medium")
        urgency = job_data.get("urgent_adhoc", False)
        
        # CRITICAL: Detect platform from job description and skills
        platform = self._detect_platform(job_data)
        if platform:
            logger.info(f"  → Detected platform: {platform.upper()} - will prioritize {platform} projects")

        # CRITICAL: Extract client intents (what they actually want done)
        client_intents = MetadataExtractor.extract_client_intents(job_data)
        if client_intents:
            logger.info(f"  → Detected client intents: {client_intents[:3]}")
            # Store intents in job_data for later use in ranking
            job_data["client_intents"] = client_intents

        # Log urgency detection
        if urgency:
            logger.info(f"  → URGENT/AD-HOC project detected - will prioritize similar urgent projects")

        # Build flexible criteria (not overly restrictive)
        # We want to find projects with feedback first, then broaden if needed
        return FilterCriteria(
            industries=[industry] if industry else None,
            skills=skills if skills else None,
            task_type=task_type if task_type else None,
            with_feedback=False,  # Don't require feedback - include if available
            completed_only=True,  # Only completed projects (proven success)
            platform=platform,  # CRITICAL: Platform-specific filtering
            urgent_adhoc=urgency if urgency else None  # Prioritize urgent projects when job is urgent
        )
    
    def _detect_platform(self, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Detect the primary platform from job description and skills.
        
        This is CRITICAL for ensuring WordPress jobs get WordPress examples,
        Shopify jobs get Shopify examples, etc.
        
        Returns:
            Platform name (lowercase) or None if not detected
        """
        job_desc = job_data.get("job_description", "").lower()
        job_title = job_data.get("job_title", "").lower()
        skills = [s.lower() for s in job_data.get("skills_required", [])]
        
        # Combine all text for searching
        search_text = f"{job_title} {job_desc} {' '.join(skills)}"
        
        # Check each platform's keywords
        platform_scores = {}
        for platform, keywords in self.PLATFORM_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in search_text:
                    score += 1
                    # Extra weight for skill matches (more specific)
                    if keyword in skills:
                        score += 2
            if score > 0:
                platform_scores[platform] = score
        
        if not platform_scores:
            return None
        
        # Return the platform with highest score
        detected = max(platform_scores, key=platform_scores.get)
        logger.debug(f"  Platform scores: {platform_scores} → Selected: {detected}")
        return detected

    def _metadata_filter(
        self,
        jobs: List[Dict[str, Any]],
        criteria: FilterCriteria
    ) -> List[Dict[str, Any]]:
        """
        STAGE 1: Fast metadata filtering with platform exclusion.
        Priority: Urgent → Platform+Industry → Platform → Industry → Skills → Fallback
        """
        # All CMS platforms are mutually exclusive
        CMS_PLATFORMS = {"wordpress", "woocommerce", "shopify", "wix", "magento", "webflow", "squarespace"}
        JS_FRAMEWORKS = {"react": {"angular", "vue"}, "angular": {"react", "vue"}, "vue": {"react", "angular"}}
        
        # Get excluded platforms
        if criteria.platform in CMS_PLATFORMS:
            excluded = CMS_PLATFORMS - {criteria.platform, "woocommerce" if criteria.platform == "wordpress" else criteria.platform}
            if criteria.platform in ("wordpress", "woocommerce"):
                excluded -= {"wordpress", "woocommerce"}  # WP and WooCommerce are related
        else:
            excluded = JS_FRAMEWORKS.get(criteria.platform, set())
        
        target_industry = (criteria.industries[0].lower() if criteria.industries else None)
        results = {"urgent": [], "platform": [], "industry": [], "skills": []}
        
        for job in jobs:
            if criteria.completed_only and job.get("project_status", "").lower() != "completed":
                continue
            
            job_platform = self._detect_job_platform(job)
            job_skills_text = " ".join(s.lower() for s in job.get("skills_required", []))
            
            # CRITICAL: Skip if competing platform detected OR in skills
            if criteria.platform:
                if job_platform in excluded:
                    logger.debug(f"  ✗ Excluding {job.get('company_name')} - platform {job_platform} conflicts")
                    continue
                if any(kw in job_skills_text for p in excluded for kw in self.PLATFORM_KEYWORDS.get(p, [p])):
                    logger.debug(f"  ✗ Excluding {job.get('company_name')} - has competing platform skill")
                    continue
            
            # Check matches
            job_industry = (job.get("industry") or "").lower()
            industry_match = target_industry and (job_industry == target_industry or 
                            target_industry in [t.lower() for t in job.get("industry_tags", [])])
            platform_match = (job_platform == criteria.platform or 
                             self._are_platforms_related(criteria.platform, job_platform))
            
            # Categorize by priority
            if criteria.platform and platform_match:
                if criteria.urgent_adhoc and job.get("urgent_adhoc"):
                    results["urgent"].append(job)
                elif industry_match:
                    results["platform"].insert(0, job)  # Platform+Industry at front
                else:
                    results["platform"].append(job)
            elif industry_match and job_platform not in excluded:
                results["industry"].append(job)
            elif criteria.skills and job_platform not in excluded:
                if set(s.lower() for s in job.get("skills_required", [])) & set(s.lower() for s in criteria.skills):
                    results["skills"].append(job)
        
        # Return by priority
        for key in ["urgent", "platform", "industry", "skills"]:
            if results[key]:
                if key == "urgent":
                    return results["urgent"] + [p for p in results["platform"] if p not in results["urgent"]]
                logger.info(f"  → Found {len(results[key])} {key}-matched projects")
                return results[key]
        
        # Fallback: non-competing completed projects
        logger.warning(f"  ⚠ No matches, returning non-competing completed projects")
        return [j for j in jobs if j.get("project_status", "").lower() == "completed" 
                and self._detect_job_platform(j) not in excluded][:10]
    
    def _detect_job_platform(self, job: Dict[str, Any]) -> Optional[str]:
        """Detect platform from job data using weighted keyword scoring."""
        text = f"{job.get('job_title', '')} {job.get('job_description', '')}".lower()
        skills = [s.lower() for s in job.get("skills_required", [])]
        skills_text = " ".join(skills)
        
        scores = {}
        for platform, keywords in self.PLATFORM_KEYWORDS.items():
            score = sum(
                4 if kw in skills else 3 if kw in skills_text else 2 if kw in job.get("job_title", "").lower() else 1 if kw in text else 0
                for kw in keywords
            )
            if score:
                scores[platform] = score
        
        if not scores:
            return None
        detected = max(scores, key=scores.get)
        logger.debug(f"  Platform: {job.get('company_name', '?')} → {detected} (scores: {scores})")
        return detected
    
    def _are_platforms_related(self, platform1: Optional[str], platform2: Optional[str]) -> bool:
        """
        Check if two platforms are related (e.g., WooCommerce is part of WordPress).
        This helps find relevant projects when exact platform match isn't available.
        """
        if not platform1 or not platform2:
            return False
        
        # Define related platforms (bi-directional relationships)
        related_platforms = {
            "wordpress": ["woocommerce", "elementor", "php"],
            "woocommerce": ["wordpress", "php"],
            "shopify": ["liquid"],  # Shopify uses Liquid templating
            "react": ["next.js", "javascript"],
            "vue": ["nuxt", "javascript"],
            "angular": ["typescript", "javascript"],
            "html_css": ["javascript", "php"],  # Basic web can relate to backend
        }
        
        return platform2 in related_platforms.get(platform1, [])
        
        return platform2 in related_platforms.get(platform1, [])

    def _semantic_rank(
        self,
        new_job: Dict[str, Any],
        candidate_jobs: List[Dict[str, Any]],
        use_semantic_search: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        STAGE 2: Semantic ranking using Pinecone embeddings + metadata.
        
        Combines:
        1. Semantic similarity from Pinecone embeddings (if available)
        2. Metadata similarity from project attributes
        3. Diversity boost to avoid returning the same portfolios repeatedly
        """
        from app.utils.metadata_extractor import MetadataExtractor

        ranked = []
        pinecone_scores = {}  # contract_id -> semantic score from Pinecone

        # Try to use Pinecone semantic search for better ranking
        if use_semantic_search and self.pinecone_service:
            try:
                import os
                from app.utils.openai_service import OpenAIService
                from app.config import settings
                
                api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                    
                openai_service = OpenAIService(api_key=api_key)
                
                # Create query text from new job
                query_text = f"{new_job.get('job_title', '')} {new_job.get('job_description', '')[:500]}"
                query_embedding = openai_service.get_embedding(query_text)
                
                if query_embedding:
                    # Query Pinecone for semantic matches
                    pinecone_results = self.pinecone_service.query_vectors(
                        query_embedding=query_embedding,
                        top_k=50,  # Get more results for diversity
                        include_metadata=True
                    )
                    
                    # Build contract_id -> score mapping from Pinecone results
                    for result in pinecone_results:
                        contract_id = result.get("metadata", {}).get("contract_id")
                        if contract_id:
                            # Keep highest score for each contract
                            current_score = pinecone_scores.get(contract_id, 0)
                            pinecone_scores[contract_id] = max(current_score, result.get("score", 0))
                    
                    logger.info(f"  → Pinecone returned {len(pinecone_scores)} unique contracts with scores")
            except Exception as e:
                logger.warning(f"  ⚠ Pinecone semantic search failed, using metadata only: {e}")

        # Calculate combined scores for each candidate job
        for job in candidate_jobs:
            contract_id = job.get("contract_id", "")
            
            # Metadata similarity (always computed)
            similarity_scores = MetadataExtractor.compare_projects(new_job, job)
            metadata_sim = MetadataExtractor.calculate_overall_similarity(similarity_scores)
            
            # Semantic similarity from Pinecone (if available)
            semantic_sim = pinecone_scores.get(contract_id, 0.0)
            
            # Combined score: weight semantic search higher when available
            if semantic_sim > 0:
                # 60% semantic + 40% metadata when we have Pinecone scores
                combined_score = (semantic_sim * 0.6) + (metadata_sim * 0.4)
            else:
                # Fall back to metadata only
                combined_score = metadata_sim

            ranked.append((job, combined_score))

        # Sort by combined score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity boost: avoid returning jobs with same company name or very similar portfolios
        diversified = self._apply_diversity(ranked)

        return diversified

    def _apply_diversity(
        self,
        ranked_projects: List[Tuple[Dict[str, Any], float]],
        max_per_company: int = 1,
        max_shared_urls: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply diversity to avoid returning the same companies/portfolios repeatedly.
        
        This ensures proposal references are varied and don't always cite the same projects.
        
        ENHANCED: Now tracks INDIVIDUAL portfolio URLs to prevent the same links
        from appearing across different referenced projects.
        
        Args:
            ranked_projects: Ranked list of (project, score) tuples
            max_per_company: Max projects from same company (default 1)
            max_shared_urls: Max times a single portfolio URL can appear (default 1)
        """
        seen_companies = {}  # company_name -> count
        seen_portfolio_urls = {}  # individual URL -> count (track each URL separately)
        seen_feedback_urls = set()  # Track feedback URLs for uniqueness
        diversified = []
        
        for job, score in ranked_projects:
            company = job.get("company_name", "").lower().strip()
            portfolio_urls = job.get("portfolio_urls", [])
            feedback_url = job.get("client_feedback_url", "")
            
            # Skip if we've already selected max_per_company projects from this company
            if company and seen_companies.get(company, 0) >= max_per_company:
                logger.debug(f"  → Skipping duplicate company: {company}")
                continue
            
            # Check if this project has UNIQUE portfolio URLs
            # A project is good if at least ONE of its URLs is fresh
            has_fresh_urls = False
            fresh_url_count = 0
            for url in portfolio_urls:
                if url:
                    url_lower = url.lower().strip()
                    if seen_portfolio_urls.get(url_lower, 0) < max_shared_urls:
                        has_fresh_urls = True
                        fresh_url_count += 1
            
            # Skip if ALL portfolio URLs have already been used in other projects
            if portfolio_urls and not has_fresh_urls:
                logger.debug(f"  → Skipping project with no fresh portfolio URLs: {company}")
                continue
            
            # Prefer projects with MORE unique URLs (diversity bonus)
            # Slight score boost for projects bringing new portfolio examples
            diversity_bonus = min(fresh_url_count * 0.02, 0.1)  # Up to 10% bonus
            adjusted_score = score + diversity_bonus
            
            # Add to diversified results
            diversified.append((job, adjusted_score))
            
            # Track company usage
            if company:
                seen_companies[company] = seen_companies.get(company, 0) + 1
            
            # Track EACH portfolio URL individually
            for url in portfolio_urls:
                if url:
                    url_lower = url.lower().strip()
                    seen_portfolio_urls[url_lower] = seen_portfolio_urls.get(url_lower, 0) + 1
            
            # Track feedback URLs
            if feedback_url:
                seen_feedback_urls.add(feedback_url.lower().strip())
        
        # Re-sort by adjusted score after diversity bonus
        diversified.sort(key=lambda x: x[1], reverse=True)
        
        unique_urls = len(seen_portfolio_urls)
        unique_companies = len(seen_companies)
        logger.info(f"  → Diversity filter: {len(ranked_projects)} → {len(diversified)} unique projects ({unique_companies} companies, {unique_urls} unique portfolio URLs)")
        return diversified

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
            if job.get("project_status", "").lower() != "completed":
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
            if job.get("project_status", "").lower() == "completed" and success_count < 3:
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

        # Calculate estimated success rate (case-insensitive)
        completed = sum(1 for j, _ in similar_projects if j.get("project_status", "").lower() == "completed")
        context["estimated_success_rate"] = completed / len(similar_projects) if similar_projects else 0.5

        return context
