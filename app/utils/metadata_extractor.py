"""
Metadata Extractor for Multi-Dimensional Job Analysis

Computes and extracts:
1. Proposal Effectiveness Scores (0-1) from feedback
2. Client Satisfaction Ratings (1-5) from feedback sentiment
3. Task Complexity Levels (low, medium, high)
4. Industry Tags for categorical filtering
5. Success Patterns from completed projects
6. Reusable Sections from successful proposals

These metadata dimensions enable optimal hybrid retrieval and proposal generation.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts multi-dimensional metadata from job data"""

    # Industry mapping for standardization
    INDUSTRY_KEYWORDS = {
        "saas": ["saas", "software as a service", "cloud", "subscription"],
        "e-commerce": ["e-commerce", "ecommerce", "shopify", "woo commerce", "store"],
        "healthcare": ["healthcare", "health", "medical", "hospital", "clinic", "telemedicine"],
        "finance": ["finance", "financial", "banking", "fintech", "crypto"],
        "education": ["education", "edtech", "learning", "course", "university"],
        "real_estate": ["real estate", "realestate", "property", "rental"],
        "manufacturing": ["manufacturing", "factory", "industrial", "logistics"],
        "travel": ["travel", "tourism", "booking", "hotel", "flight"],
        "social": ["social", "networking", "community", "forum"],
        "media": ["media", "entertainment", "streaming", "video", "podcast"],
    }

    # Complexity factors
    COMPLEXITY_INDICATORS = {
        "high": ["machine learning", "ai", "blockchain", "real-time", "high volume", "complex integration", "multi-tenant", "microservices", "distributed"],
        "medium": ["api", "database", "authentication", "payment", "integration", "mobile responsive"],
        "low": ["landing page", "blog", "portfolio", "static", "basic crud"],
    }

    # CLIENT INTENT CATEGORIES - What the client ACTUALLY wants done
    # This is CRITICAL for matching jobs by actual requirement, not just platform
    CLIENT_INTENT_KEYWORDS = {
        # Migration/Transfer intents
        "content_migration": ["migrate", "migration", "transfer content", "move content", "import content", "export content", "convert", "transition", "switch from", "move from", "substack", "mailchimp import", "medium import"],
        "platform_switch": ["switch to wordpress", "move to shopify", "migrate to", "convert to wordpress", "rebuild on", "recreate on"],
        
        # Membership/Subscription intents  
        "membership_setup": ["membership", "woomembership", "subscription site", "paid content", "paywall", "member area", "paid subscriber", "recurring payment", "member pages", "restrict content", "premium content"],
        "newsletter_email": ["newsletter", "email list", "mailing list", "email marketing", "mailchimp", "convertkit", "email subscribers", "email campaign", "email automation"],
        
        # E-commerce intents
        "store_setup": ["online store", "e-commerce store", "ecommerce store", "sell products", "product listing", "shopping cart", "checkout page"],
        "payment_integration": ["payment gateway", "stripe integration", "paypal integration", "woocommerce payments", "checkout integration", "buy button", "accept payments"],
        
        # Performance/Optimization intents
        "speed_optimization": ["speed up", "pagespeed", "core web vitals", "performance optimization", "load time", "site optimization", "gtmetrix", "lighthouse score", "website slow", "slow loading"],
        "seo_optimization": ["seo optimization", "search engine optimization", "google ranking", "keyword optimization", "organic traffic", "meta tags", "on-page seo"],
        
        # Design/Development intents
        "website_redesign": ["redesign website", "restyle website", "makeover", "new look", "refresh design", "modernize website", "update design"],
        "new_website": ["build website", "create website", "new website", "from scratch", "brand new site", "develop website"],
        "bug_fixes": ["fix bug", "fix issue", "broken", "not working", "website issue", "website problem", "fix error", "repair"],
        "feature_addition": ["add feature", "new feature", "add functionality", "integrate", "custom feature", "enhancement", "extend"],
        
        # Content/Blog intents
        "content_management": ["blog posts", "articles", "content management", "cms setup", "publish content", "editorial"],
        "form_setup": ["contact form", "web form", "form submission", "lead capture form", "cf7", "gravity forms", "formidable forms", "form plugin"],
    }

    # RELATED TASK TYPES - Groups of task types that are semantically similar
    # Used for fuzzy matching when exact task_type doesn't match
    RELATED_TASK_TYPES = {
        # Migration/Transfer cluster
        "content_migration": ["migration", "transfer", "content", "import", "export", "rebuild"],
        "platform_switch": ["migration", "rebuild", "recreate", "convert"],
        
        # Membership cluster
        "membership_setup": ["membership", "subscription", "woocommerce", "paid content", "member pages"],
        "newsletter_email": ["newsletter", "email", "subscription", "mailchimp"],
        
        # E-commerce cluster
        "store_setup": ["complete website", "woocommerce", "shopify", "ecommerce", "store"],
        "payment_integration": ["woocommerce", "enhance functionality", "integration"],
        
        # Performance cluster
        "speed_optimization": ["speed optimization", "optimization", "performance", "pagespeed"],
        "seo_optimization": ["seo", "optimization", "marketing"],
        
        # Development cluster - COMPLETE WEBSITE is a primary cluster
        "website_redesign": ["redesign website", "restyle", "design", "redesign"],
        "new_website": ["complete website", "new website", "build", "build website", "website development", "develop website"],
        "bug_fixes": ["bug fixes", "fixes", "repair", "maintenance", "fix"],
        "feature_addition": ["enhance functionality", "integration", "custom development", "add feature"],
        
        # Content cluster
        "content_management": ["blogs webdesign", "content", "editorial", "blog"],
        "form_setup": ["enhance functionality", "integration", "contact form"],
    }

    @staticmethod
    def extract_client_intents(job_data: Dict[str, Any]) -> List[str]:
        """
        Extract client intents from job description - WHAT they actually want done.
        
        This is CRITICAL for matching jobs by actual requirement, not just platform.
        For example, "migrate Substack to WordPress" should match projects that did
        content migration or membership setup, NOT speed optimization.
        
        Args:
            job_data: Job data dictionary with job_description, job_title, skills_required
            
        Returns:
            List of detected intent categories (e.g., ["content_migration", "membership_setup"])
        """
        job_desc = job_data.get("job_description", "").lower()
        job_title = job_data.get("job_title", "").lower()
        skills = " ".join(s.lower() for s in job_data.get("skills_required", []))
        
        # Combine all text for searching
        search_text = f"{job_title} {job_desc} {skills}"
        
        detected_intents = []
        intent_scores = {}
        
        for intent, keywords in MetadataExtractor.CLIENT_INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in search_text:
                    # Weight by keyword length (longer = more specific = more weight)
                    score += len(keyword.split())
            
            if score > 0:
                intent_scores[intent] = score
        
        # Sort by score and return top intents
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            detected_intents = [intent for intent, score in sorted_intents if score >= 1]
            
            logger.debug(f"  Detected intents: {detected_intents[:5]} from scores {intent_scores}")
        
        return detected_intents[:5]  # Return top 5 intents

    @staticmethod
    def get_intent_similarity(intents1: List[str], intents2: List[str]) -> float:
        """
        Calculate similarity between two sets of client intents.
        
        Uses Jaccard similarity with bonus for matching primary (first) intent.
        
        Returns:
            Similarity score 0-1
        """
        if not intents1 or not intents2:
            return 0.0
        
        set1 = set(intents1)
        set2 = set(intents2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        base_similarity = intersection / union if union > 0 else 0.0
        
        # Bonus for matching primary intent (first one is most important)
        primary_match_bonus = 0.3 if intents1[0] == intents2[0] else 0.0
        
        return min(1.0, base_similarity + primary_match_bonus)

    @staticmethod
    def get_task_type_similarity(task1: str, task2: str, intents1: List[str] = None, intents2: List[str] = None) -> float:
        """
        Calculate semantic similarity between task types.
        
        Uses related task type clusters for fuzzy matching instead of just exact match.
        If task types don't match exactly, checks if they're semantically related.
        
        Args:
            task1: First task type
            task2: Second task type  
            intents1: Client intents for first job (optional, for fallback matching)
            intents2: Client intents for second job (optional)
            
        Returns:
            Similarity score 0-1
        """
        task1 = str(task1).lower().strip() if task1 else ""
        task2 = str(task2).lower().strip() if task2 else ""
        
        # Exact match = perfect similarity
        if task1 and task2 and task1 == task2:
            return 1.0
        
        # Check if tasks are in the same semantic cluster
        for intent, related_tasks in MetadataExtractor.RELATED_TASK_TYPES.items():
            related_lower = [t.lower() for t in related_tasks]
            
            # Check if both tasks are in same cluster
            task1_in_cluster = any(task1 in rt or rt in task1 for rt in related_lower) if task1 else False
            task2_in_cluster = any(task2 in rt or rt in task2 for rt in related_lower) if task2 else False
            
            if task1_in_cluster and task2_in_cluster:
                logger.debug(f"  Task type fuzzy match: '{task1}' ~ '{task2}' via cluster '{intent}'")
                return 0.7  # Semantic match = 70% similarity
        
        # Fallback: Check if intents overlap (even if task types don't)
        if intents1 and intents2:
            intent_sim = MetadataExtractor.get_intent_similarity(intents1, intents2)
            if intent_sim > 0.3:
                logger.debug(f"  Intent-based match: {intents1[:2]} ~ {intents2[:2]} = {intent_sim}")
                return intent_sim * 0.6  # Intent match weighted at 60%
        
        return 0.0  # No match

    @staticmethod
    def extract_all_metadata(job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all metadata dimensions from job data.
        
        Args:
            job_data: Complete job data dictionary
            
        Returns:
            Enriched job_data with computed metadata
        """
        job_id = job_data.get("contract_id", "unknown")
        logger.debug(f"[Metadata] Extracting metadata for {job_id}")

        # Extract all dimensions
        job_data["proposal_effectiveness_score"] = MetadataExtractor.compute_effectiveness_score(job_data)
        job_data["client_satisfaction"] = MetadataExtractor.extract_satisfaction_score(job_data)
        job_data["task_complexity"] = MetadataExtractor.assess_complexity(job_data)
        job_data["industry_tags"] = MetadataExtractor.extract_industry_tags(job_data)
        job_data["project_duration_days"] = MetadataExtractor.calculate_duration(job_data)
        job_data["reusable_sections"] = MetadataExtractor.identify_reusable_sections(job_data)
        
        # NEW: Extract client intents
        job_data["client_intents"] = MetadataExtractor.extract_client_intents(job_data)

        logger.debug(f"✓ Metadata extracted: effectiveness={job_data.get('proposal_effectiveness_score')}, "
                    f"satisfaction={job_data.get('client_satisfaction')}, "
                    f"complexity={job_data.get('task_complexity')}, "
                    f"intents={job_data.get('client_intents', [])[:2]}")

        return job_data

        return job_data

    @staticmethod
    def compute_effectiveness_score(job_data: Dict[str, Any]) -> float:
        """
        Compute proposal effectiveness score (0-1) based on:
        - Project completion status
        - Client feedback sentiment
        - Feedback length (more detailed = more positive engagement)
        
        Returns:
            Effectiveness score 0-1
        """
        score = 0.0

        # Base score on completion
        if job_data.get("project_status") == "completed":
            score += 0.5
        elif job_data.get("project_status") == "ongoing":
            score += 0.25

        # Add based on feedback
        feedback_text = (job_data.get("client_feedback_text") or 
                        job_data.get("client_feedback") or "")
        
        if feedback_text:
            # Longer feedback = more engagement
            feedback_length = len(feedback_text.split())
            if feedback_length > 100:
                score += 0.25
            elif feedback_length > 50:
                score += 0.15
            elif feedback_length > 20:
                score += 0.05

            # Sentiment analysis
            sentiment = MetadataExtractor._analyze_sentiment(feedback_text)
            if sentiment == "positive":
                score += 0.2
            elif sentiment == "negative":
                score -= 0.2

        # Normalize to 0-1
        return max(0.0, min(1.0, score))

    @staticmethod
    def extract_satisfaction_score(job_data: Dict[str, Any]) -> Optional[float]:
        """
        Extract or infer client satisfaction (1-5 scale) from feedback.
        
        Returns:
            Satisfaction score 1-5 or None
        """
        feedback_text = (job_data.get("client_feedback_text") or 
                        job_data.get("client_feedback") or "")
        
        if not feedback_text:
            return None

        # Look for explicit ratings (★, 5 stars, etc.)
        rating_match = re.search(r'(\d+)\s*(?:\/5|out of 5|stars?)', feedback_text, re.IGNORECASE)
        if rating_match:
            try:
                return float(rating_match.group(1))
            except:
                pass

        # Infer from sentiment
        sentiment = MetadataExtractor._analyze_sentiment(feedback_text)
        if sentiment == "positive":
            return 4.5
        elif sentiment == "neutral":
            return 3.0
        elif sentiment == "negative":
            return 2.0
        
        return 3.5  # Default neutral

    @staticmethod
    def assess_complexity(job_data: Dict[str, Any]) -> str:
        """
        Assess task complexity (low, medium, high) based on:
        - Skills required count
        - Complexity keywords in description
        - Task type
        
        Returns:
            Complexity level: 'low', 'medium', or 'high'
        """
        job_desc = job_data.get("job_description", "").lower()
        skills = job_data.get("skills_required", [])
        task_type = job_data.get("task_type", "").lower()

        score = 0

        # High complexity keywords
        for keyword in MetadataExtractor.COMPLEXITY_INDICATORS["high"]:
            if keyword in job_desc:
                score += 3

        # Medium complexity keywords
        for keyword in MetadataExtractor.COMPLEXITY_INDICATORS["medium"]:
            if keyword in job_desc:
                score += 2

        # Low complexity keywords
        for keyword in MetadataExtractor.COMPLEXITY_INDICATORS["low"]:
            if keyword in job_desc:
                score += 1

        # Factor in skills count
        score += len(skills)

        # Map score to complexity
        if score >= 10:
            return "high"
        elif score >= 5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def extract_industry_tags(job_data: Dict[str, Any]) -> List[str]:
        """
        Extract industry tags from industry field and description.
        
        Returns:
            List of standardized industry tags
        """
        tags = set()

        industry = job_data.get("industry", "").lower()
        job_desc = job_data.get("job_description", "").lower()
        company_name = job_data.get("company_name", "").lower()

        text = f"{industry} {job_desc} {company_name}".lower()

        # Match against known industries
        for industry_tag, keywords in MetadataExtractor.INDUSTRY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    tags.add(industry_tag)
                    break

        # Add raw industry if provided
        if industry and industry != "general":
            tags.add(industry)

        return list(tags) if tags else ["general"]

    @staticmethod
    def calculate_duration(job_data: Dict[str, Any]) -> Optional[int]:
        """
        Calculate project duration in days from start and end dates.
        
        Returns:
            Duration in days or None
        """
        start_date = job_data.get("start_date")
        end_date = job_data.get("end_date")

        if not (start_date and end_date):
            return None

        try:
            from datetime import datetime as dt
            start = dt.fromisoformat(start_date.replace('Z', '+00:00'))
            end = dt.fromisoformat(end_date.replace('Z', '+00:00'))
            duration = (end - start).days
            return duration if duration > 0 else None
        except Exception as e:
            logger.debug(f"Could not parse dates: {e}")
            return None

    @staticmethod
    def identify_reusable_sections(job_data: Dict[str, Any]) -> List[str]:
        """
        Identify sections from the proposal that could be reused in future proposals.
        
        This analyzes the proposal text and feedback to find what worked.
        
        Returns:
            List of reusable section descriptions
        """
        sections = []
        proposal = job_data.get("your_proposal_text", "").lower()
        feedback = (job_data.get("client_feedback_text") or 
                   job_data.get("client_feedback") or "").lower()

        # Check if proposal has common winning sections
        if "timeline" in proposal or "schedule" in feedback and "great" in feedback:
            sections.append("detailed_timeline")

        if "approach" in proposal or "method" in feedback and "impressed" in feedback:
            sections.append("technical_approach")

        if "experience" in proposal or "portfolio" in proposal:
            if "perfect" in feedback or "impressed" in feedback:
                sections.append("portfolio_reference")

        if "communication" in proposal or "responsive" in feedback or "responsive" in feedback:
            sections.append("communication_plan")

        if "risk" in proposal or "challenges" in proposal:
            sections.append("risk_mitigation")

        if "budget" in proposal or "price" in proposal or "cost" in proposal:
            if "reasonable" in feedback or "fair" in feedback or "good value" in feedback:
                sections.append("pricing_justification")

        # Identify specific patterns that got praise
        sentiment = MetadataExtractor._analyze_sentiment(feedback)
        if sentiment == "positive":
            sections.append("winning_pattern")  # General positive pattern

        return sections if sections else []

    @staticmethod
    def _analyze_sentiment(text: str) -> str:
        """
        Simple sentiment analysis using keyword matching.
        
        Returns:
            'positive', 'negative', or 'neutral'
        """
        positive_words = [
            'excellent', 'great', 'amazing', 'wonderful', 'fantastic',
            'love', 'awesome', 'perfect', 'professional', 'highly',
            'recommend', 'best', 'impressed', 'satisfied', 'happy',
            'great work', 'well done', 'impressed', 'on time'
        ]

        negative_words = [
            'poor', 'bad', 'terrible', 'awful', 'horrible',
            'hate', 'disappointed', 'issues', 'problem', 'broken',
            'late', 'slow', 'regret', 'unsatisfied', 'poor quality'
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
    def compare_projects(
        job1: Dict[str, Any],
        job2: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare two projects across multiple dimensions including CLIENT INTENT.
        
        CRITICAL: This now uses semantic task type matching and client intent
        similarity to find projects that did SIMILAR WORK, not just same platform.
        
        For example: "Substack migration" should match projects that did:
        - Content migration/transfer
        - Membership/subscription setup
        - Newsletter integration
        NOT just any WordPress project like "speed optimization"
        
        Returns:
            Similarity scores for each dimension (0-1)
        """
        similarity = {}

        # Extract client intents for both jobs
        intents1 = job1.get("client_intents") or MetadataExtractor.extract_client_intents(job1)
        intents2 = job2.get("client_intents") or MetadataExtractor.extract_client_intents(job2)

        # CLIENT INTENT similarity - HIGHEST PRIORITY
        # This captures WHAT the client actually wants done
        similarity["intent"] = MetadataExtractor.get_intent_similarity(intents1, intents2)

        # Industry similarity
        industries1 = job1.get("industry_tags") or []
        industries2 = job2.get("industry_tags") or []
        industries1 = set(industries1) if isinstance(industries1, list) else set()
        industries2 = set(industries2) if isinstance(industries2, list) else set()
        if industries1 or industries2:
            intersection = len(industries1 & industries2)
            union = len(industries1 | industries2)
            similarity["industry"] = intersection / union if union > 0 else 0.0
        else:
            similarity["industry"] = 0.0

        # Skills similarity
        skills1 = set(s.lower() for s in job1.get("skills_required", []))
        skills2 = set(s.lower() for s in job2.get("skills_required", []))
        if skills1 or skills2:
            intersection = len(skills1 & skills2)
            union = len(skills1 | skills2)
            similarity["skills"] = intersection / union if union > 0 else 0.0
        else:
            similarity["skills"] = 0.0

        # Task type similarity - NOW USES SEMANTIC MATCHING
        job1_task = str(job1.get("task_type", "")).lower() if job1.get("task_type") else ""
        job2_task = str(job2.get("task_type", "")).lower() if job2.get("task_type") else ""
        similarity["task_type"] = MetadataExtractor.get_task_type_similarity(
            job1_task, job2_task, intents1, intents2
        )

        # Complexity similarity (exact match)
        if job1.get("task_complexity") == job2.get("task_complexity"):
            similarity["complexity"] = 1.0
        else:
            similarity["complexity"] = 0.5 if (
                job1.get("task_complexity") in ["low", "medium", "high"] and
                job2.get("task_complexity") in ["low", "medium", "high"]
            ) else 0.0

        # Duration similarity (within 50%)
        duration1 = job1.get("project_duration_days")
        duration2 = job2.get("project_duration_days")
        if duration1 and duration2:
            ratio = min(duration1, duration2) / max(duration1, duration2)
            similarity["duration"] = ratio
        else:
            similarity["duration"] = 0.5 if duration1 == duration2 else 0.0

        return similarity

    @staticmethod
    def calculate_overall_similarity(similarity_scores: Dict[str, float]) -> float:
        """
        Calculate overall similarity from individual dimension scores.
        
        CRITICAL CHANGE: Now includes CLIENT INTENT as the HIGHEST weighted factor.
        Intent captures WHAT the client wants done (migration, membership, speed, etc.)
        which is more important than just platform or skill match.
        
        Weights prioritize:
        1. CLIENT INTENT (what they want done) - 35%
        2. TASK TYPE (type of work) - 25%
        3. SKILLS (technical overlap) - 20%
        4. INDUSTRY (domain relevance) - 10%
        5. COMPLEXITY/DURATION - 10%
        """
        weights = {
            "intent": 0.35,         # HIGHEST: What the client actually wants done
            "task_type": 0.25,      # HIGH: Type of work (now with semantic matching)
            "skills": 0.20,         # Medium: Skill overlap
            "industry": 0.10,       # Lower: Industry relevance
            "complexity": 0.05,     # Minor: Complexity match
            "duration": 0.05        # Minor: Duration reference
        }

        total_score = 0.0
        total_weight = 0.0

        for dimension, score in similarity_scores.items():
            weight = weights.get(dimension, 0.05)
            total_score += score * weight
            total_weight += weight

        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Log when intent makes a big difference
        intent_score = similarity_scores.get("intent", 0)
        task_score = similarity_scores.get("task_type", 0)
        if intent_score > 0.5 or task_score > 0.5:
            logger.debug(f"  Similarity breakdown: intent={intent_score:.2f}, task={task_score:.2f}, overall={final_score:.2f}")
        
        return final_score

    @staticmethod
    def rank_similar_projects(
        reference_job: Dict[str, Any],
        all_jobs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rank all jobs by similarity to reference job.
        
        Returns:
            List of (contract_id, similarity_score) tuples, sorted by score descending
        """
        rankings = []

        for job in all_jobs:
            if job.get("contract_id") == reference_job.get("contract_id"):
                continue  # Skip self-comparison

            similarity_scores = MetadataExtractor.compare_projects(reference_job, job)
            overall_similarity = MetadataExtractor.calculate_overall_similarity(similarity_scores)
            
            rankings.append((job.get("contract_id"), overall_similarity))

        # Sort by similarity descending
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings[:top_k] if top_k else rankings

    @staticmethod
    def generate_metadata_summary(job_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of extracted metadata.
        
        Returns:
            Summary string
        """
        summary = f"""
Metadata Summary for {job_data.get('contract_id')}:

📊 Effectiveness Metrics:
  - Proposal Effectiveness: {job_data.get('proposal_effectiveness_score', 0):.0%}
  - Client Satisfaction: {job_data.get('client_satisfaction', 0)}/5
  - Project Status: {job_data.get('project_status', 'unknown')}

🎯 Task Profile:
  - Complexity: {job_data.get('task_complexity', 'unknown')}
  - Task Type: {job_data.get('task_type', 'unknown')}
  - Duration: {job_data.get('project_duration_days', 'unknown')} days
  - Industries: {', '.join(job_data.get('industry_tags', []))}

💡 Reusable Elements:
  - Sections: {', '.join(job_data.get('reusable_sections', [])) if job_data.get('reusable_sections') else 'None identified'}

🛠️ Skills Required:
  - {', '.join(job_data.get('skills_required', []))}
"""
        return summary
