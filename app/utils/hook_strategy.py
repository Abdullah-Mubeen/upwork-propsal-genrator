"""
Hook Strategy System for Winning Proposals

This module creates VARIED, COMPELLING opening hooks that force clients to click.
On Upwork, only ~2.5 lines are visible in preview - making the hook critical.

Key Principles (learned from 20+ winning proposals):
1. DON'T always start with "I see you're dealing with..."
2. Mirror the client's TONE and ENERGY from their job post
3. Lead with VALUE or CURIOSITY, not self-introduction
4. Reference something SPECIFIC from their post (shows you read it)
5. Create urgency or intrigue in the first line

Hook Categories (based on job sentiment analysis):
- URGENCY: Client sounds stressed, needs fast help → Lead with availability/speed
- TECHNICAL: Specific tech challenge → Lead with technical insight/solution hint
- FRUSTRATED: Bad past experience → Lead with empathy + differentiation  
- CURIOUS: Exploring options → Lead with unique value proposition
- BUSINESS_IMPACT: Revenue/conversion focus → Lead with ROI/results
- COLLABORATIVE: Looking for partner → Lead with questions showing interest
"""

import logging
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class JobSentiment(str, Enum):
    """Detected sentiment/mood from job description"""
    URGENT = "urgent"               # Needs it NOW, stressed, time pressure
    FRUSTRATED = "frustrated"       # Bad past experience, something broken
    TECHNICAL = "technical"         # Detailed technical requirements, specs-focused
    CURIOUS = "curious"             # Exploring options, not urgent
    BUSINESS_FOCUS = "business"     # Revenue, conversions, ROI focused
    COLLABORATIVE = "collaborative" # Looking for ongoing partner
    CASUAL = "casual"               # Relaxed, simple request
    PROFESSIONAL = "professional"   # Formal, enterprise-level


class JobIntent(str, Enum):
    """What the client actually wants done"""
    FIX_PROBLEM = "fix"            # Something is broken/not working
    BUILD_NEW = "build"            # Create something from scratch
    IMPROVE = "improve"            # Enhance existing (speed, design, features)
    MIGRATE = "migrate"            # Move from one platform/system to another
    MAINTAIN = "maintain"          # Ongoing support/maintenance
    CONSULT = "consult"            # Need advice/strategy, not just execution


@dataclass
class JobAnalysis:
    """Complete analysis of a job posting"""
    sentiment: JobSentiment
    intent: JobIntent
    urgency_level: int  # 1-5 (5 = critical)
    pain_points: List[str]
    specific_details: List[str]  # Numbers, tools, deadlines mentioned
    tone_words: List[str]  # Key emotional words from the post
    platform: str  # WordPress, Shopify, etc.
    task_type: str  # speed optimization, bug fix, etc.
    hook_strategy: str  # Recommended hook approach


class HookStrategyEngine:
    """
    Generates varied, compelling hooks based on job analysis.
    
    The goal: Create hooks so good that clients MUST click to read more.
    Only ~2.5 lines visible on Upwork = every word matters.
    """

    # ============== WINNING PATTERNS FROM REAL PROPOSALS ==============
    # These are actual opening patterns that won jobs
    
    WINNING_HOOK_PATTERNS = {
        "direct_solution": [
            "I know exactly why {problem} is happening and how to fix it.",
            "{problem}? I've fixed this exact issue {count}+ times.",
            "That {problem} you're seeing - there's a 90% chance it's {likely_cause}.",
            "Good news: {problem} is usually a quick fix. Here's why...",
        ],
        "immediate_value": [
            "Just wrapped up something nearly identical - {portfolio_url}",
            "I literally just solved this last week for another client: {portfolio_url}",
            "Here's a live example of exactly what you need: {portfolio_url}",
            "Quick preview of what I can do for you: {portfolio_url}",
        ],
        "empathy_first": [
            "I know how frustrating {problem} can be when {consequence}.",
            "Sounds like your last developer left you in a tough spot.",
            "That's a stressful situation - let me help you fix it fast.",
            "I get it - {problem} is the worst, especially when {context}.",
        ],
        "question_hook": [
            "Quick question: Is the {issue} affecting {impact}?",
            "Curious - have you checked if {technical_detail}?",
            "When did the {problem} start? That might tell us a lot.",
            "Is this happening on all {context} or just specific ones?",
        ],
        "result_lead": [
            "Got a similar store from {before} to {after} last month.",
            "Took another client's {metric} from {before} to {after} - here's how.",
            "My last {platform} project went from {before} to {after} in {timeframe}.",
            "{metric_improvement} on a similar project last week.",
        ],
        "availability_lead": [
            "I can start right now - this shouldn't take more than {timeframe}.",
            "Free today and this is right up my alley.",
            "At my desk and ready to dive in immediately.",
            "I've got time blocked for urgent projects like this today.",
        ],
        "specific_insight": [
            "Looking at your {context}, the issue is likely {likely_cause}.",
            "Based on what you described, I'd start by checking {technical_detail}.",
            "That {tool} + {issue} combination usually means {insight}.",
            "Interesting challenge - most {platform} sites have this because of {reason}.",
        ],
        "social_proof": [
            "Just got 5 stars on the same type of work yesterday.",
            "Been doing exactly this for {timeframe} - dozens of happy clients.",
            "This is literally 80% of my work. Here's my latest: {portfolio_url}",
            "I specialize in {task_type} - it's what I do best.",
        ],
    }

    # Sentiment-based hook strategy selection
    SENTIMENT_STRATEGIES = {
        JobSentiment.URGENT: ["availability_lead", "direct_solution", "immediate_value"],
        JobSentiment.FRUSTRATED: ["empathy_first", "direct_solution", "social_proof"],
        JobSentiment.TECHNICAL: ["specific_insight", "question_hook", "result_lead"],
        JobSentiment.CURIOUS: ["immediate_value", "result_lead", "question_hook"],
        JobSentiment.BUSINESS_FOCUS: ["result_lead", "immediate_value", "social_proof"],
        JobSentiment.COLLABORATIVE: ["question_hook", "specific_insight", "empathy_first"],
        JobSentiment.CASUAL: ["immediate_value", "availability_lead", "social_proof"],
        JobSentiment.PROFESSIONAL: ["result_lead", "specific_insight", "social_proof"],
    }

    # Urgency detection patterns
    URGENCY_PATTERNS = {
        5: ["emergency", "site down", "not working", "broken", "losing sales", "every hour", "critical", "asap today"],
        4: ["urgent", "asap", "immediately", "rush"],
        3: ["soon", "this week", "deadline", "time-sensitive", "quickly", "fast"],
        2: ["when you can", "flexible timeline"],
        1: ["no rush", "exploring options", "considering", "thinking about", "eventually", "nothing urgent", "not urgent"],
    }

    # Sentiment detection patterns
    SENTIMENT_PATTERNS = {
        JobSentiment.URGENT: ["urgent", "asap", "immediately", "emergency", "critical", "today", "now", "fast", "quickly", "deadline"],
        JobSentiment.FRUSTRATED: ["frustrated", "struggling", "issues", "problems", "doesn't work", "broken", "nightmare", "headache", "last developer", "went mia", "ghosted", "tried before"],
        JobSentiment.TECHNICAL: ["architecture", "api", "database", "algorithm", "optimize", "performance", "stack", "server", "deploy", "integration", "custom code"],
        JobSentiment.BUSINESS_FOCUS: ["revenue", "conversions", "sales", "roi", "customers", "business", "growth", "profit", "traffic"],
        JobSentiment.COLLABORATIVE: ["partner", "long-term", "ongoing", "team", "collaborate", "work together", "relationship"],
        JobSentiment.CASUAL: ["simple", "easy", "quick", "small", "minor", "straightforward", "basic"],
    }

    def __init__(self):
        """Initialize hook strategy engine"""
        logger.info("HookStrategyEngine initialized")

    def analyze_job(self, job_data: Dict[str, Any]) -> JobAnalysis:
        """
        Deeply analyze a job posting to understand client's true needs.
        
        Args:
            job_data: Job data dictionary with job_description, job_title, etc.
            
        Returns:
            JobAnalysis with sentiment, intent, and recommended hook strategy
        """
        job_desc = job_data.get("job_description", "").lower()
        job_title = job_data.get("job_title", "").lower()
        combined_text = f"{job_title} {job_desc}"
        
        # Detect sentiment
        sentiment = self._detect_sentiment(combined_text)
        
        # Detect intent
        intent = self._detect_intent(combined_text)
        
        # Detect urgency level (1-5)
        urgency = self._detect_urgency(combined_text)
        
        # Extract pain points
        pain_points = self._extract_pain_points(combined_text)
        
        # Extract specific details (numbers, tools, etc.)
        specific_details = self._extract_specific_details(job_data)
        
        # Get tone words
        tone_words = self._extract_tone_words(combined_text)
        
        # Detect platform
        platform = self._detect_platform(combined_text, job_data.get("skills_required", []))
        
        # Get task type
        task_type = job_data.get("task_type", self._detect_task_type(combined_text))
        
        # Determine best hook strategy
        hook_strategy = self._select_hook_strategy(sentiment, urgency)
        
        analysis = JobAnalysis(
            sentiment=sentiment,
            intent=intent,
            urgency_level=urgency,
            pain_points=pain_points,
            specific_details=specific_details,
            tone_words=tone_words,
            platform=platform,
            task_type=task_type,
            hook_strategy=hook_strategy
        )
        
        logger.info(f"[JobAnalysis] Sentiment: {sentiment.value}, Intent: {intent.value}, "
                   f"Urgency: {urgency}/5, Strategy: {hook_strategy}")
        
        return analysis

    def generate_hook(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        portfolio_url: Optional[str] = None
    ) -> str:
        """
        Generate a compelling, varied hook based on job analysis.
        
        Args:
            job_analysis: Analysis results from analyze_job()
            job_data: Original job data
            similar_projects: Similar past projects for references
            portfolio_url: Best portfolio URL to include
            
        Returns:
            Compelling hook string (1-3 sentences)
        """
        strategy = job_analysis.hook_strategy
        templates = self.WINNING_HOOK_PATTERNS.get(strategy, self.WINNING_HOOK_PATTERNS["direct_solution"])
        
        # Select template randomly for variety
        template = random.choice(templates)
        
        # Build context for template filling
        context = self._build_hook_context(job_analysis, job_data, similar_projects, portfolio_url)
        
        # Fill in the template
        try:
            hook = self._fill_template(template, context)
        except Exception as e:
            logger.warning(f"Template fill failed, using fallback: {e}")
            hook = self._generate_fallback_hook(job_analysis, context)
        
        # Add portfolio link if we have one and it's not already in hook
        if portfolio_url and portfolio_url not in hook and "{portfolio_url}" not in template:
            hook = f"{hook}\n\nCheck out a similar project: {portfolio_url}"
        
        return hook

    def get_hook_variations(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        portfolio_url: Optional[str] = None,
        count: int = 3
    ) -> List[str]:
        """
        Generate multiple hook variations to choose from.
        
        Returns:
            List of different hook options
        """
        hooks = []
        used_strategies = set()
        
        # Get primary strategy and alternates
        primary_strategy = job_analysis.hook_strategy
        alternate_strategies = self.SENTIMENT_STRATEGIES.get(
            job_analysis.sentiment, 
            ["direct_solution", "immediate_value", "empathy_first"]
        )
        
        all_strategies = [primary_strategy] + [s for s in alternate_strategies if s != primary_strategy]
        
        context = self._build_hook_context(job_analysis, job_data, similar_projects, portfolio_url)
        
        # If no portfolio URL, prioritize strategies that don't require it
        if not portfolio_url:
            no_url_strategies = ["direct_solution", "empathy_first", "question_hook", "availability_lead", "specific_insight"]
            all_strategies = [s for s in all_strategies if s in no_url_strategies] + \
                            [s for s in all_strategies if s not in no_url_strategies]
        
        for strategy in all_strategies[:count + 2]:  # Try a few extra to get enough valid ones
            if len(hooks) >= count:
                break
            if strategy in used_strategies:
                continue
            used_strategies.add(strategy)
            
            templates = self.WINNING_HOOK_PATTERNS.get(strategy, self.WINNING_HOOK_PATTERNS["direct_solution"])
            
            # Filter out templates that need portfolio_url if we don't have one
            if not portfolio_url:
                templates = [t for t in templates if "{portfolio_url}" not in t]
                if not templates:
                    continue
            
            template = random.choice(templates)
            
            try:
                hook = self._fill_template(template, context)
                # Skip hooks that still have unfilled placeholders or end with empty content
                if "{" not in hook and not hook.endswith(": ") and hook.strip():
                    hooks.append(hook)
            except:
                continue
        
        return hooks if hooks else [self._generate_fallback_hook(job_analysis, context)]

    def _detect_sentiment(self, text: str) -> JobSentiment:
        """Detect the primary sentiment/mood from job text"""
        sentiment_scores = {}
        
        for sentiment, patterns in self.SENTIMENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                sentiment_scores[sentiment] = score
        
        if not sentiment_scores:
            return JobSentiment.PROFESSIONAL  # Default
        
        # Return highest scoring sentiment
        return max(sentiment_scores, key=sentiment_scores.get)

    def _detect_intent(self, text: str) -> JobIntent:
        """Detect what the client actually wants done"""
        intent_patterns = {
            JobIntent.FIX_PROBLEM: ["fix", "broken", "not working", "error", "issue", "problem", "bug", "repair"],
            JobIntent.BUILD_NEW: ["build", "create", "new", "develop", "make", "from scratch", "design"],
            JobIntent.IMPROVE: ["improve", "optimize", "enhance", "upgrade", "better", "faster", "increase"],
            JobIntent.MIGRATE: ["migrate", "transfer", "move", "import", "export", "switch", "convert"],
            JobIntent.MAINTAIN: ["maintain", "support", "ongoing", "regular", "manage", "update"],
            JobIntent.CONSULT: ["advise", "consult", "strategy", "review", "audit", "recommend", "assess"],
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return JobIntent.BUILD_NEW  # Default
        
        return max(intent_scores, key=intent_scores.get)

    def _detect_urgency(self, text: str) -> int:
        """Detect urgency level (1-5)"""
        # Check for explicit "no rush" / "not urgent" first - these override other signals
        no_rush_patterns = ["no rush", "not urgent", "nothing urgent", "no hurry", "take your time", "flexible"]
        if any(p in text for p in no_rush_patterns):
            return 1
        
        for level, patterns in sorted(self.URGENCY_PATTERNS.items(), reverse=True):
            if any(p in text for p in patterns):
                return level
        return 2  # Default: normal priority

    def _extract_pain_points(self, text: str) -> List[str]:
        """Extract specific pain points mentioned"""
        pain_points = []
        
        pain_indicators = [
            (r"(\w+)\s+(?:is|are)\s+(?:broken|not working|slow)", "issue"),
            (r"(?:losing|lost)\s+(\w+)", "loss"),
            (r"(?:can't|cannot)\s+(\w+)", "blocker"),
            (r"(?:frustrated|struggling)\s+with\s+(\w+)", "frustration"),
        ]
        
        for pattern, category in pain_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                pain_points.append(f"{category}: {match}")
        
        return pain_points[:3]  # Limit to top 3

    def _extract_specific_details(self, job_data: Dict[str, Any]) -> List[str]:
        """Extract specific numbers, tools, metrics from job"""
        details = []
        text = f"{job_data.get('job_title', '')} {job_data.get('job_description', '')}"
        
        # Numbers with context
        number_patterns = re.findall(
            r'(\d+[\+]?\s*(?:subscribers?|products?|pages?|seconds?|ms|visitors?|items?|%|hours?|days?|users?))',
            text, re.IGNORECASE
        )
        details.extend(number_patterns[:3])
        
        # Tool/platform mentions
        tools = ["wordpress", "shopify", "woocommerce", "elementor", "substack", "mailchimp", 
                 "stripe", "paypal", "gempages", "liquid", "php", "javascript", "react"]
        for tool in tools:
            if tool in text.lower():
                details.append(tool.title())
        
        # Speed metrics
        speed_patterns = re.findall(r'(\d+[\.\d]*\s*(?:seconds?|ms|s)\s*(?:load|time)?)', text, re.IGNORECASE)
        details.extend(speed_patterns[:2])
        
        return details[:5]  # Limit to 5 most important

    def _extract_tone_words(self, text: str) -> List[str]:
        """Extract emotional/tone words"""
        tone_words = []
        
        positive = ["excited", "love", "great", "awesome", "amazing", "perfect", "excellent"]
        negative = ["frustrated", "angry", "disappointed", "struggling", "nightmare", "horrible"]
        urgent = ["urgent", "asap", "immediately", "critical", "emergency", "rush"]
        
        for word in positive + negative + urgent:
            if word in text:
                tone_words.append(word)
        
        return tone_words[:5]

    def _detect_platform(self, text: str, skills: List[str]) -> str:
        """Detect the primary platform/technology"""
        platforms = {
            "wordpress": ["wordpress", "wp", "elementor", "divi", "wpbakery", "geo directory", "geodirectory"],
            "shopify": ["shopify", "liquid", "gempages", "shopify store"],
            "woocommerce": ["woocommerce", "woo commerce"],
            "wix": ["wix", "wix site", "wix website"],
            "webflow": ["webflow"],
            "squarespace": ["squarespace"],
            "custom": ["react", "node", "python", "django", "flask", "next.js"],
        }
        
        combined = f"{text} {' '.join(skills)}".lower()
        
        for platform, indicators in platforms.items():
            if any(ind in combined for ind in indicators):
                return platform
        
        return "general"

    def _detect_task_type(self, text: str) -> str:
        """Detect the type of task"""
        task_patterns = {
            "speed optimization": ["speed", "pagespeed", "core web vitals", "performance", "optimize", "slow", "loading"],
            "bug fixes": ["fix", "bug", "broken", "error", "issue", "not working", "repair"],
            "complete website": ["build", "create", "new website", "from scratch", "design"],
            "migration": ["migrate", "transfer", "move", "import", "convert"],
            "enhance functionality": ["add feature", "enhance", "integrate", "custom", "functionality"],
            "redesign": ["redesign", "refresh", "makeover", "new look", "modernize"],
        }
        
        for task_type, patterns in task_patterns.items():
            if any(p in text for p in patterns):
                return task_type
        
        return "general"

    def _select_hook_strategy(self, sentiment: JobSentiment, urgency: int) -> str:
        """Select the best hook strategy based on analysis"""
        # High urgency always leads with availability
        if urgency >= 4:
            return "availability_lead"
        
        # Use sentiment-based strategy selection
        strategies = self.SENTIMENT_STRATEGIES.get(sentiment, ["direct_solution"])
        return strategies[0]  # Return primary strategy for sentiment

    def _build_hook_context(
        self,
        job_analysis: JobAnalysis,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        portfolio_url: Optional[str]
    ) -> Dict[str, str]:
        """Build context dictionary for template filling"""
        job_desc = job_data.get("job_description", "")
        
        # Extract the main problem
        problem = self._extract_main_problem(job_desc, job_analysis)
        
        # Get consequence of problem
        consequence = self._infer_consequence(job_analysis)
        
        # Get likely cause for technical issues
        likely_cause = self._suggest_likely_cause(job_analysis, job_data)
        
        # Get metrics from similar projects
        metrics = self._get_project_metrics(similar_projects)
        
        context = {
            "problem": problem,
            "consequence": consequence,
            "likely_cause": likely_cause,
            "portfolio_url": portfolio_url or "",
            "platform": job_analysis.platform.title(),
            "task_type": job_analysis.task_type,
            "timeframe": self._estimate_timeframe(job_analysis),
            "count": str(random.randint(10, 50)),
            "before": metrics.get("before", "30s"),
            "after": metrics.get("after", "2s"),
            "metric": metrics.get("metric", "load time"),
            "metric_improvement": metrics.get("improvement", "90+ mobile score"),
            "issue": self._extract_specific_issue(job_analysis),
            "impact": self._extract_business_impact(job_data),
            "technical_detail": self._get_technical_detail(job_analysis),
            "context": job_analysis.specific_details[0] if job_analysis.specific_details else "your site",
            "insight": likely_cause,
            "tool": job_analysis.platform.title(),
            "reason": likely_cause,
        }
        
        return context

    def _extract_main_problem(self, job_desc: str, analysis: JobAnalysis) -> str:
        """Extract the main problem being described"""
        # First check specific details
        if analysis.specific_details:
            for detail in analysis.specific_details:
                if any(word in detail.lower() for word in ["slow", "seconds", "not working", "broken"]):
                    return detail
        
        # Check for common problem phrases
        problem_patterns = [
            r"(\w+\s+(?:is|are)\s+(?:not working|broken|slow|failing))",
            r"((?:slow|poor|bad)\s+\w+)",
            r"(need\s+to\s+(?:fix|repair|improve)\s+\w+)",
        ]
        
        for pattern in problem_patterns:
            match = re.search(pattern, job_desc, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback to task type
        return f"{analysis.task_type}" if analysis.task_type != "general" else "this issue"

    def _infer_consequence(self, analysis: JobAnalysis) -> str:
        """Infer the consequence of the problem"""
        consequences = {
            JobIntent.FIX_PROBLEM: "you're losing potential customers",
            JobIntent.IMPROVE: "it's affecting your conversions",
            JobIntent.BUILD_NEW: "you need this to launch",
            JobIntent.MIGRATE: "your content is stuck on the old platform",
            JobSentiment.URGENT: "every hour counts",
            JobSentiment.BUSINESS_FOCUS: "it's impacting your bottom line",
        }
        
        if analysis.sentiment == JobSentiment.URGENT:
            return consequences[JobSentiment.URGENT]
        
        return consequences.get(analysis.intent, "it needs to get fixed")

    def _suggest_likely_cause(self, analysis: JobAnalysis, job_data: Dict[str, Any]) -> str:
        """Suggest a likely cause for the issue"""
        causes = {
            "speed optimization": "unoptimized images or render-blocking JS",
            "bug fixes": "a plugin conflict or theme issue",
            "migration": "different data structures between platforms",
            "enhance functionality": "the current setup's limitations",
        }
        
        return causes.get(analysis.task_type, "something in the current setup")

    def _get_project_metrics(self, similar_projects: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract metrics from similar projects for social proof"""
        defaults = {
            "before": "30+ seconds",
            "after": "under 2s",
            "metric": "load time",
            "improvement": "90+ mobile score"
        }
        
        if not similar_projects:
            return defaults
        
        # Try to find real metrics from project feedback
        for proj in similar_projects[:3]:
            feedback = proj.get("client_feedback_text", "")
            if feedback:
                # Look for numbers in feedback
                numbers = re.findall(r'\d+[\+]?(?:\s*%|\s*seconds?|\s*/\s*5)?', feedback)
                if numbers:
                    defaults["improvement"] = numbers[0]
                    break
        
        return defaults

    def _estimate_timeframe(self, analysis: JobAnalysis) -> str:
        """Estimate timeframe based on task type"""
        timeframes = {
            "bug fixes": "a few hours",
            "speed optimization": "1-2 days",
            "enhance functionality": "2-3 days",
            "migration": "3-5 days",
            "complete website": "1-2 weeks",
            "redesign": "1-2 weeks",
        }
        return timeframes.get(analysis.task_type, "a few days")

    def _extract_specific_issue(self, analysis: JobAnalysis) -> str:
        """Extract a specific issue to reference"""
        if analysis.specific_details:
            return analysis.specific_details[0]
        return analysis.task_type

    def _extract_business_impact(self, job_data: Dict[str, Any]) -> str:
        """Extract business impact mentioned"""
        desc = job_data.get("job_description", "").lower()
        
        impacts = {
            "sales": "your sales numbers",
            "conversion": "your conversion rate",
            "customers": "customer experience",
            "revenue": "your revenue",
            "traffic": "your traffic",
        }
        
        for keyword, impact in impacts.items():
            if keyword in desc:
                return impact
        
        return "your business"

    def _get_technical_detail(self, analysis: JobAnalysis) -> str:
        """Get a technical detail to reference"""
        details = {
            "wordpress": "your theme's functions.php",
            "shopify": "your Liquid templates",
            "speed optimization": "your Core Web Vitals",
            "bug fixes": "your browser console",
        }
        
        return details.get(analysis.platform, details.get(analysis.task_type, "the code"))

    def _fill_template(self, template: str, context: Dict[str, str]) -> str:
        """Fill a template with context values"""
        result = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                # Skip if value is empty and it's a URL placeholder
                if not value and key in ['portfolio_url', 'feedback_url']:
                    # Replace with a fallback message or remove the line
                    result = result.replace(placeholder, "[portfolio link]")
                else:
                    result = result.replace(placeholder, str(value))
        return result

    def _generate_fallback_hook(self, analysis: JobAnalysis, context: Dict[str, str]) -> str:
        """Generate a safe fallback hook"""
        problem = context.get("problem", "this")
        platform = context.get("platform", "your site")
        
        fallbacks = [
            f"This is exactly the kind of {analysis.task_type} work I specialize in.",
            f"I've solved this exact {platform} issue multiple times.",
            f"Let me show you how I'd approach {problem}.",
        ]
        return random.choice(fallbacks)


# Create singleton instance
_hook_engine: Optional[HookStrategyEngine] = None


def get_hook_engine() -> HookStrategyEngine:
    """Get singleton hook engine instance"""
    global _hook_engine
    if _hook_engine is None:
        _hook_engine = HookStrategyEngine()
    return _hook_engine
