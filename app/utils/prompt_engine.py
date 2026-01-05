"""
Prompt Engine for Intelligent Proposal Generation

Handles:
- Custom prompt building based on style and tone
- Success pattern injection
- Portfolio/feedback URL integration
- Quality scoring and validation
- Iterative improvement suggestions

This separates prompt logic from proposal generation for better maintainability.
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ProposalStyle(str, Enum):
    """Proposal writing styles"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    DATA_DRIVEN = "data_driven"


class ProposalTone(str, Enum):
    """Proposal tones"""
    CONFIDENT = "confident"
    HUMBLE = "humble"
    ENTHUSIASTIC = "enthusiastic"
    ANALYTICAL = "analytical"
    FRIENDLY = "friendly"


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""
    style: ProposalStyle = ProposalStyle.PROFESSIONAL
    tone: ProposalTone = ProposalTone.CONFIDENT
    include_metrics: bool = True
    include_timeline: bool = True
    include_portfolio: bool = True
    include_testimonials: bool = True
    max_word_count: int = 500
    emphasis: str = "expertise"  # expertise, price, speed, quality, innovation


class PromptEngine:
    """
    Intelligent prompt engine for proposal generation.
    
    Generates optimized prompts based on:
    - Job characteristics
    - Historical success patterns
    - Target style and tone
    - Portfolio and feedback data
    """

    # Style-specific instructions
    STYLE_INSTRUCTIONS = {
        ProposalStyle.PROFESSIONAL: """
Your response should be:
- Formal yet warm
- Structured with clear sections
- Emphasis on expertise and credentials
- Professional formatting with proper grammar
- Bullet points for key benefits
- Clear Call-to-Action at end
""",
        ProposalStyle.CASUAL: """
Your response should be:
- Conversational and approachable
- Personal and relatable
- Friendly tone without losing professionalism
- Shorter paragraphs
- Genuine enthusiasm visible
- Direct communication style
""",
        ProposalStyle.TECHNICAL: """
Your response should be:
- Deep technical details
- Architecture discussions
- Technology stack explanation
- Performance metrics and benchmarks
- Technical trade-offs explained
- Implementation approach detailed
""",
        ProposalStyle.CREATIVE: """
Your response should be:
- Engaging and memorable
- Unique positioning
- Creative problem-solving emphasis
- Innovation highlighted
- Visual/metaphorical language
- Storytelling approach
""",
        ProposalStyle.DATA_DRIVEN: """
Your response should be:
- Backed by metrics and data
- Success rate references
- ROI focused
- Performance guarantees
- Historical data cited
- Quantifiable results promised
"""
    }

    # Tone-specific instructions
    TONE_INSTRUCTIONS = {
        ProposalTone.CONFIDENT: "Convey strong confidence in abilities. Use words like 'will deliver', 'proven', 'guaranteed'.",
        ProposalTone.HUMBLE: "Show humility while demonstrating competence. Use words like 'eager to', 'honored to', 'privileged'.",
        ProposalTone.ENTHUSIASTIC: "Show genuine excitement. Use energetic language, exclamation marks, positive energy.",
        ProposalTone.ANALYTICAL: "Focus on facts and analysis. Logical structure, data references, systematic approach.",
        ProposalTone.FRIENDLY: "Warm and approachable. Show personality, use 'we', collaborative language, genuine interest."
    }

    # Pain point keywords that indicate client frustration/urgency
    PAIN_POINT_INDICATORS = {
        "frustration": ["frustrated", "struggling", "can't figure out", "doesn't work", "broken", "not working", "issues", "problems", "nightmare", "headache", "stuck"],
        "urgency": ["urgent", "asap", "immediately", "deadline", "time-sensitive", "need quickly", "fast turnaround", "rush", "critical", "today", "right now", "this week", "costing us", "every hour", "emergency"],
        "previous_failure": ["previous developer", "last freelancer", "didn't work out", "need someone new", "past experience", "tried before", "went mia", "disappeared", "ghosted"],
        "business_impact": ["losing sales", "losing customers", "revenue", "customers complaining", "bad reviews", "conversion", "cart abandonment", "costing us money", "affecting business"],
        "complexity": ["complex", "complicated", "difficult", "challenging", "advanced", "sophisticated"],
        "growth": ["scaling", "growing", "expansion", "more traffic", "increased demand", "outgrown"]
    }

    # Urgency timeline responses - faster promises for urgent jobs
    URGENCY_TIMELINE_PROMISES = {
        "critical": "I can start right now and have this fixed within hours",
        "today": "I'm available today - can jump on this immediately",
        "asap": "Can get started today and have it sorted by tomorrow",
        "this_week": "I can prioritize this and wrap it up in 2-3 days",
        "standard": None  # Use normal timeline
    }

    # Empathy responses for different pain points
    EMPATHY_RESPONSES = {
        "frustration": [
            "I know how frustrating that can be",
            "That sounds really annoying to deal with", 
            "I get it - that's a common pain point",
            "Been there, totally understand the headache"
        ],
        "urgency": [
            "I understand you need this handled quickly - I'm available now",
            "Time pressure is real - I can start immediately",
            "No worries, I can prioritize this and jump on it today",
            "I know every hour counts - let's fix this ASAP",
            "I can see this is urgent - I'm free to start right now"
        ],
        "previous_failure": [
            "Sorry to hear the last experience didn't work out",
            "I understand the hesitation after a bad experience",
            "Let me show you a different approach"
        ],
        "business_impact": [
            "Losing sales is never fun - let's fix that",
            "I know how much that impacts your bottom line",
            "Revenue matters, so let's get this working ASAP"
        ],
        "complexity": [
            "I love a good technical challenge",
            "This is exactly the kind of project I specialize in",
            "Complex projects are where I do my best work"
        ],
        "growth": [
            "Congrats on the growth - exciting problem to have!",
            "Scaling challenges are a good sign",
            "Love helping businesses level up"
        ]
    }

    @staticmethod
    def detect_urgency_level(job_description: str, job_title: str = "") -> str:
        """
        Detect the urgency level of a job to provide appropriate timeline promises.
        
        Returns:
            Urgency level: 'critical', 'today', 'asap', 'this_week', or 'standard'
        """
        text = f"{job_title} {job_description}".lower()
        
        # Critical urgency indicators (hours matter)
        critical_keywords = ["emergency", "critical", "site down", "broken", "not working", "right now", "within hours", "losing sales", "costing us", "every hour"]
        for kw in critical_keywords:
            if kw in text:
                return "critical"
        
        # Today urgency (same day)
        today_keywords = ["today", "immediately", "asap", "urgent help"]
        for kw in today_keywords:
            if kw in text:
                return "today"
        
        # ASAP urgency (within 1-2 days)
        asap_keywords = ["asap", "urgent", "rush", "quick turnaround", "fast", "need quickly"]
        for kw in asap_keywords:
            if kw in text:
                return "asap"
        
        # This week urgency
        week_keywords = ["this week", "deadline", "time-sensitive"]
        for kw in week_keywords:
            if kw in text:
                return "this_week"
        
        return "standard"

    @staticmethod
    def get_urgency_timeline_promise(urgency_level: str) -> Optional[str]:
        """
        Get appropriate timeline promise based on urgency level.
        
        Returns:
            Timeline promise string or None for standard urgency
        """
        return PromptEngine.URGENCY_TIMELINE_PROMISES.get(urgency_level)

    @staticmethod
    def extract_pain_points(job_description: str) -> Dict[str, List[str]]:
        """
        Extract specific pain points and frustrations from the job description.
        
        This helps craft proposals that directly address what's bothering the client.
        
        Returns:
            Dict with pain point categories and matching phrases found
        """
        job_desc_lower = job_description.lower()
        found_pain_points = {}
        
        for category, keywords in PromptEngine.PAIN_POINT_INDICATORS.items():
            matches = []
            for keyword in keywords:
                if keyword in job_desc_lower:
                    # Extract the sentence containing this keyword for context
                    sentences = job_description.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            matches.append(sentence.strip())
                            break
            if matches:
                found_pain_points[category] = matches[:2]  # Max 2 per category
        
        return found_pain_points

    @staticmethod
    def build_empathy_statement(pain_points: Dict[str, List[str]], tone: str = "friendly") -> str:
        """
        Build a natural empathy statement based on detected pain points.
        
        This creates genuine human connection in the proposal opening.
        """
        import random
        
        if not pain_points:
            return ""
        
        # Prioritize business impact and urgency
        priority_order = ["business_impact", "urgency", "frustration", "previous_failure", "growth", "complexity"]
        
        for category in priority_order:
            if category in pain_points:
                responses = PromptEngine.EMPATHY_RESPONSES.get(category, [])
                if responses:
                    return random.choice(responses)
        
        return ""

    @staticmethod  
    def extract_specific_problem(job_description: str, job_title: str) -> str:
        """
        Extract the SPECIFIC problem the client needs solved.
        
        This goes beyond generic "build a website" to "migrate 5000 subscribers from Substack"
        """
        job_desc_lower = job_description.lower()
        
        # Look for specific quantifiable mentions
        import re
        
        # Find numbers + context (e.g., "5000+ subscribers", "2000 products", "30+ seconds")
        number_patterns = re.findall(r'(\d+[\+]?\s*(?:subscribers|products|items|users|customers|posts|articles|pages|seconds|ms|visitors|orders|\%))', job_desc_lower)
        
        # Find specific tool/platform mentions
        tool_mentions = []
        tools = ['substack', 'mailchimp', 'woocommerce', 'shopify', 'wordpress', 'elementor', 'stripe', 'paypal']
        for tool in tools:
            if tool in job_desc_lower:
                tool_mentions.append(tool.title())
        
        # Build specific problem statement
        specific_parts = []
        
        if number_patterns:
            specific_parts.append(number_patterns[0])
        
        if tool_mentions:
            specific_parts.append(f"with {'/'.join(tool_mentions[:2])}")
        
        return ', '.join(specific_parts) if specific_parts else ""

    def build_proposal_prompt(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        success_patterns: List[str],
        style: str = "professional",
        tone: str = "confident",
        max_words: int = 300,
        include_portfolio: bool = True,
        include_feedback: bool = True,
        include_timeline: bool = False,
        timeline_duration: str = None
    ) -> str:
        """
        Build an optimized prompt for proposal generation using HOOK→PROOF→APPROACH→(TIMELINE)→CTA structure.
        
        Key Requirements:
        1. HOOK: 2 sentences acknowledging THEIR specific problem
        2. PROOF: 2-3 past projects with REAL COMPANY NAMES and ACTUAL portfolio links
        3. APPROACH: 3-4 sentences with specific solution for THEIR tech stack
        4. TIMELINE (optional): Only if include_timeline=True, add conversational timeline
        5. CTA: 1 sentence friendly, conversational call-to-action
        
        CRITICAL: Only mention projects if they have actual portfolio links.
        Never suggest portfolio URLs - only use real URLs from historical data.
        
        Args:
            job_data: The new job requirements
            similar_projects: Similar past successful projects (MUST have portfolio_urls)
            success_patterns: Patterns that worked
            style: Writing style
            tone: Proposal tone
            max_words: Target word count (default 300, ideal range 200-350)
            include_portfolio: Include portfolio links?
            include_feedback: Include feedback/testimonials?
            include_timeline: Include timeline section? (default False)
            timeline_duration: Custom timeline (e.g., "2-3 weeks") if include_timeline is True
            
        Returns:
            Optimized prompt for OpenAI
        """
        logger.debug(f"[PromptEngine] Building proposal prompt (style={style}, tone={tone}, target={max_words} words, timeline={include_timeline})")

        # Get style and tone instructions
        style_enum = ProposalStyle(style) if isinstance(style, str) else style
        tone_enum = ProposalTone(tone) if isinstance(tone, str) else tone

        style_guide = self.STYLE_INSTRUCTIONS.get(style_enum, "")
        tone_guide = self.TONE_INSTRUCTIONS.get(tone_enum, "")

        # Build sections
        job_section = self._build_job_section(job_data)
        projects_section = self._build_projects_section(similar_projects, include_portfolio, include_feedback)
        patterns_section = self._build_patterns_section(success_patterns)
        structure_section = self._build_hook_proof_approach_structure(job_data, similar_projects, include_timeline, timeline_duration)

        # Build timeline instruction based on include_timeline flag
        timeline_instruction = ""
        if include_timeline:
            if timeline_duration:
                timeline_instruction = f"7. Include a casual, conversational timeline mention (use: {timeline_duration})"
            else:
                timeline_instruction = "7. Include a casual, conversational timeline (e.g., 'Looking at about 2-3 weeks to get this wrapped up')"
        else:
            timeline_instruction = "7. DO NOT include any timeline - skip timeline section entirely"

        # Combine everything
        prompt = f"""
{self._get_system_role(style)}

{self._get_proposal_system_rules(include_timeline, job_data)}

{job_section}

{projects_section}

{patterns_section}

{structure_section}

{style_guide}

{tone_guide}

Generate the proposal NOW. Target: {max_words} words (ideal range: 200-350).

CRITICAL RULES:
1. NO "As an AI", "I'm an AI", corporate jargon, or formal language
2. Sound like a REAL person having a conversation - casual, natural, human
3. Start with acknowledgment of THEIR specific problem
4. Reference 2-3 past similar projects with outcomes
5. Use PLAIN URLs (not markdown) for portfolio links and feedback URLs
6. Be conversational, direct, punchy - every word counts
{timeline_instruction}
8. End with a friendly, easygoing CTA (e.g., "Happy to hop on a quick call" or "Let me know if you want to chat")
9. AIM FOR 200-350 words (balanced length - not too short, not overwhelming)
10. PLATFORM MATCH: WordPress job = WordPress examples ONLY, Shopify job = Shopify examples ONLY
11. NO robotic phrases like "I am eager to", "I would be delighted" - talk like a real human
"""
        return prompt

    def build_improvement_prompt(
        self,
        current_proposal: str,
        feedback: str,
        aspect: str
    ) -> str:
        """
        Build prompt to improve an existing proposal.
        
        Args:
            current_proposal: Original proposal text
            feedback: Feedback or suggestions
            aspect: What to improve (tone, length, details, structure, impact)
            
        Returns:
            Improvement prompt
        """
        improvements = {
            "tone": "Adjust the tone to be more engaging while maintaining professionalism",
            "length": "Adjust length while maintaining key information",
            "details": "Add relevant details and specific examples",
            "structure": "Reorganize for better flow and impact",
            "impact": "Make it more persuasive and compelling"
        }

        improvement_instruction = improvements.get(aspect, improvements["impact"])

        return f"""
Review this proposal and improve it based on the feedback provided.

CURRENT PROPOSAL:
{current_proposal}

FEEDBACK:
{feedback}

IMPROVEMENT FOCUS:
{improvement_instruction}

Please provide an improved version that:
1. Addresses the specific feedback
2. Maintains professional tone
3. Keeps key information
4. Improves overall impact
5. Follows proposal best practices

Improved Proposal:
"""

    def _get_system_role(self, style: str) -> str:
        """Get system role based on style"""
        roles = {
            "professional": "You are an expert professional proposal writer with 10+ years experience winning high-value contracts.",
            "casual": "You are a friendly freelancer who writes approachable, personable proposals.",
            "technical": "You are a senior technical architect writing proposals that demonstrate deep expertise.",
            "creative": "You are a creative innovator who writes memorable, unique proposals.",
            "data_driven": "You are a data analyst who writes proposals backed by metrics and ROI analysis."
        }
        return roles.get(style, roles["professional"])

    def _get_proposal_system_rules(self, include_timeline: bool = False, job_data: Dict[str, Any] = None) -> str:
        """Get critical system rules for generating SHORT, HUMAN, WINNING proposals"""
        
        timeline_rule = ""
        timeline_pattern = ""
        if include_timeline:
            timeline_rule = "✓ Include casual timeline mention (conversational, not formal)"
            timeline_pattern = "4. TIMELINE (1 casual sentence): e.g., 'Looking at about 2-3 weeks to wrap this up'"
        else:
            timeline_rule = "✗ NO timeline section - skip it entirely"
            timeline_pattern = "4. SKIP TIMELINE - do not mention duration or timeline"
        
        # Extract pain points and build empathy guidance
        pain_points_section = ""
        empathy_statement = ""
        specific_problem = ""
        urgency_promise = ""
        if job_data:
            job_desc = job_data.get('job_description', '')
            job_title = job_data.get('job_title', '')
            pain_points = self.extract_pain_points(job_desc)
            empathy_statement = self.build_empathy_statement(pain_points)
            specific_problem = self.extract_specific_problem(job_desc, job_title)
            
            # Detect urgency level and get appropriate timeline promise
            urgency_level = self.detect_urgency_level(job_desc, job_title)
            urgency_timeline = self.get_urgency_timeline_promise(urgency_level)
            
            if pain_points:
                pain_points_section = f"""

DETECTED CLIENT PAIN POINTS (address these directly!):
"""
                for category, phrases in pain_points.items():
                    pain_points_section += f"• {category.upper()}: {phrases[0][:100]}...\n"
            
            if empathy_statement:
                pain_points_section += f"\n💡 SUGGESTED EMPATHY OPENER: \"{empathy_statement}\"\n"
            
            if specific_problem:
                pain_points_section += f"\n🎯 SPECIFIC PROBLEM TO ADDRESS: {specific_problem}\n"
            
            # Add urgency-based timeline promise
            if urgency_level != "standard" and urgency_timeline:
                pain_points_section += f"\n⚡ URGENT JOB DETECTED ({urgency_level.upper()})!\n"
                pain_points_section += f"   → Promise fast turnaround: \"{urgency_timeline}\"\n"
                pain_points_section += f"   → Show availability and willingness to prioritize\n"
                urgency_promise = urgency_timeline
        
        return f"""
SYSTEM MESSAGE - CRITICAL RULES:
You are a SHORT, HUMAN, WINNING proposal writer. Your goal: 3-5x better response rates.
{pain_points_section}

🧠 THE HUMAN CONNECTION FORMULA:
1. SHOW you read their job post (reference SPECIFIC details they mentioned)
2. FEEL their frustration/urgency (empathize, don't just acknowledge)
3. PROVE you've solved this EXACT problem before (with links)
4. EXPLAIN your specific approach for THEIR situation
5. MAKE IT EASY to say yes (friendly, low-pressure CTA)

WHAT NOT TO DO:
❌ NEVER say "As an AI" or "I'm an AI"
❌ NO corporate jargon, buzzwords, formal tone
❌ NO generic opening or "I'm excited to help" or "I'm thrilled"
❌ NO long introductions - get straight to the point
❌ NO talking about yourself - focus on THEIR problem
❌ NO proposals over 350 words - CONCISE = HIGH IMPACT
❌ NO robotic phrases like "I would be delighted", "I am eager to", "I look forward to"
❌ NO formal sign-offs like "Best regards", "Sincerely", "Warm regards" - keep it casual
❌ DO NOT fabricate or invent feedback URLs - only use ones provided in the data

WHAT TO DO:
✓ Sound like a REAL person having a coffee chat - natural, casual, human
✓ Start with: "I see you're dealing with [SPECIFIC PROBLEM]"
✓ Reference 2-3 REAL past projects with company names and outcomes
✓ Include portfolio links (ALWAYS available)
✓ Include feedback URLs ONLY if they exist in the data - don't make them up
✓ Use conversational, punchy language - contractions are good (I've, you're, that's)!
✓ Show specific approach for THEIR tech stack
{timeline_rule}
✓ End with friendly, easygoing CTA (e.g., "Happy to chat more" or "Let me know what you think")
✓ Total: 200-350 words (concise, direct, human)

CONVERSATIONAL TONE EXAMPLES:
✓ "Saw your job post and this is right up my alley"
✓ "I've tackled this exact issue before"
✓ "Here's what worked for a similar client"
✓ "Happy to hop on a quick call if you want to discuss"
✓ "Let me know if you have any questions"
✓ "Cheers!" or just end naturally without formal sign-off

CRITICAL - MATCH PROJECT INTENT, NOT JUST PLATFORM:
⚠️ UNDERSTAND what the client ACTUALLY WANTS:
   - Migration/Transfer → Show projects where you did content migration, data transfer, platform switching
   - Membership Setup → Show projects where you set up subscriptions, paid content, member areas
   - Speed Optimization → Show projects where you improved pagespeed, core web vitals
   - Bug Fixes → Show projects where you fixed issues, solved problems
   - New Feature → Show projects where you added functionality
⚠️ DO NOT mismatch project types:
   - If client wants MIGRATION → DON'T show speed optimization work
   - If client wants SPEED → DON'T show membership setup work
   - If client wants MEMBERSHIP → DON'T show unrelated WordPress work
⚠️ The project you reference MUST match what they're asking for!

CRITICAL - PLATFORM-SPECIFIC EXAMPLES:
⚠️ If the job is for WORDPRESS → ONLY show WordPress project examples
⚠️ If the job is for SHOPIFY → ONLY show Shopify project examples
⚠️ If the job is for WOOCOMMERCE → Show WooCommerce/WordPress examples
⚠️ NEVER mix platforms - WordPress job = WordPress proof, Shopify job = Shopify proof
⚠️ This builds credibility - show you've done EXACTLY this type of work before

COMBINED REQUIREMENT - INTENT + PLATFORM:
✓ Match BOTH the platform AND the type of work
✓ Example: "Substack to WordPress migration with membership" needs:
  - WordPress platform ✓
  - Migration/content transfer work ✓
  - OR membership/subscription setup work ✓
✓ NOT just any WordPress project - must be RELEVANT work!

SUCCESS PATTERN:
1. HOOK (1-2 sentences): DIRECTLY ADDRESS THEIR PAIN POINT + include ONE portfolio link
   ❌ BAD HOOK: "I see you need WordPress help" (too generic)
   ❌ BAD HOOK: "I'm an experienced developer" (about you, not them)
   ✅ GOOD HOOK: "8-10 second load times are brutal for conversions - I just fixed this exact issue" + portfolio link
   ✅ GREAT HOOK: "I know how frustrating it is when checkout stops working and sales drop every hour" + portfolio link
   → Start with THEIR problem, not your qualifications
   → Include empathy if they sound frustrated
   → Drop a portfolio link immediately (they see this in preview!)
2. PROOF (2 bullets): Past similar projects + portfolio links (+ feedback URLs IF available)
3. APPROACH (2-3 sentences): How you'd solve THEIR problem specifically
{timeline_pattern}
5. CTA (1 casual sentence): "Happy to chat" or "Let me know what you think" - friendly, easygoing

KEY WINNING STRATEGY:
- Put a portfolio link IN THE HOOK - clients see this in preview!
- ONLY use platform-specific examples (WordPress→WordPress, Shopify→Shopify)
- Include feedback URLs ONLY when they actually exist in the data provided

CRITICAL LINK FORMAT:
- Use PLAIN URLs only (e.g., https://example.com/)
- DO NOT use markdown format like [Company](url)
- Portfolio URL = ALWAYS include (live work proof)
- Feedback URL = ONLY if provided in the data (don't fabricate)

TARGET: 200-350 words. Every word must count. Include enough detail to build trust.
"""

    def _build_hook_proof_approach_structure(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        include_timeline: bool = False,
        timeline_duration: str = None
    ) -> str:
        """Build template showing HOOK→PROOF→APPROACH→(TIMELINE)→CTA structure with portfolio link in hook"""
        problem = job_data.get("job_description", "")[:200]  # First 200 chars of job description
        company = job_data.get("company_name", "this client")
        
        # Get the BEST matching project for the HOOK (first one with ACTUAL project URL, not Upwork)
        hook_project = None
        hook_portfolio_url = None
        for proj in similar_projects[:3]:
            portfolio_urls = proj.get("portfolio_urls", [])
            if portfolio_urls:
                hook_project = proj.get("company", proj.get("title", "past project"))
                # PRIORITIZE actual project URLs over Upwork profile links
                for url in portfolio_urls:
                    if url and 'upwork.com' not in url.lower():
                        hook_portfolio_url = url
                        break
                # Fallback to Upwork if no actual project URL
                if not hook_portfolio_url and portfolio_urls[0]:
                    hook_portfolio_url = portfolio_urls[0]
                break
        
        projects_proof = ""
        for i, proj in enumerate(similar_projects[:3], 1):
            company_name = proj.get("company", proj.get("title", "past project"))
            portfolio_urls = proj.get("portfolio_urls", [])
            feedback_url = proj.get("client_feedback_url", "")  # May be None or empty
            
            # Prioritize actual project URL
            best_url = None
            for url in portfolio_urls:
                if url and 'upwork.com' not in url.lower():
                    best_url = url
                    break
            if not best_url and portfolio_urls and portfolio_urls[0]:
                best_url = portfolio_urls[0]
            
            portfolio = f"\n     → Live site: {best_url}" if best_url else ""
            
            # ONLY add feedback URL if it actually exists
            feedback_line = ""
            if feedback_url and feedback_url.strip():
                feedback_line = f"\n     → Client feedback: {feedback_url}"
            
            satisfaction = proj.get("satisfaction", proj.get("effectiveness", 4.5))
            projects_proof += f"  {i}. {company_name} - {satisfaction}/5 satisfaction{portfolio}{feedback_line}\n"

        # Build hook example with portfolio link (actual project URL preferred) - PLAIN URL, no markdown
        hook_example = '"Noticed you need help with [specific problem from their job description] - I\'ve got relevant experience here..."'
        if hook_project and hook_portfolio_url:
            hook_example = f'"Noticed you need help with [specific problem]. Just wrapped up something similar: {hook_portfolio_url}"'

        # Build timeline section based on include_timeline flag
        timeline_section = ""
        if include_timeline:
            if timeline_duration:
                timeline_section = f"""
[TIMELINE - 1 casual sentence]
Something like: "Looking at about {timeline_duration} to get this wrapped up" or "Should take around {timeline_duration}"
Keep it conversational, not formal!
"""
            else:
                timeline_section = """
[TIMELINE - 1 casual sentence]
Something like: "Looking at about 2-3 weeks to wrap this up" or "Should take around a week"
Keep it conversational and easygoing, not robotic!
"""
        else:
            timeline_section = """
[NO TIMELINE - SKIP THIS SECTION]
Do NOT include any timeline or duration in this proposal.
"""

        # Extract pain points for HOOK guidance
        pain_points = self.extract_pain_points(job_data.get('job_description', ''))
        specific_problem = self.extract_specific_problem(job_data.get('job_description', ''), job_data.get('job_title', ''))
        empathy_opener = self.build_empathy_statement(pain_points)
        
        # Build pain-point-aware hook guidance with SPECIFIC instructions
        hook_guidance = ""
        if pain_points:
            if 'urgency' in pain_points:
                urgency_level = self.detect_urgency_level(job_data.get('job_description', ''), job_data.get('job_title', ''))
                urgency_timeline = self.get_urgency_timeline_promise(urgency_level)
                hook_guidance = f"\n⚡ URGENT JOB - Start your HOOK acknowledging the time pressure!\n"
                hook_guidance += f"   Example opener: \"I know every hour counts when checkout is down - {urgency_timeline or 'I can prioritize this'}\"\n"
                hook_guidance += f"   → Show immediate availability and commitment to fast delivery"
            elif 'frustration' in pain_points or 'previous_failure' in pain_points:
                hook_guidance = "\n💡 FRUSTRATED CLIENT - Start your HOOK with empathy!\n"
                hook_guidance += f"   Example opener: \"{empathy_opener or 'I know how frustrating this can be'}\"\n"
                hook_guidance += "   → Acknowledge their pain BEFORE talking about your skills"
            elif 'business_impact' in pain_points:
                hook_guidance = "\n💰 BUSINESS IMPACT - Start your HOOK acknowledging the revenue pressure!\n"
                hook_guidance += "   Example opener: \"Losing sales every hour is brutal - let's get this fixed fast\"\n"
                hook_guidance += "   → Show you understand the business cost, not just the technical problem"

        return f"""
PROPOSAL STRUCTURE TO USE:

[HOOK - 2-3 sentences WITH PORTFOLIO LINK]
{hook_example}
{hook_guidance}

🎯 HUMANIZE YOUR HOOK - Don't just acknowledge the problem, FEEL it with them:
• BAD: "I see you need a WordPress migration"
• GOOD: "Migrating 5000 subscribers while keeping your membership tiers intact - that's no small task"
• BEST: "I know how stressful platform migrations can be, especially with 5000+ subscribers counting on you"

⚠️ IMPORTANT: Include a portfolio link IN THE HOOK! Clients see this in the preview - it triggers curiosity!
⚠️ USE PLAIN URLs only (e.g., https://example.com) - NOT markdown format like [text](url)

[PROOF - 2-3 bullets with portfolio links]
Reference these similar past projects:
{projects_proof}

📝 Format each proof point CONVERSATIONALLY:
• BAD: "Successfully completed WordPress project for Acme Corp"
• GOOD: "Just wrapped up something similar for Acme Corp - migrated their entire blog and membership system"
  Live work: https://portfolio-url.com
  (Only include feedback URL if it exists in the data above - DON'T fabricate)

[APPROACH - 2-3 sentences that show YOU UNDERSTAND THEIR SPECIFIC SITUATION]
🎯 Personalize to THEIR situation, not generic "I will deliver excellent results":
• BAD: "I will ensure a smooth migration process"
• GOOD: "For your Substack → WordPress move, I'd handle the subscriber import first (preserving tiers), then migrate content, and finally set up your paywall"
{timeline_section}
[CTA - 1 casual sentence]
End naturally - like texting a colleague, not writing a formal letter:
• "Happy to hop on a quick call to discuss"
• "Let me know what you think!"
• "Shoot me a message if you have questions"
• Just end naturally - no sign-off needed!

❌ NEVER use: "Best regards", "Sincerely", "Looking forward to hearing from you"

CRITICAL FORMAT RULES:
1. Use PLAIN URLs (https://example.com) - NOT markdown [text](url) format
2. Include feedback URLs alongside portfolio links
3. Target: 150-250 words MAXIMUM. PUT A PORTFOLIO LINK IN THE HOOK!
4. PLATFORM MATCH: WordPress job = WordPress examples, Shopify job = Shopify examples
5. Sound like a human having a CONVERSATION - casual, genuine, understanding

🔑 THE WINNING FORMULA:
Empathy → Proof → Specific Solution → Easy Next Step
"""

    def _build_job_section(self, job_data: Dict[str, Any]) -> str:
        """Build job requirements section"""
        return f"""
JOB DETAILS:
Company: {job_data.get('company_name')}
Position: {job_data.get('job_title')}
Industry: {job_data.get('industry', 'Not specified')}
Complexity: {job_data.get('task_complexity', 'Medium')}
Timeline: {job_data.get('project_duration_days', 'Not specified')} days
Budget: ${job_data.get('estimated_budget', 'Not specified')}
Urgency: {'URGENT - Ad-hoc' if job_data.get('urgent_adhoc') else 'Planned'}

REQUIRED SKILLS:
{', '.join(job_data.get('skills_required', []))}

JOB DESCRIPTION:
{job_data.get('job_description', 'No description provided')}
"""

    def _build_projects_section(
        self,
        similar_projects: List[Dict[str, Any]],
        include_portfolio: bool,
        include_feedback: bool
    ) -> str:
        """
        Build similar projects reference section formatted for proposal generation.
        
        CRITICAL: 
        1. Only include projects that have actual portfolio URLs
        2. PREFER actual project URLs (https://example.com/) over Upwork profile links
        3. Only include feedback URL if it ACTUALLY EXISTS - don't fabricate
        4. Format: Company name, TASK TYPE, brief outcome, portfolio link, feedback (if available)
        5. Include task_type so AI knows what work was actually done (not just platform)
        6. DIVERSIFY: Track used URLs to avoid repeating same portfolio links
        """
        if not similar_projects:
            return "SIMILAR PAST PROJECTS: None found in database yet."

        section = "SIMILAR PAST PROJECTS (WITH PORTFOLIO PROOF):\n\n"
        section += "⚠️ IMPORTANT: Reference projects that match the platform AND type of work!\n"
        section += "⚠️ USE DIFFERENT PORTFOLIO LINKS for each project - don't repeat the same URLs!\n\n"
        projects_added = 0
        used_portfolio_urls = set()  # Track used URLs for diversity

        for i, project in enumerate(similar_projects[:5], 1):  # Check up to 5 to find 3 good ones
            if projects_added >= 3:
                break
                
            company = project.get('company') or project.get('title', 'Past project')
            portfolio_urls = project.get('portfolio_urls') or []
            feedback_url = project.get('client_feedback_url')  # May be None/empty
            feedback_text = project.get('client_feedback_text', '')  # May be empty
            task_type = project.get('task_type', '')  # What work was actually done
            
            # Only include projects with portfolio links
            if not portfolio_urls:
                continue
            
            # PRIORITIZE actual project URLs over Upwork profile links
            # AND ensure we haven't used this URL already (diversification)
            actual_project_url = None
            upwork_profile_url = None
            
            for url in portfolio_urls:
                if not url:  # Skip empty strings
                    continue
                url_lower = url.lower().strip()
                # Skip if already used in another project
                if url_lower in used_portfolio_urls:
                    continue
                if 'upwork.com' in url_lower:
                    if not upwork_profile_url:  # Only set if not already set
                        upwork_profile_url = url
                else:
                    actual_project_url = url
                    break  # Prefer first non-Upwork URL that's not used
            
            # Use actual project URL first, fallback to Upwork profile
            best_portfolio_url = actual_project_url or upwork_profile_url
            
            if not best_portfolio_url:
                continue
            
            # Track this URL as used
            used_portfolio_urls.add(best_portfolio_url.lower().strip())
            
            projects_added += 1
                
            # Format: Company name, TASK TYPE, specific outcome, link
            section += f"{projects_added}. **{company}**"
            
            # Add task type - CRITICAL for AI to know what work was done
            if task_type:
                section += f" [Task: {task_type}]"
            
            section += ": "
            
            # Add specific outcome/skills for context
            skills = project.get('skills', [])
            if skills:
                section += f"Built with {', '.join(skills[:3])}"
            else:
                section += "Delivered successfully"
            
            # Add portfolio link (prioritize actual project URL) - PLAIN URLs
            if include_portfolio and best_portfolio_url:
                section += f"\n   → Live project: {best_portfolio_url}"
            
            # ONLY add feedback URL if it ACTUALLY EXISTS and is not empty
            # Don't suggest or fabricate feedback URLs
            if include_feedback and feedback_url and feedback_url.strip():
                section += f"\n   → Client feedback: {feedback_url}"
                # Only show feedback text preview if it exists
                if feedback_text and feedback_text.strip():
                    preview = feedback_text[:100] + '...' if len(feedback_text) > 100 else feedback_text
                    section += f"\n     \"{preview}\""
            
            section += "\n\n"
        
        if projects_added == 0:
            return "SIMILAR PAST PROJECTS: No projects with portfolio links found."
        
        # Add diversity reminder
        section += f"\n⚠️ DIVERSITY CHECK: {len(used_portfolio_urls)} unique portfolio URLs - use DIFFERENT links for each project!\n"

        return section

    def _build_patterns_section(self, success_patterns: List[str]) -> str:
        """Build success patterns section"""
        if not success_patterns:
            return "SUCCESS PATTERNS: Generic best practices apply."

        section = "PROVEN SUCCESS PATTERNS:\n"
        for pattern in success_patterns[:5]:
            section += f"• {pattern}\n"

        return section

    def _build_requirements_section(self, style: str, tone: str, max_words: int) -> str:
        """Build detailed requirements section aligned with SHORT, HUMAN, WINNING goals"""
        return f"""
PROPOSAL REQUIREMENTS (SHORT, HUMAN, WINNING):
1. Target word count: {max_words} words (NOT longer, be concise)
2. Structure: HOOK → PROOF → APPROACH → TIMELINE → CTA
3. HOOK: Start with "I see you're dealing with [their specific problem]"
4. PROOF: Reference 2-3 past projects by company name with outcomes
5. PROOF: Include portfolio links and client feedback URLs for credibility
6. APPROACH: Specific solution for THEIR tech stack and requirements
7. APPROACH: Reference techniques from past successful projects
8. TIMELINE: Realistic phases and duration based on similar work
9. CTA: Friendly, direct call-to-action (e.g., "Let's discuss specifics")
10. TONE: Sound like a real person. Conversational. Direct. Human. NO AI language.

QUALITY CHECKS:
✓ Word count 250-350? (SHORT = HIGH IMPACT)
✓ Conversational tone? (No corporate jargon?)
✓ References past projects by name?
✓ Includes portfolio links?
✓ Shows understanding of THEIR specific problem?
✓ Specific to THEIR tech stack?
✓ Includes timeline?
✓ Has clear CTA?
✓ No "As an AI" language?

If ANY quality check fails, fix it before submitting.
"""

    def score_proposal_quality(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        references: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score proposal quality based on SHORT, HUMAN, WINNING criteria.
        
        Target: 150-250 words, conversational tone, references to past projects.
        
        Returns:
            Quality metrics and suggestions
        """
        score = 0.0
        feedback = []
        
        # Ideal word count: 150-250 (SHORT = HIGH IMPACT)
        word_count = len(proposal_text.split())
        if 150 <= word_count <= 250:
            score += 0.25  # Perfect word count
        elif 120 < word_count < 300:
            score += 0.15  # Acceptable word count
            feedback.append(f"Word count is {word_count} - ideal is 150-250 for maximum impact")
        else:
            score += 0.05
            feedback.append(f"Proposal is {word_count} words - should be 150-250 for SHORT impact")

        # Check for AI language (SHOULD NOT have these)
        ai_phrases = ["as an ai", "i'm an ai", "artificial intelligence", "machine learning", "algorithm", "as a language model"]
        has_ai_language = any(phrase in proposal_text.lower() for phrase in ai_phrases)
        if not has_ai_language:
            score += 0.2  # No AI language detected
        else:
            score -= 0.1
            feedback.append("Remove AI language - sounds like ChatGPT, not a real person")

        # Check for conversational tone (should have personal pronouns, contractions)
        has_conversational = (
            ("i've" in proposal_text.lower() or "i have" in proposal_text.lower()) and
            ("you're" in proposal_text.lower() or "your" in proposal_text.lower()) and
            ("." in proposal_text)  # Sentences
        )
        if has_conversational:
            score += 0.15  # Conversational tone
        else:
            feedback.append("Add conversational tone - use 'I've', 'you're', contractions")

        # Check for problem acknowledgment (HOOK)
        has_problem_ack = any(
            phrase in proposal_text.lower() 
            for phrase in ["i see you", "i understand", "deal with", "challenge", "problem", "looking for"]
        )
        if has_problem_ack:
            score += 0.15  # Acknowledges their problem
        else:
            feedback.append("Start by acknowledging THEIR specific problem (HOOK)")

        # Check for past project references (PROOF)
        refs_used = len(references.get("projects_referenced", []))
        if refs_used >= 2:
            score += 0.15  # Good number of references
        else:
            feedback.append(f"Reference 2-3 past projects by company name (currently {refs_used})")

        # Check for portfolio/feedback (credibility)
        portfolio_count = len(references.get("portfolio_links_used", []))
        feedback_count = len(references.get("feedback_urls_cited", []))
        credibility_refs = portfolio_count + feedback_count
        
        if credibility_refs >= 2:
            score += 0.15  # Strong credibility with links
        elif credibility_refs >= 1:
            score += 0.08  # Some credibility
        else:
            feedback.append("Add portfolio links and feedback URLs for social proof")

        return {
            "overall_score": round(min(score, 1.0), 2),
            "word_count": word_count,
            "ideal_word_count": "250-350",
            "components": {
                "length": 0.25 if 250 <= word_count <= 350 else 0.15,
                "human_language": 0.2 if not has_ai_language else 0.0,
                "conversational_tone": 0.15 if has_conversational else 0.0,
                "problem_acknowledgment": 0.15 if has_problem_ack else 0.0,
                "past_project_references": min(refs_used / 3, 1.0) * 0.15,
                "credibility": min(credibility_refs / 3, 1.0) * 0.15
            },
            "feedback": feedback,
            "suggestions": self._get_improvement_suggestions(proposal_text, word_count),
            "is_short_human_winning": score >= 0.85 and 250 <= word_count <= 350
        }

    def _get_improvement_suggestions(self, proposal_text: str, word_count: int) -> List[str]:
        """Get improvement suggestions based on content"""
        suggestions = []

        if word_count < 200:
            suggestions.append("Add more detail about your approach and experience")
        if word_count > 700:
            suggestions.append("Consider condensing - shorter proposals often perform better")

        if proposal_text.count(".") < 5:
            suggestions.append("Use more varied sentence structure")

        if "I have" not in proposal_text and "I've" not in proposal_text:
            suggestions.append("Add personal touch - mention your own relevant experience")

        return suggestions

    def generate_followup_questions(
        self,
        job_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate intelligent follow-up questions to ask the client.
        
        These questions help clarify requirements and show professionalism.
        """
        questions = []

        # Budget question
        if not job_data.get("estimated_budget"):
            questions.append({
                "category": "Budget",
                "question": "What is your estimated budget for this project?",
                "why": "Understanding budget helps plan resource allocation"
            })

        # Timeline question
        if not job_data.get("project_duration_days"):
            questions.append({
                "category": "Timeline",
                "question": "When do you need this completed by?",
                "why": "Timeline helps determine schedule and team capacity"
            })

        # Complexity question
        if job_data.get("task_complexity") == "medium":
            questions.append({
                "category": "Scope",
                "question": "Are there any hidden complexities or dependencies we should know about?",
                "why": "Understanding full scope prevents surprises later"
            })

        # Integration question
        if "api" in job_data.get("job_description", "").lower() or "integration" in job_data.get("job_description", "").lower():
            questions.append({
                "category": "Technical",
                "question": "What existing systems need to integrate with this solution?",
                "why": "Integration requirements affect architecture and timeline"
            })

        # Communication question
        questions.append({
            "category": "Communication",
            "question": "How would you prefer to communicate and what's your expected response time?",
            "why": "Clear communication expectations prevent misalignment"
        })

        return questions

    def enhance_proposal_with_questions(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        include_questions: bool = True
    ) -> str:
        """
        Optionally enhance proposal with relevant follow-up questions.
        
        Shows professionalism and initiative.
        """
        if not include_questions:
            return proposal_text

        questions = self.generate_followup_questions(job_data)

        if not questions:
            return proposal_text

        questions_section = "\n\nHELPFUL QUESTIONS FOR CLARIFICATION:\n"
        for q in questions[:3]:  # Limit to 3 questions
            questions_section += f"• {q['question']}\n"

        return proposal_text + questions_section
