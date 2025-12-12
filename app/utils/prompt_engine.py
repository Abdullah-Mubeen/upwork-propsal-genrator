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

    def build_proposal_prompt(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]],
        success_patterns: List[str],
        style: str = "professional",
        tone: str = "confident",
        max_words: int = 200,
        include_portfolio: bool = True,
        include_feedback: bool = True
    ) -> str:
        """
        Build an optimized prompt for proposal generation using HOOK→PROOF→APPROACH→TIMELINE→CTA structure.
        
        Key Requirements:
        1. HOOK: 2 sentences acknowledging THEIR specific problem
        2. PROOF: 2-3 past projects with REAL COMPANY NAMES and ACTUAL portfolio links
        3. APPROACH: 3-4 sentences with specific solution for THEIR tech stack
        4. TIMELINE: 1-2 sentences with realistic phases
        5. CTA: 1 sentence friendly call-to-action
        
        CRITICAL: Only mention projects if they have actual portfolio links.
        Never suggest portfolio URLs - only use real URLs from historical data.
        
        Args:
            job_data: The new job requirements
            similar_projects: Similar past successful projects (MUST have portfolio_urls)
            success_patterns: Patterns that worked
            style: Writing style
            tone: Proposal tone
            max_words: Target word count (default 200, ideal range 150-250)
            include_portfolio: Include portfolio links?
            include_feedback: Include feedback/testimonials?
            
        Returns:
            Optimized prompt for OpenAI
        """
        logger.debug(f"[PromptEngine] Building proposal prompt (style={style}, tone={tone}, target={max_words} words)")

        # Get style and tone instructions
        style_enum = ProposalStyle(style) if isinstance(style, str) else style
        tone_enum = ProposalTone(tone) if isinstance(tone, str) else tone

        style_guide = self.STYLE_INSTRUCTIONS.get(style_enum, "")
        tone_guide = self.TONE_INSTRUCTIONS.get(tone_enum, "")

        # Build sections
        job_section = self._build_job_section(job_data)
        projects_section = self._build_projects_section(similar_projects, include_portfolio, include_feedback)
        patterns_section = self._build_patterns_section(success_patterns)
        structure_section = self._build_hook_proof_approach_structure(job_data, similar_projects)

        # Combine everything
        prompt = f"""
{self._get_system_role(style)}

{self._get_proposal_system_rules()}

{job_section}

{projects_section}

{patterns_section}

{structure_section}

{style_guide}

{tone_guide}

Generate the proposal NOW. Target: {max_words} words (ideal range: 150-250).

CRITICAL RULES:
1. NO "As an AI", "I'm an AI", corporate jargon, or formal language
2. Sound like a REAL person, not ChatGPT
3. Start with acknowledgment of THEIR specific problem
4. Reference 2-3 past similar projects with outcomes
5. Use PLAIN URLs (not markdown) for portfolio links and feedback URLs
6. Be conversational, direct, punchy - every word counts
7. Include timeline and clear Call-to-Action
8. AIM FOR 150-250 words (short proposals get 3-5x better response rates)
9. PLATFORM MATCH: WordPress job = WordPress examples ONLY, Shopify job = Shopify examples ONLY
10. Add more detail about your approach and what makes you different
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

    def _get_proposal_system_rules(self) -> str:
        """Get critical system rules for generating SHORT, HUMAN, WINNING proposals"""
        return """
SYSTEM MESSAGE - CRITICAL RULES:
You are a SHORT, HUMAN, WINNING proposal writer. Your goal: 3-5x better response rates.

WHAT NOT TO DO:
❌ NEVER say "As an AI" or "I'm an AI"
❌ NO corporate jargon, buzzwords, formal tone
❌ NO generic opening or "I'm excited to help"
❌ NO long introductions - get straight to the point
❌ NO talking about yourself - focus on THEIR problem
❌ NO proposals over 250 words - SHORT = HIGH IMPACT

WHAT TO DO:
✓ Sound like a REAL person who "gets it"
✓ Start with: "I see you're dealing with [SPECIFIC PROBLEM]"
✓ Reference 2-3 REAL past projects with company names and outcomes
✓ Include portfolio proof (links to live work) and feedback URLs
✓ Use conversational, punchy language
✓ Show specific approach for THEIR tech stack
✓ Include realistic timeline based on similar work
✓ End with clear, friendly call-to-action
✓ Total: 150-250 words (SHORT, direct, human)

CRITICAL - PLATFORM-SPECIFIC EXAMPLES:
⚠️ If the job is for WORDPRESS → ONLY show WordPress project examples
⚠️ If the job is for SHOPIFY → ONLY show Shopify project examples
⚠️ NEVER mix platforms - WordPress job = WordPress proof, Shopify job = Shopify proof
⚠️ This builds credibility - show you've done EXACTLY this type of work before

SUCCESS PATTERN:
1. HOOK (1-2 sentences): Acknowledge THEIR specific problem + include ONE portfolio link to similar work
   Example: "I see you're dealing with slow WooCommerce load times. Check out my recent work: https://ggov.no/"
2. PROOF (2 bullets): Past similar projects + portfolio links + feedback URLs
3. APPROACH (2-3 sentences): How you'd solve THEIR problem specifically
4. TIMELINE (1 sentence): Realistic duration
5. CTA (1 sentence): "Let's discuss" - friendly, direct

KEY WINNING STRATEGY:
- Put a portfolio link IN THE HOOK - clients see this in preview!
- ONLY use platform-specific examples (WordPress→WordPress, Shopify→Shopify)
- Include feedback URLs to show what past clients said

CRITICAL LINK FORMAT:
- Use PLAIN URLs only (e.g., https://ggov.no/)
- DO NOT use markdown format like [Company](url)
- Include both: portfolio URL (live work) + feedback URL (client review)

TARGET: 150-250 words MAXIMUM. Every word must count.
"""

    def _build_hook_proof_approach_structure(
        self,
        job_data: Dict[str, Any],
        similar_projects: List[Dict[str, Any]]
    ) -> str:
        """Build template showing HOOK→PROOF→APPROACH→TIMELINE→CTA structure with portfolio link in hook"""
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
                    if 'upwork.com' not in url.lower():
                        hook_portfolio_url = url
                        break
                # Fallback to Upwork if no actual project URL
                if not hook_portfolio_url:
                    hook_portfolio_url = portfolio_urls[0]
                break
        
        projects_proof = ""
        for i, proj in enumerate(similar_projects[:3], 1):
            company_name = proj.get("company", proj.get("title", "past project"))
            portfolio_urls = proj.get("portfolio_urls", [])
            
            # Prioritize actual project URL
            best_url = None
            for url in portfolio_urls:
                if 'upwork.com' not in url.lower():
                    best_url = url
                    break
            if not best_url and portfolio_urls:
                best_url = portfolio_urls[0]
            
            portfolio = f"\n     → Live site: {best_url}" if best_url else ""
            
            satisfaction = proj.get("satisfaction", proj.get("effectiveness", 4.5))
            projects_proof += f"  {i}. {company_name} - {satisfaction}/5 satisfaction{portfolio}\n"

        # Build hook example with portfolio link (actual project URL preferred) - PLAIN URL, no markdown
        hook_example = '"I see you\'re dealing with [specific problem from their job description]..."'
        if hook_project and hook_portfolio_url:
            hook_example = f'"I see you\'re dealing with [specific problem]. I just solved this exact issue - check it out: {hook_portfolio_url}"'

        return f"""
PROPOSAL STRUCTURE TO USE:

[HOOK - 2-3 sentences WITH PORTFOLIO LINK]
{hook_example}
⚠️ IMPORTANT: Include a portfolio link IN THE HOOK! Clients see this in the preview - it triggers curiosity!
⚠️ USE PLAIN URLs only (e.g., https://example.com) - NOT markdown format like [text](url)

[PROOF - 2-3 bullets with portfolio + feedback URLs]
Reference these similar past projects:
{projects_proof}

Format each proof point like:
- **Company Name**: Brief outcome
  Live work: https://portfolio-url.com
  Client feedback: https://feedback-url.com

[APPROACH - 2-3 sentences]
"For you, I'd [specific approach for their tech stack]..."

[TIMELINE - 1 sentence]
"Timeline: [duration]"

[CTA - 1 sentence]
"Let's discuss specifics"

CRITICAL FORMAT RULES:
1. Use PLAIN URLs (https://example.com) - NOT markdown [text](url) format
2. Include feedback URLs alongside portfolio links
3. Target: 150-250 words MAXIMUM. PUT A PORTFOLIO LINK IN THE HOOK!
4. PLATFORM MATCH: WordPress job = WordPress examples, Shopify job = Shopify examples
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
        2. PREFER actual project URLs (https://ggov.no/) over Upwork profile links
        3. Format: Company name, brief outcome, portfolio link
        """
        if not similar_projects:
            return "SIMILAR PAST PROJECTS: None found in database yet."

        section = "SIMILAR PAST PROJECTS (WITH PORTFOLIO PROOF):\n\n"

        for i, project in enumerate(similar_projects[:3], 1):
            company = project.get('company') or project.get('title', 'Past project')
            portfolio_urls = project.get('portfolio_urls') or []
            feedback_url = project.get('client_feedback_url')
            
            # Only include projects with portfolio links
            if not portfolio_urls:
                continue
            
            # PRIORITIZE actual project URLs over Upwork profile links
            # Actual project URL = more impressive (client can see live work)
            actual_project_url = None
            upwork_profile_url = None
            
            for url in portfolio_urls:
                if 'upwork.com' in url.lower():
                    upwork_profile_url = url
                else:
                    actual_project_url = url
                    break  # Prefer first non-Upwork URL
            
            # Use actual project URL first, fallback to Upwork profile
            best_portfolio_url = actual_project_url or upwork_profile_url
                
            # Format: Company name, specific outcome, link
            section += f"{i}. **{company}**: "
            
            # Add specific outcome/skills for context
            skills = project.get('skills', [])
            if skills:
                section += f"Built with {', '.join(skills[:3])}"
            else:
                section += "Delivered successfully"
            
            # Add portfolio link (prioritize actual project URL) - PLAIN URLs
            if include_portfolio and best_portfolio_url:
                section += f"\n   → Live project: {best_portfolio_url}"
            
            # Add feedback URL if available - this shows what client said about this work
            if include_feedback and feedback_url:
                section += f"\n   → Client feedback: {feedback_url}"
                # Get feedback text preview if available
                feedback_text = project.get('client_feedback_text', '')
                if feedback_text:
                    # Show first 100 chars of feedback as preview
                    preview = feedback_text[:100] + '...' if len(feedback_text) > 100 else feedback_text
                    section += f"\n     \"{preview}\""
            
            section += "\n\n"

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
