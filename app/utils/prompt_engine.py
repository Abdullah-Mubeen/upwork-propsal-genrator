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
        max_words: int = 500,
        include_portfolio: bool = True,
        include_feedback: bool = True
    ) -> str:
        """
        Build an optimized prompt for proposal generation.
        
        Args:
            job_data: The new job requirements
            similar_projects: Similar past successful projects
            success_patterns: Patterns that worked
            style: Writing style
            tone: Proposal tone
            max_words: Target word count
            include_portfolio: Include portfolio links?
            include_feedback: Include feedback/testimonials?
            
        Returns:
            Optimized prompt for OpenAI
        """
        logger.debug(f"[PromptEngine] Building proposal prompt (style={style}, tone={tone})")

        # Get style and tone instructions
        style_enum = ProposalStyle(style) if isinstance(style, str) else style
        tone_enum = ProposalTone(tone) if isinstance(tone, str) else tone

        style_guide = self.STYLE_INSTRUCTIONS.get(style_enum, "")
        tone_guide = self.TONE_INSTRUCTIONS.get(tone_enum, "")

        # Build sections
        job_section = self._build_job_section(job_data)
        projects_section = self._build_projects_section(similar_projects, include_portfolio, include_feedback)
        patterns_section = self._build_patterns_section(success_patterns)
        requirements_section = self._build_requirements_section(style, tone, max_words)

        # Combine everything
        prompt = f"""
{self._get_system_role(style)}

{job_section}

{projects_section}

{patterns_section}

{style_guide}

{tone_guide}

{requirements_section}

Generate the proposal now. Make it compelling, specific, and data-backed.
Keep it around {max_words} words.
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
        """Build similar projects reference section"""
        if not similar_projects:
            return "SIMILAR PAST PROJECTS: None found in database yet."

        section = "SIMILAR PAST PROJECTS THAT SUCCEEDED:\n"

        for i, project in enumerate(similar_projects[:3], 1):
            effectiveness = project.get('effectiveness') or 0
            satisfaction = project.get('satisfaction') or 0
            section += f"""
Project {i}: {project.get('company')}
- Role: {project.get('title', 'Similar project')}
- Skills: {', '.join(project.get('skills', []) or [])}
- Success: {effectiveness * 100:.0f}% effective
- Client Satisfaction: {satisfaction}/5
"""
            if include_feedback and project.get('feedback_url'):
                section += f"- Feedback: {project.get('feedback_url')}\n"
            if include_portfolio and project.get('portfolio_urls'):
                portfolio_urls = project.get('portfolio_urls') or []
                if portfolio_urls:
                    section += f"- Portfolio: {portfolio_urls[0]}\n"

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
        """Build detailed requirements section"""
        return f"""
PROPOSAL REQUIREMENTS:
1. Reference 2-3 of the similar past projects mentioned above
2. Highlight expertise that matches this specific job
3. Include relevant portfolio work for credibility
4. Explain your approach and methodology
5. Set realistic timeline based on similar projects
6. Address the specific urgency/complexity level
7. Show understanding of their business needs
8. Include a clear Call-to-Action
9. Use {style} style with {tone} tone
10. Keep it approximately {max_words} words (flexible)
"""

    def score_proposal_quality(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        references: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score proposal quality based on various factors.
        
        Returns:
            Quality metrics and suggestions
        """
        score = 0.0
        feedback = []

        # Length check
        word_count = len(proposal_text.split())
        if 150 < word_count < 800:
            score += 0.2
        else:
            feedback.append(f"Proposal is {word_count} words - consider 200-600 for better impact")

        # Structure check
        has_opening = any(phrase in proposal_text.lower() for phrase in ["understand", "know", "familiar"])
        has_experience = "experience" in proposal_text.lower() or "built" in proposal_text.lower()
        has_approach = "approach" in proposal_text.lower() or "methodology" in proposal_text.lower()
        has_closing = "next" in proposal_text.lower() or "contact" in proposal_text.lower()

        structure_score = sum([has_opening, has_experience, has_approach, has_closing]) / 4
        score += structure_score * 0.3

        # Reference check
        refs_used = len(references.get("projects_referenced", []))
        if refs_used >= 2:
            score += 0.2
        else:
            feedback.append(f"Only {refs_used} past projects referenced - aim for 2-3")

        # Portfolio/feedback check
        portfolio_count = len(references.get("portfolio_links_used", []))
        feedback_count = len(references.get("feedback_urls_cited", []))

        if portfolio_count > 0:
            score += 0.15
        else:
            feedback.append("Consider adding portfolio links for credibility")

        if feedback_count > 0:
            score += 0.15
        else:
            feedback.append("Consider adding feedback/testimonial URLs")

        return {
            "overall_score": round(min(score, 1.0), 2),
            "components": {
                "length": 0.2 if 150 < word_count < 800 else 0.1,
                "structure": round(structure_score, 2),
                "references": 0.2 if refs_used >= 2 else 0.1,
                "credibility": round((portfolio_count + feedback_count) / 4, 2)
            },
            "feedback": feedback,
            "suggestions": self._get_improvement_suggestions(proposal_text, word_count)
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
