"""
Prompt Engine v2 - Streamlined Proposal Generation

Optimized from 1,249 → ~400 lines by:
- Moving templates to constants.py
- Removing duplicate rules
- Consolidating methods
"""
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from app.domain.constants import (
    PAIN_POINT_INDICATORS, URGENCY_TIMELINE_PROMISES, EMPATHY_RESPONSES,
    STYLE_INSTRUCTIONS, TONE_INSTRUCTIONS, HOOK_STRATEGIES, STALE_OPENINGS,
    ANTI_HALLUCINATION_RULES, PROPOSAL_FORMAT_RULES, PROPOSAL_STRUCTURE,
    CONVERSATIONAL_EXAMPLES, FORBIDDEN_PHRASES, SYSTEM_ROLES,
)
from app.utils.text_analysis import (
    detect_urgency_level as _detect_urgency_level,
    extract_pain_points as _extract_pain_points,
)

logger = logging.getLogger(__name__)

# Import hook strategy engine
try:
    from app.utils.hook_strategy import get_hook_engine, JobAnalysis
except ImportError:
    get_hook_engine = None
    JobAnalysis = None


class ProposalStyle(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    DATA_DRIVEN = "data_driven"


class ProposalTone(str, Enum):
    CONFIDENT = "confident"
    HUMBLE = "humble"
    ENTHUSIASTIC = "enthusiastic"
    ANALYTICAL = "analytical"
    FRIENDLY = "friendly"


@dataclass
class PromptConfig:
    style: ProposalStyle = ProposalStyle.PROFESSIONAL
    tone: ProposalTone = ProposalTone.CONFIDENT
    include_metrics: bool = True
    include_timeline: bool = True
    include_portfolio: bool = True
    max_word_count: int = 300


class PromptEngine:
    """Streamlined prompt engine for proposal generation."""

    # Delegate to text_analysis
    @staticmethod
    def detect_urgency_level(job_description: str, job_title: str = "") -> str:
        return _detect_urgency_level(job_description, job_title)

    @staticmethod
    def get_urgency_timeline_promise(urgency_level: str) -> Optional[str]:
        return URGENCY_TIMELINE_PROMISES.get(urgency_level)

    @staticmethod
    def extract_pain_points(job_description: str) -> Dict[str, List[str]]:
        return _extract_pain_points(job_description)

    @staticmethod
    def build_empathy_statement(pain_points: Dict[str, List[str]]) -> str:
        if not pain_points:
            return ""
        import random
        for category in ["business_impact", "urgency", "frustration", "previous_failure"]:
            if category in pain_points and category in EMPATHY_RESPONSES:
                return random.choice(EMPATHY_RESPONSES[category])
        return ""

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
        timeline_duration: str = None,
        profile_context: Optional[Dict[str, Any]] = None,
        requirements_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build optimized prompt using HOOK→PROOF→APPROACH→CTA structure."""
        
        style_key = style.value if isinstance(style, ProposalStyle) else style
        tone_key = tone.value if isinstance(tone, ProposalTone) else tone

        # Build sections
        system_role = SYSTEM_ROLES.get(style_key, SYSTEM_ROLES["professional"])
        style_guide = STYLE_INSTRUCTIONS.get(style_key, "")
        tone_guide = TONE_INSTRUCTIONS.get(tone_key, "")
        
        job_section = self._build_job_section(job_data)
        projects_section = self._build_projects_section(similar_projects, include_portfolio, include_feedback)
        profile_section = self._build_profile_section(profile_context) if profile_context else ""
        
        # NEW: Build requirements section from extracted job understanding
        requirements_section = self._build_requirements_section(requirements_context) if requirements_context else ""
        
        # Dynamic hook generation
        hook_section = self._build_hook_section(job_data, similar_projects)
        
        # Timeline instruction
        timeline_inst = self._get_timeline_instruction(include_timeline, timeline_duration)
        
        # Build anti-hallucination rules (enhanced with do_not_assume)
        anti_hallucination = self._build_anti_hallucination_rules(requirements_context)

        return f"""
{system_role}

{self._get_core_rules()}

{job_section}
{requirements_section}
{profile_section}
{projects_section}

{hook_section}

{PROPOSAL_STRUCTURE}

{style_guide}
{tone_guide}

Generate the proposal NOW. Target: {max_words} words (ideal: 200-350).

{anti_hallucination}

CRITICAL RULES:
1. NO "As an AI" - sound like a REAL person
2. USE A VARIED HOOK - not "I see you're dealing with"
3. Reference ONLY verified projects above
4. Use PLAIN URLs, NO markdown
{timeline_inst}
6. End with casual CTA: "Happy to chat" or "Let me know"

{PROPOSAL_FORMAT_RULES}
"""

    def _get_core_rules(self) -> str:
        """Core system rules - consolidated."""
        hooks = "\n".join(f"   {i}. {h}" for i, h in enumerate(HOOK_STRATEGIES, 1))
        stale = "\n".join(f'   - "{s}"' for s in STALE_OPENINGS)
        examples = "\n".join(f'   ✓ "{e}"' for e in CONVERSATIONAL_EXAMPLES)
        
        return f"""
SYSTEM: SHORT, HUMAN, WINNING proposal writer. Goal: 3-5x better response rates.

🧠 HUMAN CONNECTION FORMULA:
1. SHOW you read their job post (reference SPECIFIC details)
2. FEEL their frustration/urgency (empathize)
3. PROVE you've solved this EXACT problem (with links)
4. EXPLAIN your specific approach for THEIR situation
5. MAKE IT EASY to say yes (friendly CTA)

⚠️ VARIED HOOKS - First 2.5 lines = ALL client sees on Upwork!

❌ NEVER USE:
{stale}

✅ WINNING HOOK STRATEGIES:
{hooks}

CONVERSATIONAL TONE:
{examples}
"""

    def _build_hook_section(self, job_data: Dict[str, Any], similar_projects: List[Dict[str, Any]]) -> str:
        """Build dynamic hook suggestions."""
        section = ""
        
        # Try hook engine
        if job_data and get_hook_engine is not None:
            try:
                hook_engine = get_hook_engine()
                analysis = hook_engine.analyze_job(job_data)
                
                # Get portfolio URL for hook
                portfolio_url = self._get_best_portfolio_url(similar_projects)
                hooks = hook_engine.get_hook_variations(analysis, job_data, similar_projects, portfolio_url, count=3)
                
                section = f"""
🎯 JOB ANALYSIS:
• Sentiment: {analysis.sentiment.value.upper()} | Intent: {analysis.intent.value.upper()}
• Urgency: {analysis.urgency_level}/5 | Platform: {analysis.platform.upper()}
• Strategy: {analysis.hook_strategy}
"""
                if hooks:
                    section += "\n🔥 GENERATED HOOKS:\n"
                    for i, hook in enumerate(hooks, 1):
                        section += f'   {i}. "{hook}"\n'
                        
            except Exception as e:
                logger.warning(f"Hook engine error: {e}")
        
        # Fallback pain points
        if not section and job_data:
            pain_points = self.extract_pain_points(job_data.get('job_description', ''))
            if pain_points:
                empathy = self.build_empathy_statement(pain_points)
                section = f"\n💡 DETECTED PAIN POINTS: {list(pain_points.keys())[:3]}\n"
                if empathy:
                    section += f'   Empathy opener: "{empathy}"\n'
        
        return section

    def _get_best_portfolio_url(self, projects: List[Dict[str, Any]]) -> Optional[str]:
        """Get best portfolio URL (prefer non-Upwork)."""
        for proj in projects[:3]:
            for url in proj.get("portfolio_urls", []):
                if url and 'upwork.com' not in url.lower():
                    return url
        return None

    def _get_timeline_instruction(self, include: bool, duration: str = None) -> str:
        if include:
            dur = duration or "2-3 weeks"
            return f'5. Include casual timeline: "Looking at about {dur} to wrap this up"'
        return "5. NO timeline - skip timeline section"

    def _build_requirements_section(self, requirements: Dict[str, Any]) -> str:
        """
        Build section from extracted job requirements.
        
        This provides the LLM with structured understanding of what the client
        ACTUALLY wants, improving proposal relevance and reducing hallucination.
        """
        if not requirements:
            return ""
        
        sections = []
        
        # Exact task (most important)
        if requirements.get("exact_task"):
            sections.append(f"📌 EXACT CLIENT NEED: {requirements['exact_task']}")
        
        # Deliverables
        if requirements.get("deliverables"):
            delivs = ", ".join(requirements["deliverables"][:5])
            sections.append(f"📦 EXPECTED DELIVERABLES: {delivs}")
        
        # Client tone (for matching communication style)
        if requirements.get("client_tone"):
            tone_guide = {
                "urgent": "⚡ Client is URGENT - emphasize availability and speed",
                "frustrated": "😤 Client is FRUSTRATED - show empathy, highlight reliability",
                "technical": "🔧 Client is TECHNICAL - use precise language, show expertise",
                "casual": "😊 Client is CASUAL - keep it friendly and relaxed",
                "professional": "💼 Client is PROFESSIONAL - formal but warm",
                "exploratory": "🤔 Client is EXPLORING - educate and guide gently"
            }
            sections.append(tone_guide.get(requirements["client_tone"], ""))
        
        # Problems mentioned (for empathy)
        if requirements.get("problems"):
            probs = ", ".join(requirements["problems"][:3])
            sections.append(f"🎯 CLIENT PAIN POINTS: {probs}")
        
        # Constraints
        if requirements.get("constraints"):
            cons = ", ".join(requirements["constraints"][:3])
            sections.append(f"⚠️ CONSTRAINTS: {cons}")
        
        # Resources provided
        if requirements.get("resources_provided"):
            res = ", ".join(requirements["resources_provided"][:3])
            sections.append(f"📁 RESOURCES CLIENT WILL PROVIDE: {res}")
        
        if sections:
            return "\n🧠 EXTRACTED JOB UNDERSTANDING:\n" + "\n".join(sections) + "\n"
        return ""

    def _build_anti_hallucination_rules(self, requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Build anti-hallucination rules, enhanced with do_not_assume from extraction.
        
        The do_not_assume field is CRITICAL - it lists things the client did NOT
        specify that the proposal writer should NOT invent or assume.
        """
        base_rules = ANTI_HALLUCINATION_RULES
        
        if requirements and requirements.get("do_not_assume"):
            do_not_assume = requirements["do_not_assume"]
            if do_not_assume:
                avoid_list = "\n".join(f"   ❌ {item}" for item in do_not_assume[:5])
                enhanced_rules = f"""
{base_rules}

🚫 DO NOT ASSUME OR MENTION (client did not specify):
{avoid_list}

If you need to discuss any of the above, ask questions instead of assuming!
"""
                return enhanced_rules
        
        return base_rules

    def _build_job_section(self, job_data: Dict[str, Any]) -> str:
        """Build job details section."""
        return f"""
JOB DETAILS:
Company: {job_data.get('company_name')}
Position: {job_data.get('job_title')}
Industry: {job_data.get('industry', 'Not specified')}
Skills: {', '.join(job_data.get('skills_required', []))}
Urgency: {'URGENT' if job_data.get('urgent_adhoc') else 'Planned'}

DESCRIPTION:
{job_data.get('job_description', 'No description')}
"""

    def _build_profile_section(self, profile: Optional[Dict[str, Any]]) -> str:
        """Build freelancer profile section."""
        if not profile:
            return ""
        parts = ["FREELANCER PROFILE:"]
        if profile.get('display_name'):
            parts.append(f"Name: {profile['display_name']}")
        if profile.get('headline'):
            parts.append(f"Title: {profile['headline']}")
        if profile.get('skills'):
            parts.append(f"Skills: {', '.join(profile['skills'][:10])}")
        return "\n".join(parts) + "\n"

    def _build_projects_section(
        self,
        projects: List[Dict[str, Any]],
        include_portfolio: bool,
        include_feedback: bool
    ) -> str:
        """Build verified projects section."""
        if not projects:
            return """VERIFIED PROJECTS: None available.
🚫 NO PAST PROJECTS - Focus on APPROACH and SKILLS only. DO NOT fabricate experience."""

        # Check relevance
        has_relevant = any(p.get('similarity_score', 0) >= 0.45 for p in projects[:3])
        if not has_relevant:
            return """VERIFIED PROJECTS: Available but not closely relevant.
🚫 Focus on APPROACH. Don't force irrelevant project references."""

        section = f"VERIFIED PAST PROJECTS (Reference ONLY these):\n\n{ANTI_HALLUCINATION_RULES}\n\n"
        used_urls = set()
        added = 0

        for proj in projects[:5]:
            if added >= 3:
                break
            
            company = proj.get('company') or proj.get('title', 'Past project')
            urls = proj.get('portfolio_urls', [])
            
            # Get best URL (non-Upwork, not used)
            best_url = None
            for url in urls:
                if url and url.lower() not in used_urls:
                    if 'upwork.com' not in url.lower():
                        best_url = url
                        break
            if not best_url:
                for url in urls:
                    if url and url.lower() not in used_urls:
                        best_url = url
                        break
            
            if not best_url:
                continue
            
            used_urls.add(best_url.lower())
            added += 1
            
            section += f"PROJECT {added}: {company}\n"
            if proj.get('task_type'):
                section += f"   Task: {proj['task_type']}\n"
            if proj.get('deliverables'):
                delivs = proj['deliverables']
                section += f"   Deliverables: {', '.join(delivs[:4]) if isinstance(delivs, list) else delivs}\n"
            if include_portfolio:
                section += f"   Live: {best_url}\n"
            
            feedback_url = proj.get('client_feedback_url')
            if include_feedback and feedback_url and feedback_url.strip():
                section += f"   Feedback: {feedback_url}\n"
            section += "\n"

        return section if added > 0 else "PROJECTS: No projects with portfolio links found."

    def build_improvement_prompt(self, current: str, feedback: str, aspect: str) -> str:
        """Build prompt to improve existing proposal."""
        improvements = {
            "tone": "Adjust tone to be more engaging",
            "length": "Adjust length while keeping key info",
            "details": "Add relevant details and examples",
            "structure": "Reorganize for better flow",
            "impact": "Make more persuasive and compelling"
        }
        return f"""
Improve this proposal based on feedback.

CURRENT: {current}

FEEDBACK: {feedback}

FOCUS: {improvements.get(aspect, improvements['impact'])}

Provide improved version maintaining professional tone.
"""

    def score_proposal_quality(
        self,
        proposal_text: str,
        job_data: Dict[str, Any],
        references: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score proposal quality."""
        score = 0.0
        feedback = []
        text_lower = proposal_text.lower()
        word_count = len(proposal_text.split())
        
        # Word count (target 200-350)
        if 200 <= word_count <= 350:
            score += 0.25
        elif 150 < word_count < 400:
            score += 0.15
            feedback.append(f"Word count {word_count} - ideal is 200-350")
        else:
            feedback.append(f"Word count {word_count} - should be 200-350")

        # No AI language
        if not any(p in text_lower for p in ["as an ai", "i'm an ai", "as a language model"]):
            score += 0.2
        else:
            feedback.append("Remove AI language")

        # Conversational
        if ("i've" in text_lower or "i have" in text_lower) and "your" in text_lower:
            score += 0.15
        else:
            feedback.append("Add conversational tone - contractions, personal touch")

        # Problem acknowledgment
        if any(w in text_lower for w in ["deal with", "challenge", "problem", "looking for"]):
            score += 0.15

        # Project references
        refs = len(references.get("projects_referenced", []))
        score += min(refs / 3, 1.0) * 0.15

        # Portfolio/feedback links
        links = len(references.get("portfolio_links_used", [])) + len(references.get("feedback_urls_cited", []))
        score += min(links / 3, 1.0) * 0.1

        return {
            "overall_score": round(min(score, 1.0), 2),
            "word_count": word_count,
            "feedback": feedback,
            "is_short_human_winning": score >= 0.8 and 200 <= word_count <= 350
        }
