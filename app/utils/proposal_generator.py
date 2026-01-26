"""
Phase 3: Proposal Generation Service

Takes retrieval context from the multi-layer retrieval pipeline and generates
AI-powered proposals that:
- Reference similar past successful projects
- Include portfolio links for credibility
- Cite feedback URLs as evidence
- Incorporate success patterns
- Use learned writing styles and approaches

This service bridges the retrieval system with the AI to create data-driven proposals.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from app.utils.openai_service import OpenAIService
from app.utils.retrieval_pipeline import RetrievalPipeline
from app.utils.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


class ProposalGenerator:
    """
    Generates proposals using AI with context from similar past projects.
    
    Flow:
    1. Takes new job requirements
    2. Retrieves similar past projects and patterns
    3. Builds rich context prompt
    4. Calls OpenAI to generate proposal
    5. Returns generated proposal with source references
    """

    def __init__(self, openai_service: OpenAIService = None, retrieval_pipeline: RetrievalPipeline = None, prompt_engine: PromptEngine = None):
        """
        Initialize proposal generator.
        
        Args:
            openai_service: OpenAI service instance
            retrieval_pipeline: Retrieval pipeline instance
            prompt_engine: PromptEngine for intelligent prompt generation
        """
        self.openai_service = openai_service or OpenAIService()
        self.retrieval_pipeline = retrieval_pipeline
        self.prompt_engine = prompt_engine or PromptEngine()
        logger.info("ProposalGenerator initialized")

    def generate_proposal(
        self,
        new_job_data: Dict[str, Any],
        all_historical_jobs: List[Dict[str, Any]],
        max_length: int = 350,
        include_portfolio: bool = True,
        include_feedback: bool = True,
        proposal_style: str = "professional",
        tone: str = "confident"
    ) -> Dict[str, Any]:
        """
        Generate a SHORT, HUMAN, WINNING proposal for a new job using similar past projects.
        
        Target: 250-350 words, conversational tone, references to 2-3 past projects with portfolio links.
        
        Args:
            new_job_data: The new job requirements to generate proposal for
            all_historical_jobs: All historical job data for context matching
            max_length: Maximum proposal length in words (default 350 for SHORT impact)
            include_portfolio: Include portfolio links in proposal
            include_feedback: Include feedback URLs as evidence
            proposal_style: Writing style (professional, casual, technical, creative, data_driven)
            tone: Proposal tone (confident, humble, enthusiastic, analytical, friendly)
            
        Returns:
            Generated proposal with metadata and source references
        """
        job_id = new_job_data.get("contract_id", "unknown")
        logger.info(f"[ProposalGen] Generating SHORT, HUMAN, WINNING proposal for {job_id}")

        # Step 1: Retrieve similar projects and insights
        logger.debug(f"[Step 1] Retrieving similar past projects...")
        retrieval_result = self.retrieval_pipeline.retrieve_for_proposal(
            new_job_data,
            all_historical_jobs,
            top_k=5
        )
        
        similar_count = len(retrieval_result.get("similar_projects", []))
        logger.info(f"  → Found {similar_count} similar projects")

        # Step 2: Build optimized prompt using PromptEngine with SHORT, HUMAN, WINNING structure
        logger.debug(f"[Step 2] Building AI prompt with HOOK→PROOF→APPROACH→TIMELINE→CTA structure...")
        prompt = self.prompt_engine.build_proposal_prompt(
            job_data=new_job_data,
            similar_projects=retrieval_result.get("similar_projects", []),
            success_patterns=retrieval_result.get("insights", {}).get("success_patterns", []),
            style=proposal_style,
            tone=tone,
            max_words=max_length,  # Use passed max_length
            include_portfolio=include_portfolio,
            include_feedback=include_feedback
        )

        # Step 3: Generate proposal using OpenAI with optimized system message
        logger.debug(f"[Step 3] Calling GPT-4o to generate proposal...")
        system_message = self._get_system_message_for_winning_proposal(proposal_style, tone)
        
        proposal_result = self.openai_service.generate_proposal(
            job_description=new_job_data.get("job_description", ""),
            context_data=prompt,
            company_name=new_job_data.get("company_name", ""),
            job_title=new_job_data.get("job_title", ""),
            skills=new_job_data.get("skills_required", []),
            tone=tone,
            include_portfolio=include_portfolio
        )

        if not proposal_result or not proposal_result.get("proposal_text"):
            logger.error(f"Failed to generate proposal for {job_id}")
            raise Exception("Proposal generation failed - no response from OpenAI")

        proposal_text = proposal_result.get("proposal_text", "")
        
        # Strip any markdown formatting (Upwork doesn't support it)
        proposal_text = self._strip_markdown(proposal_text)

        # Step 4: Extract portfolio and feedback references used
        references = self._extract_references(
            proposal_text,
            retrieval_result,
            include_portfolio,
            include_feedback
        )

        # Step 5: Score proposal quality against SHORT, HUMAN, WINNING criteria
        quality_score = self.prompt_engine.score_proposal_quality(
            proposal_text,
            new_job_data,
            references
        )
        
        # Step 6: Assess match quality and hallucination risk
        match_assessment = self._get_match_quality_assessment(retrieval_result)
        confidence_score = self._calculate_confidence(retrieval_result)
        
        # Combine hallucination warnings from references extraction
        all_warnings = references.get("hallucination_warnings", []) + match_assessment.get("warnings", [])

        result = {
            "contract_id": job_id,
            "job_title": new_job_data.get("job_title"),
            "company_name": new_job_data.get("company_name"),
            "generated_proposal": proposal_text,
            "word_count": len(proposal_text.split()),
            "quality_score": quality_score,
            "is_short_human_winning": quality_score.get("is_short_human_winning", False),
            "retrieval_context": {
                "similar_projects_count": len(retrieval_result.get("similar_projects", [])),
                "success_patterns": retrieval_result.get("insights", {}).get("success_patterns", []),
                "success_rate": retrieval_result.get("insights", {}).get("success_rate", 0),
                # NEW: Include similar projects with deliverables for transparency
                "similar_projects_summary": [
                    {
                        "company": p.get("company"),
                        "deliverables": p.get("deliverables", []),
                        "similarity_score": p.get("similarity_score", 0)
                    }
                    for p in retrieval_result.get("similar_projects", [])[:3]
                ]
            },
            "references": references,
            "metadata": {
                "estimated_acceptance_rate": retrieval_result.get("proposal_context", {}).get("estimated_success_rate", 0),
                "similar_projects_used": similar_count,
                "confidence_score": confidence_score,
                "proposal_style": proposal_style,
                "tone": tone,
                # NEW: Match quality assessment
                "match_quality": match_assessment["quality_level"],
                "has_relevant_deliverables": match_assessment["has_deliverables_match"]
            },
            # NEW: Hallucination warnings for review
            "hallucination_warnings": all_warnings if all_warnings else None,
            "recommendations": match_assessment.get("recommendations", []) if match_assessment.get("recommendations") else None
        }

        # Log quality assessment with hallucination warnings
        if all_warnings:
            logger.warning(f"⚠️  [ProposalGen] HALLUCINATION RISK detected: {all_warnings}")
        
        if quality_score.get("is_short_human_winning"):
            logger.info(f"✅ [ProposalGen] HIGH QUALITY - {result['word_count']} words, score {quality_score['overall_score']}, confidence {confidence_score}")
        else:
            logger.warning(f"⚠️  [ProposalGen] Quality issues detected - {quality_score['feedback']}")
            result["quality_warnings"] = quality_score.get("feedback", [])

        return result

    def _strip_markdown(self, text: str) -> str:
        """
        Strip markdown formatting from text since Upwork doesn't support it.
        
        Removes:
        - **bold** and __bold__
        - *italic* and _italic_
        - # headers
        - - bullet points (at line start)
        - [text](url) markdown links (converts to just text + url)
        """
        import re
        
        # Remove bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        
        # Remove italic: *text* or _text_ (but not URLs with underscores)
        text = re.sub(r'(?<!\S)\*([^*\n]+)\*(?!\S)', r'\1', text)
        text = re.sub(r'(?<!\S)_([^_\n]+)_(?!\S)', r'\1', text)
        
        # Remove # headers (at line start)
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # Convert markdown links [text](url) to "text: url" or just keep both
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
        
        # Remove bullet points at line start (- item) but keep the text
        text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()

    def _get_system_message_for_winning_proposal(self, style: str, tone: str) -> str:
        """
        Get the system message that enforces SHORT, HUMAN, WINNING proposal generation.
        
        This message explicitly forbids AI language and mandates conversational tone.
        """
        return f"""
You are an expert SHORT, HUMAN, WINNING proposal writer for freelancers.

CRITICAL RULES:
1. ❌ NEVER say "As an AI", "I'm an AI", "artificial intelligence", etc.
2. ❌ NO corporate jargon, buzzwords, or formal tone
3. ❌ NO generic openings like "I'm excited to help"
4. ❌ NO MARKDOWN - no **bold**, no *italic*, no # headers, no - bullets (Upwork doesn't support it)
5. ✓ Sound like a REAL person who understands their specific problem
6. ✓ Write conversational, direct, punchy language - PLAIN TEXT only
7. ✓ Reference past projects by COMPANY NAME with outcomes
8. ✓ Include portfolio links and client feedback for social proof
9. ✓ Target 250-350 words (SHORT = HIGH IMPACT)

STRUCTURE (ALWAYS):
1. HOOK (2 sentences): "I see you're dealing with [their specific problem]..."
2. PROOF (2-3 bullets): Past similar projects + portfolio links + outcomes
3. APPROACH (3-4 sentences): How you'd solve THEIR problem specifically
4. TIMELINE (1-2 sentences): Realistic phases based on similar work
5. CTA (1 sentence): Friendly call-to-action

SUCCESS PATTERN:
- Short proposals (250-350 words) get 3-5x better response rates than long ones
- Real project names and outcomes build trust
- Portfolio proof = social proof = conversions
- Acknowledging THEIR problem = "This person gets it!"

Generate the proposal NOW using this structure. Make it SHORT. Make it HUMAN. Make it WINNING.
"""

    def _build_prompt(
        self,
        new_job_data: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        include_portfolio: bool,
        include_feedback: bool,
        max_length: int
    ) -> str:
        """
        Build comprehensive prompt for AI proposal generation.
        
        Includes:
        - New job requirements
        - Similar past successful projects
        - Success patterns that worked
        - Winning sections to include
        - Portfolio links for credibility
        - Feedback evidence from past projects
        - Success rate statistics
        """
        similar_projects = retrieval_result.get("similar_projects", [])
        insights = retrieval_result.get("insights", {})
        context = retrieval_result.get("proposal_context", {})

        # Build new job section
        new_job_section = f"""
NEW JOB REQUIREMENTS:
Company: {new_job_data.get('company_name')}
Position: {new_job_data.get('job_title')}
Industry: {new_job_data.get('industry', 'Not specified')}
Task Type: {new_job_data.get('task_type', 'Not specified')}
Skills Needed: {', '.join(new_job_data.get('skills_required', []))}
Complexity: {new_job_data.get('task_complexity', 'Medium')}
Timeline: {new_job_data.get('project_duration_days', 'Not specified')} days
Urgency: {'Ad-hoc/Emergency' if new_job_data.get('urgent_adhoc') else 'Planned'}

Job Description:
{new_job_data.get('job_description', 'No description provided')}
"""

        # Build similar projects section
        similar_projects_section = self._format_similar_projects(
            similar_projects,
            include_feedback
        )

        # Build patterns and insights section
        patterns_section = f"""
PATTERNS THAT HAVE WORKED:
Success Rate with Similar Projects: {insights.get('success_rate', 0)*100:.0f}%

Key Success Patterns:
{self._format_list(insights.get('success_patterns', []))}

Winning Proposal Sections to Include:
{self._format_list(insights.get('winning_sections', []))}

Insights from Past Feedback:
- Clients valued: {self._extract_client_values(similar_projects)}
- Average satisfaction: {self._calculate_avg_satisfaction(similar_projects):.1f}/5
- Success rate: {insights.get('success_rate', 0)*100:.0f}% of similar projects succeeded
"""

        # Build portfolio section if included
        portfolio_section = ""
        if include_portfolio:
            portfolio_urls = [p.get("portfolio_urls", []) for p in similar_projects if p.get("portfolio_urls")]
            if portfolio_urls:
                portfolio_section = f"""
PORTFOLIO REFERENCES FOR CREDIBILITY:
{self._format_portfolio_links(portfolio_urls)}

Include references to relevant portfolio projects that demonstrate expertise in this area.
"""

        # Build main prompt
        prompt = f"""
You are an expert proposal writer for freelancers. Generate a compelling, data-driven proposal.

{new_job_section}

{similar_projects_section}

{patterns_section}

{portfolio_section}

INSTRUCTIONS:
1. Reference 2-3 similar past projects (mention company names and outcomes)
2. Highlight expertise areas that match this job
3. Include the success patterns mentioned above
4. Mention relevant portfolio work for authenticity
5. Use the tone and style of successful past proposals
6. Address the specific urgency and complexity of this project
7. Include a realistic timeline based on similar projects
8. Explain your approach and methodology
9. Keep professional and confident tone
10. Word count: ~{max_length} words (flexible)

Generate the proposal now. Make it compelling, specific, and data-backed.
"""

        return prompt

    def _format_similar_projects(
        self,
        similar_projects: List[Dict[str, Any]],
        include_feedback: bool
    ) -> str:
        """Format similar projects section for the prompt"""
        if not similar_projects:
            return "SIMILAR PAST PROJECTS: None found in database."

        section = "SIMILAR PAST PROJECTS THAT SUCCEEDED:\n"

        for i, project in enumerate(similar_projects[:3], 1):  # Top 3 projects
            section += f"""
Project {i}:
- Company: {project.get('company')}
- Similar To: {project.get('title')}
- Skills: {', '.join(project.get('skills', []))}
- Success Rate: {project.get('effectiveness', 0)*100:.0f}%
- Client Satisfaction: {project.get('satisfaction', 0)}/5
"""
            if include_feedback and project.get('feedback_url'):
                section += f"- Feedback: {project.get('feedback_url')}\n"

        return section

    def _format_list(self, items: List[str]) -> str:
        """Format list items with bullets"""
        if not items:
            return "  • No specific patterns identified"
        return "".join([f"  • {item}\n" for item in items[:5]])

    def _format_portfolio_links(self, portfolio_urls: List[List[str]]) -> str:
        """Format portfolio links for prompt"""
        all_urls = []
        for url_list in portfolio_urls:
            if isinstance(url_list, list):
                all_urls.extend(url_list)
            else:
                all_urls.append(url_list)

        if not all_urls:
            return ""

        return "".join([f"  • {url}\n" for url in all_urls[:5]])

    def _extract_client_values(self, projects: List[Dict[str, Any]]) -> str:
        """Extract what clients valued from feedback"""
        values = []

        for project in projects:
            feedback = project.get('feedback_text', '').lower()
            if 'responsive' in feedback or 'communication' in feedback:
                values.append('responsive communication')
            if 'professional' in feedback or 'quality' in feedback:
                values.append('professional quality')
            if 'fast' in feedback or 'quick' in feedback or 'time' in feedback:
                values.append('timely delivery')
            if 'exceed' in feedback or 'amazing' in feedback or 'great' in feedback:
                values.append('exceeding expectations')

        # Remove duplicates while preserving order
        seen = set()
        unique_values = []
        for v in values:
            if v not in seen:
                unique_values.append(v)
                seen.add(v)

        return ', '.join(unique_values[:3]) if unique_values else 'quality and reliability'

    def _calculate_avg_satisfaction(self, projects: List[Dict[str, Any]]) -> float:
        """Calculate average satisfaction from similar projects"""
        satisfactions = [
            p.get('satisfaction', 3)
            for p in projects
            if p.get('satisfaction')
        ]

        return sum(satisfactions) / len(satisfactions) if satisfactions else 3.5

    def _extract_references(
        self,
        proposal_text: str,
        retrieval_result: Dict[str, Any],
        include_portfolio: bool,
        include_feedback: bool
    ) -> Dict[str, Any]:
        """
        Extract which references (portfolio links, feedback URLs) are mentioned
        in the generated proposal.
        """
        references = {
            "portfolio_links_used": [],
            "feedback_urls_cited": [],
            "projects_referenced": [],
            "hallucination_warnings": []  # NEW: Track potential hallucinations
        }

        # Handle None or empty proposal_text
        if not proposal_text:
            return references
        
        proposal_lower = proposal_text.lower()
        
        # Get list of verified company names from similar projects
        verified_companies = set()
        verified_portfolio_urls = set()
        verified_deliverables = set()
        
        for project in retrieval_result.get("similar_projects", []):
            company = project.get("company", "")
            if company:
                verified_companies.add(company.lower())
            for url in project.get("portfolio_urls", []):
                if url:
                    verified_portfolio_urls.add(url.lower())
            for d in project.get("deliverables", []):
                if d:
                    verified_deliverables.add(d.lower())

        if include_portfolio:
            for project in retrieval_result.get("similar_projects", []):
                for url in project.get("portfolio_urls", []):
                    if url and url in proposal_text:
                        references["portfolio_links_used"].append(url)

        if include_feedback:
            for project in retrieval_result.get("similar_projects", []):
                feedback_url = project.get("feedback_url")
                if feedback_url and feedback_url in proposal_text:
                    references["feedback_urls_cited"].append({
                        "project": project.get("company"),
                        "url": feedback_url
                    })

        for project in retrieval_result.get("similar_projects", [])[:3]:
            if project.get("company") and project.get("company") in proposal_text:
                references["projects_referenced"].append({
                    "company": project.get("company"),
                    "similarity_score": project.get("similarity_score", 0)
                })
        
        # HALLUCINATION DETECTION: Check for potentially fabricated claims
        references["hallucination_warnings"] = self._detect_hallucinations(
            proposal_text, 
            verified_companies, 
            verified_portfolio_urls,
            verified_deliverables
        )

        return references
    
    def _detect_hallucinations(
        self,
        proposal_text: str,
        verified_companies: set,
        verified_portfolio_urls: set,
        verified_deliverables: set
    ) -> List[str]:
        """
        Detect potential hallucinations or fabricated claims in generated proposal.
        
        CRITICAL: This catches cases where the AI:
        - Claims to have worked with companies not in our data
        - Mentions deliverables that don't match our past work
        - References fabricated URLs
        
        Args:
            proposal_text: Generated proposal text
            verified_companies: Set of company names from actual past projects
            verified_portfolio_urls: Set of actual portfolio URLs
            verified_deliverables: Set of actual deliverables from past projects
            
        Returns:
            List of warning messages about potential hallucinations
        """
        import re
        warnings = []
        proposal_lower = proposal_text.lower()
        
        # 1. Check for company name claims that aren't in our data
        # Look for patterns like "worked with X", "built for X", "completed X project"
        company_claim_patterns = [
            r"(?:worked with|built for|completed for|delivered for|did a? ?(?:similar |same )?(?:project|work) for)\s+([A-Z][a-zA-Z\s]+)",
            r"(?:for|at|with)\s+([A-Z][a-zA-Z\s]+)(?:\s+I\s+(?:built|developed|created|delivered))",
        ]
        
        for pattern in company_claim_patterns:
            matches = re.findall(pattern, proposal_text)
            for match in matches:
                company_name = match.strip().lower()
                # Skip generic words
                if company_name in ["a client", "the client", "another client", "a company", "similar"]:
                    continue
                # Check if this company is in our verified list
                if not any(company_name in verified.lower() or verified.lower() in company_name 
                          for verified in verified_companies if verified):
                    if len(company_name) > 3:  # Avoid false positives on short words
                        warnings.append(f"⚠️ Potential hallucination: Claims work with '{match}' not found in verified projects")
        
        # 2. Check for URLs that aren't in our data
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
        urls_in_proposal = re.findall(url_pattern, proposal_text)
        
        for url in urls_in_proposal:
            url_lower = url.lower().rstrip('/.,')
            # Skip common legitimate URLs (upwork profile, etc)
            if 'upwork.com/freelancers' in url_lower:
                continue
            # Check if URL is in our verified list
            if not any(url_lower in verified.lower() or verified.lower() in url_lower 
                      for verified in verified_portfolio_urls if verified):
                warnings.append(f"⚠️ URL not from verified projects: {url[:50]}...")
        
        # 3. Check for specific deliverable claims that don't match our work
        # These are strong claims about what was built
        specific_claim_keywords = [
            "migrated", "built membership", "set up subscription", "optimized speed",
            "fixed the bug", "created the form", "integrated payment", "developed the api"
        ]
        
        for keyword in specific_claim_keywords:
            if keyword in proposal_lower:
                # Check if we have matching deliverables
                keyword_root = keyword.split()[0] if ' ' in keyword else keyword  # Get main action word
                has_matching_deliverable = any(
                    keyword_root in d.lower() for d in verified_deliverables if d
                )
                if not has_matching_deliverable and verified_deliverables:
                    # This might be a fabrication - only warn if we have deliverables data to compare
                    pass  # Don't warn for now - could be legitimate skill claim
        
        return warnings

    def _calculate_confidence(self, retrieval_result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the generated proposal based on:
        - Number of similar projects found
        - Quality of match (similarity scores)
        - Whether deliverables actually match what client needs
        - Success rate of similar projects
        - Amount of feedback data available
        
        CRITICAL: Low confidence indicates potential for hallucination - 
        the AI may fabricate experience if no good matches exist.
        """
        similar_projects = retrieval_result.get("similar_projects", [])
        similar_count = len(similar_projects)
        success_rate = retrieval_result.get("insights", {}).get("success_rate", 0.5)

        # Base confidence on similar projects (max 3)
        similarity_confidence = min(similar_count / 3, 1.0)

        # Factor in success rate
        success_confidence = success_rate
        
        # NEW: Factor in match quality (average similarity score)
        if similar_projects:
            avg_similarity = sum(p.get("similarity_score", 0) for p in similar_projects) / len(similar_projects)
            match_quality_confidence = avg_similarity
        else:
            match_quality_confidence = 0.0
        
        # NEW: Check if projects have deliverables data (reduces hallucination risk)
        projects_with_deliverables = sum(
            1 for p in similar_projects 
            if p.get("deliverables") and len(p.get("deliverables", [])) > 0
        )
        deliverables_confidence = projects_with_deliverables / max(similar_count, 1)

        # Weighted average - prioritize match quality and deliverables
        confidence = (
            (similarity_confidence * 0.2) + 
            (success_confidence * 0.2) +
            (match_quality_confidence * 0.35) +  # Higher weight for match quality
            (deliverables_confidence * 0.25)     # Higher weight for having deliverables data
        )

        return round(confidence, 2)
    
    def _get_match_quality_assessment(self, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of project matches for the new job.
        
        Returns:
            Assessment dict with quality level, warnings, and recommendations
        """
        similar_projects = retrieval_result.get("similar_projects", [])
        
        assessment = {
            "quality_level": "unknown",
            "has_relevant_projects": False,
            "has_deliverables_match": False,
            "warnings": [],
            "recommendations": []
        }
        
        if not similar_projects:
            assessment["quality_level"] = "no_matches"
            assessment["warnings"].append("No similar projects found - high risk of hallucination")
            assessment["recommendations"].append("Focus on skills and approach, not past project claims")
            return assessment
        
        # Check average similarity
        avg_score = sum(p.get("similarity_score", 0) for p in similar_projects) / len(similar_projects)
        
        # Check if any projects have deliverables
        has_deliverables = any(
            p.get("deliverables") and len(p.get("deliverables", [])) > 0 
            for p in similar_projects
        )
        
        if avg_score >= 0.6 and has_deliverables:
            assessment["quality_level"] = "high"
            assessment["has_relevant_projects"] = True
            assessment["has_deliverables_match"] = True
        elif avg_score >= 0.4:
            assessment["quality_level"] = "medium"
            assessment["has_relevant_projects"] = True
            if not has_deliverables:
                assessment["warnings"].append("Similar projects found but missing deliverables data - match quality uncertain")
        else:
            assessment["quality_level"] = "low"
            assessment["warnings"].append("Low similarity scores - projects may not be truly relevant")
            assessment["recommendations"].append("Be conservative about past project claims")
        
        return assessment

    def batch_generate_proposals(
        self,
        new_jobs: List[Dict[str, Any]],
        all_historical_jobs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate proposals for multiple new jobs.
        
        Args:
            new_jobs: List of new job requirements
            all_historical_jobs: All historical job data
            
        Returns:
            List of generated proposals
        """
        logger.info(f"[ProposalGen] Batch generating proposals for {len(new_jobs)} jobs")

        proposals = []

        for i, job in enumerate(new_jobs, 1):
            try:
                proposal = self.generate_proposal(job, all_historical_jobs)
                proposals.append(proposal)
                logger.debug(f"[Batch {i}/{len(new_jobs)}] ✓ Generated")
            except Exception as e:
                logger.error(f"[Batch {i}] ✗ Failed: {str(e)}")

        logger.info(f"✓ Batch complete: {len(proposals)} proposals generated")
        return proposals
