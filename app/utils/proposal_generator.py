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
        max_length: int = 2000,
        include_portfolio: bool = True,
        include_feedback: bool = True,
        proposal_style: str = "professional",
        tone: str = "confident"
    ) -> Dict[str, Any]:
        """
        Generate a complete proposal for a new job using similar past projects.
        
        Args:
            new_job_data: The new job requirements to generate proposal for
            all_historical_jobs: All historical job data for context matching
            max_length: Maximum proposal length in words
            include_portfolio: Include portfolio links in proposal
            include_feedback: Include feedback URLs as evidence
            proposal_style: Writing style (professional, casual, technical, creative, data_driven)
            tone: Proposal tone (confident, humble, enthusiastic, analytical, friendly)
            
        Returns:
            Generated proposal with metadata and source references
        """
        job_id = new_job_data.get("contract_id", "unknown")
        logger.info(f"[ProposalGen] Generating proposal for {job_id} (style={proposal_style}, tone={tone})")

        # Step 1: Retrieve similar projects and insights
        logger.debug(f"[Step 1] Retrieving context...")
        retrieval_result = self.retrieval_pipeline.retrieve_for_proposal(
            new_job_data,
            all_historical_jobs,
            top_k=5
        )

        # Step 2: Build optimized prompt using PromptEngine
        logger.debug(f"[Step 2] Building AI prompt with PromptEngine...")
        prompt = self.prompt_engine.build_proposal_prompt(
            job_data=new_job_data,
            similar_projects=retrieval_result.get("similar_projects", []),
            success_patterns=retrieval_result.get("insights", {}).get("success_patterns", []),
            style=proposal_style,
            tone=tone,
            max_words=max_length,
            include_portfolio=include_portfolio,
            include_feedback=include_feedback
        )

        # Step 3: Generate proposal using OpenAI
        logger.debug(f"[Step 3] Calling OpenAI to generate proposal...")
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

        # Step 4: Extract portfolio and feedback references used
        references = self._extract_references(
            proposal_text,
            retrieval_result,
            include_portfolio,
            include_feedback
        )

        result = {
            "contract_id": job_id,
            "job_title": new_job_data.get("job_title"),
            "company_name": new_job_data.get("company_name"),
            "generated_proposal": proposal_text,
            "word_count": len(proposal_text.split()),
            "retrieval_context": {
                "similar_projects_count": len(retrieval_result.get("similar_projects", [])),
                "success_patterns": retrieval_result.get("insights", {}).get("success_patterns", []),
                "winning_sections": retrieval_result.get("insights", {}).get("winning_sections", []),
                "success_rate": retrieval_result.get("insights", {}).get("success_rate", 0)
            },
            "references": references,
            "metadata": {
                "estimated_acceptance_rate": retrieval_result.get("proposal_context", {}).get("estimated_success_rate", 0),
                "similar_projects_used": len(retrieval_result.get("similar_projects", [])),
                "confidence_score": self._calculate_confidence(retrieval_result)
            }
        }

        logger.info(f"✓ [ProposalGen] Generated {result['word_count']} word proposal for {job_id}")
        return result

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
            "projects_referenced": []
        }

        # Handle None or empty proposal_text
        if not proposal_text:
            return references

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

        return references

    def _calculate_confidence(self, retrieval_result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the generated proposal based on:
        - Number of similar projects found
        - Success rate of similar projects
        - Amount of feedback data available
        """
        similar_count = len(retrieval_result.get("similar_projects", []))
        success_rate = retrieval_result.get("insights", {}).get("success_rate", 0.5)

        # Base confidence on similar projects (max 3)
        similarity_confidence = min(similar_count / 3, 1.0)

        # Factor in success rate
        success_confidence = success_rate

        # Weighted average
        confidence = (similarity_confidence * 0.4) + (success_confidence * 0.6)

        return round(confidence, 2)

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
