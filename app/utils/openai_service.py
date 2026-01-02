import logging
import base64
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI, AsyncOpenAI
import tiktoken
from tenacity import retry, wait_exponential, stop_after_attempt
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class OpenAIService:
    """Service for handling OpenAI API interactions including Vision API for OCR, embeddings, and text generation"""
    
    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o",
        vision_model: str = "gpt-4o"
    ):
        """
        Initialize OpenAI service with sync and async clients
        
        Args:
            api_key: OpenAI API key
            embedding_model: Model for embeddings (text-embedding-3-large = 3072 dimensions)
            llm_model: Model for text generation
            vision_model: Model for vision/OCR tasks
        """
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        logger.info(f"OpenAI service initialized - Embedding: {embedding_model}, LLM: {llm_model}, Vision: {vision_model}")
    
    # ===================== VISION API / OCR FUNCTIONS =====================
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def extract_text_from_image(self, image_source: str, is_url: bool = False) -> str:
        """
        Extract text from image using GPT-4o Vision API (OCR)
        
        Args:
            image_source: Either file path or URL of image
            is_url: If True, image_source is treated as URL; if False, as file path
            
        Returns:
            Extracted text from image
        """
        try:
            logger.info(f"Extracting text from image: {image_source[:50]}...")
            
            if is_url:
                # Use URL directly
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_source,
                        "detail": "high"  # high detail for better OCR
                    }
                }
            else:
                # Read image file and encode to base64
                with open(image_source, "rb") as image_file:
                    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
                
                # Determine image type from file extension
                file_extension = image_source.split(".")[-1].lower()
                media_type_map = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "gif": "image/gif",
                    "webp": "image/webp"
                }
                media_type = media_type_map.get(file_extension, "image/jpeg")
                
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}",
                        "detail": "high"
                    }
                }
            
            # Call Vision API
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {
                                "type": "text",
                                "text": """Please extract ALL text from this image. 
                                
Extract:
1. All written text
2. All typed text
3. All labels and headers
4. All numbers and dates
5. Any feedback or review content

Format the output clearly, maintaining the structure as much as possible. 
If this is a review/feedback screenshot, extract the complete feedback text."""
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            extracted_text = response.choices[0].message.content
            logger.info(f"Successfully extracted {len(extracted_text)} characters from image")
            return extracted_text
        except FileNotFoundError:
            logger. error(f"Image file not found: {image_source}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            raise
    
    async def extract_text_from_image_async(self, image_source: str, is_url: bool = False) -> str:
        """
        Async version of extract_text_from_image
        
        Args:
            image_source: Either file path or URL of image
            is_url: If True, image_source is treated as URL
            
        Returns:
            Extracted text from image
        """
        try:
            logger.info(f"[ASYNC] Extracting text from image: {image_source[:50]}...")
            
            if is_url:
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_source,
                        "detail": "high"
                    }
                }
            else:
                with open(image_source, "rb") as image_file:
                    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
                
                file_extension = image_source.split(".")[-1].lower()
                media_type_map = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "gif": "image/gif",
                    "webp": "image/webp"
                }
                media_type = media_type_map.get(file_extension, "image/jpeg")
                
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}",
                        "detail": "high"
                    }
                }
            
            response = await self.async_client. chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {
                                "type": "text",
                                "text": """Please extract ALL text from this image clearly and completely."""
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            extracted_text = response.choices[0]. message.content
            logger.info(f"[ASYNC] Successfully extracted {len(extracted_text)} characters from image")
            return extracted_text
        except Exception as e:
            logger.error(f"[ASYNC] Error extracting text from image: {str(e)}")
            raise
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def analyze_feedback_sentiment(self, feedback_text: str) -> Dict[str, Any]:
        """
        Analyze sentiment and extract key insights from client feedback
        
        Args:
            feedback_text: Client feedback text
            
        Returns:
            Dictionary with sentiment analysis and key points
        """
        try:
            logger.info("Analyzing feedback sentiment...")
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing client feedback and extracting insights."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this client feedback and provide:
1. Overall sentiment (positive, neutral, negative)
2.  Sentiment score (0-100)
3. Key positive points (list)
4. Key negative points or areas for improvement (list)
5.  Suggested improvements for future proposals
6. One-line summary

Feedback:
{feedback_text}

Respond in JSON format."""
                    }
                ],
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\{.*\}', response_text, re. DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = {"raw_analysis": response_text}
            except json.JSONDecodeError:
                analysis = {"raw_analysis": response_text}
            
            logger.info("Feedback sentiment analysis completed")
            return analysis
        except Exception as e:
            logger. error(f"Error analyzing feedback sentiment: {str(e)}")
            return {"error": str(e)}
    
    # ===================== EMBEDDING FUNCTIONS =====================
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def get_embedding(self, text: str, dimensions: int = 3072) -> List[float]:
        """
        Get embedding for a single text using text-embedding-3-large
        
        Args:
            text: Text to embed
            dimensions: Embedding dimensions (default 3072 for text-embedding-3-large)
            
        Returns:
            List of floats representing the embedding
        """
        try:
            # Ensure text is not empty
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * dimensions
            
            response = self.client.embeddings. create(
                input=text. strip(),
                model=self.embedding_model,
                dimensions=dimensions
            )
            
            logger.debug(f"Generated embedding with {len(response. data[0].embedding)} dimensions")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        dimensions: int = 3072,
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch
        Processes in chunks to handle API limits
        
        Args:
            texts: List of texts to embed
            dimensions: Embedding dimensions
            batch_size: Size of each batch to process
            
        Returns:
            List of embeddings in same order as input
        """
        try:
            # Filter empty texts but keep track of indices
            valid_texts = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
            
            if not valid_texts:
                logger.warning(f"No valid texts provided for batch embedding (input had {len(texts)} texts)")
                return []  # Return empty list instead of zero embeddings
            
            all_embeddings = [None] * len(texts)
            
            # Process in batches
            for batch_start in range(0, len(valid_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_texts))
                batch_texts = [t[1] for t in valid_texts[batch_start:batch_end]]
                batch_indices = [t[0] for t in valid_texts[batch_start:batch_end]]
                
                response = self.client.embeddings. create(
                    input=batch_texts,
                    model=self.embedding_model,
                    dimensions=dimensions
                )
                
                # Map embeddings back to original positions
                for response_idx, (text_idx, embedding_data) in enumerate(zip(batch_indices, response.data)):
                    all_embeddings[text_idx] = embedding_data. embedding
                
                logger.info(f"Processed embeddings batch {batch_start//batch_size + 1}")
            
            logger.info(f"Generated {len([e for e in all_embeddings if e])} embeddings in batch")
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    async def get_embeddings_batch_async(
        self,
        texts: List[str],
        dimensions: int = 3072,
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Async batch embeddings for better concurrency
        
        Args:
            texts: List of texts to embed
            dimensions: Embedding dimensions
            batch_size: Batch size
            
        Returns:
            List of embeddings
        """
        try:
            valid_texts = [(i, t. strip()) for i, t in enumerate(texts) if t and t. strip()]
            
            if not valid_texts:
                return [[0.0] * dimensions] * len(texts)
            
            all_embeddings = [None] * len(texts)
            tasks = []
            
            # Create tasks for all batches
            for batch_start in range(0, len(valid_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_texts))
                batch_texts = [t[1] for t in valid_texts[batch_start:batch_end]]
                batch_indices = [t[0] for t in valid_texts[batch_start:batch_end]]
                
                task = self._process_embedding_batch(batch_texts, batch_indices, dimensions)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Consolidate results
            for result in results:
                for text_idx, embedding in result:
                    all_embeddings[text_idx] = embedding
            
            logger.info(f"[ASYNC] Generated embeddings for {len(texts)} texts")
            return all_embeddings
        except Exception as e:
            logger.error(f"[ASYNC] Error generating batch embeddings: {str(e)}")
            raise
    
    async def _process_embedding_batch(
        self,
        texts: List[str],
        indices: List[int],
        dimensions: int
    ) -> List[Tuple[int, List[float]]]:
        """Helper to process single embedding batch asynchronously"""
        response = await self.async_client.embeddings.create(
            input=texts,
            model=self.embedding_model,
            dimensions=dimensions
        )
        return [(idx, data.embedding) for idx, data in zip(indices, response.data)]
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text for a given model
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return len(text. split())  # Fallback to word count
    
    # ===================== TEXT GENERATION FUNCTIONS =====================
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        json_mode: bool = False
    ) -> str:
        """
        Generate text using GPT-4o
        
        Args:
            prompt: User prompt
            system_message: System message for context
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top p for nucleus sampling
            json_mode: If True, response will be valid JSON
            
        Returns:
            Generated text
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            kwargs = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            
            generated_text = response.choices[0].message.content
            logger.info(f"Generated text with {response.usage.completion_tokens} tokens")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def generate_text_async(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9
    ) -> str:
        """Async version of generate_text"""
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.async_client. chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"[ASYNC] Generated text with {response.usage.completion_tokens} tokens")
            return generated_text
        except Exception as e:
            logger.error(f"[ASYNC] Error generating text: {str(e)}")
            raise
    
    def generate_proposal(
        self,
        job_description: str,
        context_data: str,
        company_name: str,
        job_title: str,
        skills: List[str],
        tone: str = "professional",
        include_portfolio: bool = True,
        portfolio_url: str = ""
    ) -> Dict[str, str]:
        """
        Generate a SHORT, HUMAN, WINNING proposal using retrieved context.
        
        Target: 250-350 words, conversational tone, references to past projects.
        
        Args:
            job_description: Description of the job
            context_data: Retrieved context from vector DB (past proposals, projects, reviews)
            company_name: Name of the company
            job_title: Title of the job
            skills: Required skills
            tone: Tone of proposal (professional, casual, technical)
            include_portfolio: Whether to include portfolio link
            portfolio_url: Portfolio URL
            
        Returns:
            Dictionary with proposal_text and metadata
        """
        system_message = """You are a SHORT, HUMAN, WINNING proposal writer for freelancers.

YOUR MINDSET: You're a skilled freelancer who GENUINELY wants to help. You read every job post carefully and connect with the client's actual problem - not just the task they described.

🧠 HUMAN CONNECTION FORMULA:
1. FEEL their frustration/urgency (empathize, don't just acknowledge)
2. SHOW you read their specific job post (reference EXACT details they mentioned)
3. PROVE you've solved this EXACT problem before (with portfolio links)
4. EXPLAIN your specific approach for THEIR situation (not generic)
5. MAKE IT EASY to say yes (friendly, low-pressure CTA)

CRITICAL RULES:
1. ❌ NEVER say "As an AI", "I'm an AI", or any AI language
2. ❌ NO corporate jargon, buzzwords, or formal tone
3. ❌ NO generic openers like "I'm excited to help" or "I came across your project"
4. ❌ NO robotic phrases like "I would be delighted", "I am eager to"
5. ✓ Sound like a REAL person having a coffee chat with a potential client
6. ✓ Reference their SPECIFIC problem (numbers, tools, pain points they mentioned)
7. ✓ Use contractions naturally (I've, you're, that's, don't)
8. ✓ Reference past projects by COMPANY NAME with outcomes
9. ✓ Include portfolio links for social proof
10. ✓ Target 150-250 words (SHORT = HIGH IMPACT)

STRUCTURE (ALWAYS):
1. HOOK (1-2 sentences): Show you GET their specific problem + portfolio link
2. PROOF (2-3 bullets): Past similar projects + portfolio + outcomes
3. APPROACH (2-3 sentences): How you'd solve THEIR problem specifically
4. CTA (1 casual sentence): Friendly, low-pressure next step

EXAMPLE HOOKS (conversational, specific, human):
✓ "Migrating 5000 subscribers from Substack while keeping membership tiers intact - that's no small task. Just wrapped up something similar..."
✓ "30-second checkout times with 2000+ products - I know exactly why that's happening and how to fix it."
✓ "Sounds like your last developer left you in a tough spot. Let me show you a cleaner approach..."

Remember: Short proposals get 3-5x better response rates. Every word counts. Sound like a helpful human, not a salesperson."""
        
        portfolio_section = f"\n\n**Portfolio:** {portfolio_url}" if include_portfolio and portfolio_url else ""
        
        prompt = f"""Generate a SHORT, HUMAN, WINNING proposal based on this job opportunity:

**Job Opportunity:**
Company: {company_name}
Job Title: {job_title}
Job Description: {job_description}

**Required Skills:** {', '.join(skills)}

**Your Historical Context & Relevant Experience:**
{context_data}
{portfolio_section}

**Generate a proposal that:**
- Is 150-250 words (SHORT = HIGH IMPACT)
- Opens by acknowledging THEIR SPECIFIC problem (use exact details from their job post)
- Shows genuine empathy for their situation
- References 2-3 past similar projects by company name with portfolio links
- Explains your specific approach for THEIR tech stack and situation
- Ends with a casual, friendly CTA (like texting a colleague)
- Sounds like a REAL helpful human, NOT AI or a salesperson
- Uses natural contractions and conversational language"""
        
        proposal_text = self.generate_text(
            prompt=prompt,
            system_message=system_message,
            temperature=0.70,  # Slightly higher for natural human tone
            max_tokens=2500  # Increased to prevent truncation - allows full HOOK→PROOF→APPROACH→CTA structure
        )
        
        return {
            "proposal_text": proposal_text,
            "company_name": company_name,
            "job_title": job_title,
            "generated_at": datetime.utcnow().isoformat(),
            "tone": tone
        }
    
    def improve_proposal(
        self,
        original_proposal: str,
        feedback: str,
        specific_improvements: Optional[List[str]] = None
    ) -> str:
        """
        Improve an existing proposal based on feedback
        
        Args:
            original_proposal: Original proposal text
            feedback: Feedback on the proposal
            specific_improvements: Specific areas to improve
            
        Returns:
            Improved proposal text
        """
        improvements_text = ""
        if specific_improvements:
            improvements_text = f"\n\nSpecific areas to improve:\n" + "\n".join(f"- {imp}" for imp in specific_improvements)
        
        prompt = f"""Please improve the following proposal based on the feedback provided. 

**Original Proposal:**
{original_proposal}

**Feedback:**
{feedback}
{improvements_text}

Please rewrite the proposal to address all feedback points while maintaining professionalism and impact."""
        
        improved_proposal = self.generate_text(
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        return improved_proposal
    
    def extract_proposal_insights(self, proposal_text: str) -> Dict[str, Any]:
        """
        Extract key insights and metrics from a proposal
        
        Args:
            proposal_text: Proposal text
            
        Returns:
            Dictionary with insights
        """
        try:
            prompt = f"""Analyze this proposal and extract:
1. Key value propositions (list)
2.  Mentioned achievements/metrics (list)
3. Timeline mentioned (if any)
4. Investment/pricing (if mentioned)
5. Call-to-action
6. Overall tone assessment
7. Confidence score (0-100)

Proposal:
{proposal_text}

Respond in JSON format."""
            
            response = self. generate_text(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000,
                json_mode=True
            )
            
            try:
                insights = json.loads(response)
            except json.JSONDecodeError:
                insights = {"raw_insights": response}
            
            return insights
        except Exception as e:
            logger.error(f"Error extracting proposal insights: {str(e)}")
            return {"error": str(e)}