# Copilot Instructions: Upwork Proposal Generator

## Project Overview

**AI Proposal Generator** is a FastAPI-based system that generates short, personalized, winning Upwork proposals. It learns from 23+ historical winning proposals to create responses that reference relevant past projects, include portfolio proof, and use conversational human language—targeting 3-5x better response rates than generic proposals.

**Core Pattern**: New job description → Retrieve similar past projects (metadata filtering + semantic search) → Build enriched prompt → Generate SHORT (250-350 words), HUMAN, WINNING proposal.

## Architecture Quick Reference

### 3-Layer Data Flow
1. **Job Data Ingestion** (`app/routes/job_data_ingestion.py`): Upload training data, extract text from job descriptions via OCR
2. **Retrieval Pipeline** (`app/utils/retrieval_pipeline.py`): Multi-stage search—metadata filter → semantic search via Pinecone → feedback analysis
3. **Proposal Generation** (`app/utils/proposal_generator.py`): Build context from retrieved projects → Call GPT-4o → Return proposal with source references

### Key Components
- **MongoDB**: Stores job data, chunks, feedback, and metadata
- **Pinecone**: Vector DB for semantic search using text-embedding-3-large (3072 dimensions)
- **OpenAI**: GPT-4o for text generation, text-embedding-3-large for embeddings, gpt-4o for OCR
- **FastAPI**: Two main routers: `/api/job-data` (ingestion), `/api/proposals` (generation)

### Critical Services (all in `app/utils/`)
- `openai_service.py`: Wrapper for embeddings, LLM calls, Vision API (OCR)
- `pinecone_service.py`: Vector DB initialization and semantic search
- `prompt_engine.py`: Constructs optimized prompts with style/tone options (professional, casual, technical, etc.)
- `data_chunker.py` & `advanced_chunker.py`: Split job data into 5 semantic layers for better retrieval
- `feedback_processor.py`: Extract success patterns from client feedback
- `metadata_extractor.py`: Parse skills, industries, complexity, satisfaction scores

## Essential Workflows

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server (port 8000)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, test proposal generation
curl -X POST http://localhost:8000/api/proposals/generate \
  -H "Content-Type: application/json" \
  -d '{"job_title":"Senior Backend Dev","company_name":"TechCorp","job_description":"...","skills_required":["Python"]}'
```

### Development & Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_advanced_chunker.py -v

# Check syntax/imports
python -m py_compile app/utils/*.py

# Environment setup
source env/bin/activate  # Already set up in workspace
```

### Data Ingestion Workflow
1. **Upload training data**: POST `/api/job-data/upload` with `JobDataUploadRequest` (job metadata + proposal text)
2. **System processes**: Chunks job data into 5 semantic layers, generates embeddings, stores in MongoDB & Pinecone
3. **Verify**: GET `/api/job-data/list` to confirm data persisted

### Proposal Generation Flow
1. **POST** `/api/proposals/generate` with `GenerateProposalRequest`
   - Required: `job_title`, `company_name`, `job_description`, `skills_required`
   - Optional: `industry`, `task_type`, `proposal_style`, `tone` (see `ProposalStyle` and `ProposalTone` enums in `prompt_engine.py`)
2. **System retrieves** top-K similar past projects via `RetrievalPipeline.retrieve_for_proposal()`
3. **System builds prompt** using `PromptEngine` with matched projects + portfolio URLs + feedback
4. **GPT-4o generates** SHORT, HUMAN proposal (250-350 words default)
5. **Response** includes generated text + source references (matched projects + portfolio URLs)

## Code Patterns & Conventions

### Error Handling
- Use `HTTPException` with status codes in routes
- All services log errors to `logger.error()` before raising
- Tenacity retry decorator for API calls: `@retry(wait=wait_exponential(...), stop=stop_after_attempt(3))`

### Service Initialization
- All services accept optional dependencies (e.g., `ProposalGenerator(openai_service=None, retrieval_pipeline=None)`)
- Lazy initialization in routes (see `_processor`, `_pinecone_service` in `job_data_ingestion.py`)
- Settings from `app/config.py` using environment variables with defaults

### Pydantic Models
- All API inputs use `BaseModel` with `Field()` for validation and documentation (see `job_data_schema.py`)
- Response models mirror request structure but add fields like `success`, `message`, `timestamp`
- Enums for constrained fields: `ProjectStatus`, `TaskType`, `ChunkType`, `ProposalStyle`, `ProposalTone`

### Embeddings & Vectors
- **Model**: text-embedding-3-large (3072 dimensions)
- **Chunk size**: Default 1024 chars with 200-char overlap (tunable in `config.py`)
- **Pinecone namespace**: "proposals" (keeps data isolated)
- **Similarity threshold**: MIN_SIMILARITY_SCORE (default 0.5) in config

### Proposal Writing Style
The generator targets this structure (from `SYSTEM_PROMPT_FOR_AI.md`):
1. **HOOK** (2 sentences): Acknowledge their specific problem
2. **PROOF** (2-3 bullets): Show past similar projects + portfolio links
3. **APPROACH** (3-4 sentences): Specific solution for their tech stack
4. **TIMELINE** (1-2 sentences): Phases and duration
5. **CTA** (1 sentence): Call to action
**Total**: 250-350 words, conversational tone, human language (no "As an AI", no corporate jargon)

## Testing & Validation

### Test Structure
- Tests in `test/` directory, import from parent via `sys.path`
- Use `pytest` + `pytest-asyncio` for async tests
- Sample test data in test file globals (see `test_advanced_chunker.py`)
- Tests validate: chunking logic, retrieval ranking, proposal generation quality

### Common Test Patterns
```python
# Example from test_advanced_chunker.py
from app.utils.advanced_chunker import AdvancedChunkProcessor
processor = AdvancedChunkProcessor()
chunks = processor.chunk(job_data)
assert len(chunks) > 0
assert all(c.get("chunk_type") in ChunkTypeEnum.__members__.values() for c in chunks)
```

## Configuration & Environment

### Key Environment Variables (in `.env`)
- `OPENAI_API_KEY`: Required for embeddings + proposal generation
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`: Vector DB access
- `MONGODB_URI`, `MONGODB_DB_NAME`: Job data storage
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Tuning chunking strategy
- `PROPOSAL_TOP_K`: Number of similar projects to retrieve (default 5)
- `MIN_SIMILARITY_SCORE`: Semantic search threshold (default 0.5)

### Config Hierarchy
1. Environment variables (highest priority)
2. Defaults in `app/config.py`
3. Hardcoded in services (fallback)

## Common Modification Points

### Adding New Proposal Styles
1. Add enum variant in `ProposalStyle` (in `prompt_engine.py`)
2. Add style-specific instructions in `PromptEngine.STYLE_INSTRUCTIONS` dict
3. Route will auto-support it via request validation

### Improving Retrieval Quality
1. Adjust `CHUNK_SIZE` and `MIN_CHUNK_SIZE` in `config.py` to balance context vs precision
2. Modify `RetrievalPipeline.retrieve_for_proposal()` to add/remove filter criteria
3. Tune `MIN_SIMILARITY_SCORE` for stricter/looser semantic matches

### Customizing Proposal Output
1. Edit `PromptEngine.build_proposal_prompt()` to change prompt template
2. Adjust temperature/max_tokens in `OpenAIService.generate_text()` for more/less randomness
3. Add post-processing in `ProposalGenerator.generate_proposal()` to reformatting/validation

## Data Format Reference

### Training Data JSON (`complete_accurate_training_pairs.json`)
```json
{
  "projects": [
    {
      "contract_id": "unique_id",
      "company_name": "Client Corp",
      "job_title": "Backend Engineer",
      "job_description": "Full description...",
      "your_proposal_text": "What you proposed...",
      "skills_required": ["Python", "FastAPI"],
      "industry": "technology",
      "task_type": "backend_api",
      "start_date": "2024-01-01",
      "end_date": "2024-06-01",
      "project_status": "completed",
      "client_feedback_text": "Great work!",
      "client_feedback_url": "https://upwork.com/reviews/...",
      "portfolio_urls": ["https://github.com/...", "https://portfolio.com"],
      "effectiveness_score": 0.95,
      "satisfaction_score": 4.8
    }
  ]
}
```

### API Request Example (`GenerateProposalRequest`)
```json
{
  "job_title": "Senior Backend Developer",
  "company_name": "FinTech Startup",
  "job_description": "Looking for an experienced backend engineer...",
  "skills_required": ["Python", "FastAPI", "PostgreSQL"],
  "industry": "fintech",
  "task_type": "backend_api",
  "proposal_style": "professional",
  "tone": "confident",
  "max_word_count": 300,
  "include_portfolio": true,
  "include_feedback": true
}
```

## Debugging Tips

### Common Issues

**Proposal too long/short:**
- Check `max_word_count` in request (default 500 in schema, but prompt targets 250-350)
- Adjust `temperature` in `OpenAIService.generate_text()` (higher = more creative/varied length)

**Pinecone semantic search returning irrelevant results:**
- Verify embeddings dimension matches (3072 for text-embedding-3-large)
- Increase `MIN_SIMILARITY_SCORE` in config to filter low-scoring matches
- Check that chunks were properly uploaded (`GET /api/job-data/list` should show chunk count)

**Missing portfolio links in proposal:**
- Ensure training data includes `portfolio_urls` field
- Check `include_portfolio=true` in request
- Verify `PromptEngine.build_proposal_prompt()` includes portfolio context

### Logging
- All services log to `logger` (configured in `main.py` at INFO level)
- Check terminal output during request for detailed retrieval/generation logs
- Enable DEBUG mode in `.env` for verbose logs

## Reference Files by Purpose

| Purpose | Key Files |
|---------|-----------|
| **API entry point** | `main.py` |
| **Job ingestion routes** | `app/routes/job_data_ingestion.py` |
| **Proposal generation routes** | `app/routes/proposals.py` |
| **Retrieval strategy** | `app/utils/retrieval_pipeline.py` |
| **Prompt templates** | `app/utils/prompt_engine.py` |
| **OpenAI integration** | `app/utils/openai_service.py` |
| **Vector DB management** | `app/utils/pinecone_service.py` |
| **Data chunking** | `app/utils/advanced_chunker.py`, `app/utils/data_chunker.py` |
| **MongoDB schema** | `app/models/job_data_schema.py` |
| **Configuration** | `app/config.py` |
| **Database layer** | `app/db.py` |
| **Tests** | `test/test_advanced_chunker.py`, `test/test_geration_retrieval.py` |
| **Project context** | `SYSTEM_PROMPT_FOR_AI.md` (read for domain understanding) |
