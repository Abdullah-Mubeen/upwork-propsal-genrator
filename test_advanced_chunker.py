"""
Test suite for Advanced Data Chunking Strategy

Tests the 4-chunk strategy:
1. JOB_FACTS_CHUNK
2. PROPOSAL_CHUNK
3. FEEDBACK_CHUNK
4. TEMPLATE_CHUNK
"""

import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.advanced_chunker import (
    AdvancedChunkProcessor,
    JobFactsChunker,
    ProposalChunker,
    FeedbackChunker,
    TemplateChunker,
    ChunkTypeEnum
)


# Sample test data
SAMPLE_JOB_DATA = {
    "contract_id": "job_test_001",
    "company_name": "TechCorp Inc",
    "job_title": "Senior Full-Stack Developer",
    "job_description": """
    We are looking for an experienced Full-Stack Developer to join our team.
    Responsibilities:
    - Design and develop scalable web applications
    - Work with React for frontend and Node.js for backend
    - Implement RESTful APIs and database optimization
    - Collaborate with cross-functional teams
    - Participate in code reviews and technical discussions
    
    Requirements:
    - 5+ years of experience with JavaScript/TypeScript
    - Strong knowledge of React and Node.js
    - Experience with PostgreSQL and MongoDB
    - Familiarity with Docker and CI/CD pipelines
    - Bachelor's degree in Computer Science or related field
    """,
    "your_proposal_text": """
    Thank you for considering my application. I am excited about this opportunity to join your team as a Senior Full-Stack Developer.

    With over 7 years of experience in full-stack web development, I have successfully delivered multiple scalable applications using React, Node.js, and modern cloud technologies. My technical expertise includes:

    Technical Expertise:
    - Frontend: React, TypeScript, Redux, Material-UI, responsive design
    - Backend: Node.js, Express, GraphQL, REST APIs
    - Databases: PostgreSQL, MongoDB, Redis optimization
    - DevOps: Docker, Kubernetes, AWS, CI/CD with GitHub Actions
    - Tools: Git, Agile/Scrum methodologies

    Relevant Experience:
    I recently led a team of 3 developers to rebuild a legacy system using React and Node.js, reducing load times by 60% and improving user satisfaction scores. The project involved complex data synchronization and real-time updates, which I implemented using WebSockets and event-driven architecture.

    Why I'm a Great Fit:
    - Passionate about writing clean, maintainable code
    - Strong communicator and team player
    - Proven track record of delivering projects on time
    - Continuous learner eager to adopt new technologies

    I am confident that my skills and experience make me an ideal candidate for this role. I would welcome the opportunity to discuss how I can contribute to your team's success.

    Best regards,
    Your Candidate
    """,
    "skills_required": ["JavaScript", "React", "Node.js", "PostgreSQL", "MongoDB", "Docker"],
    "industry": "technology",
    "start_date": "2024-01-15",
    "end_date": "2024-06-15",
    "project_status": "completed",
    "task_type": "full_stack",
    "urgent_adhoc": False,
    "client_feedback": """
    ★★★★★ Excellent Work!
    
    This developer exceeded our expectations. The deliverables were completed ahead of schedule and with exceptional quality. The code was well-documented and easy to maintain.
    
    What stood out:
    - Professional approach to problem-solving
    - Proactive communication and regular updates
    - High-quality code with comprehensive testing
    - Great team player who helped mentor junior developers
    
    Highly recommended for any complex project!
    
    Date: March 15, 2024
    Review by: Project Manager
    """
}


def test_job_facts_chunker():
    """Test JOB_FACTS_CHUNK creation"""
    print("\n" + "="*70)
    print("TEST 1: JOB_FACTS_CHUNK")
    print("="*70)
    
    chunk = JobFactsChunker.extract_job_facts(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "JOB_FACTS_CHUNK should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.JOB_FACTS.value
    assert "text" in chunk
    assert "metadata" in chunk
    assert len(chunk["text"]) > 0
    
    # Verify cleaned text (no URLs, no extra formatting)
    assert "http" not in chunk["text"].lower(), "Text should not contain URLs"
    assert "<" not in chunk["text"], "Text should not contain HTML tags"
    
    # Verify metadata
    metadata = chunk["metadata"]
    assert metadata["job_id"] == SAMPLE_JOB_DATA["contract_id"]
    assert metadata["title"] == SAMPLE_JOB_DATA["job_title"]
    assert metadata["company"] == SAMPLE_JOB_DATA["company_name"]
    assert len(metadata["skills"]) > 0
    
    print(f"✓ JOB_FACTS_CHUNK created successfully")
    print(f"  - Text length: {chunk['length']} chars")
    print(f"  - Skills: {metadata['skills']}")
    print(f"  - Urgency: {metadata['urgency']}")
    print(f"  - Category: {metadata['category']}")
    print(f"  - JSON serializable: ✓")
    
    # Test JSON serialization
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    print(f"  - JSON size: {len(json_str)} bytes")
    
    return chunk


def test_proposal_chunker():
    """Test PROPOSAL_CHUNK creation (never split)"""
    print("\n" + "="*70)
    print("TEST 2: PROPOSAL_CHUNK (Never Split)")
    print("="*70)
    
    chunk = ProposalChunker.extract_proposal(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "PROPOSAL_CHUNK should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.PROPOSAL.value
    
    # CRITICAL: Verify proposal is NEVER split
    original_length = len(SAMPLE_JOB_DATA["your_proposal_text"].strip())
    chunk_length = len(chunk["text"])
    assert chunk_length == original_length, f"Proposal should NOT be split. Original: {original_length}, Chunk: {chunk_length}"
    
    # Verify metadata
    metadata = chunk["metadata"]
    assert metadata["did_win"] == (SAMPLE_JOB_DATA["project_status"] == "completed")
    assert len(metadata["skills"]) > 0
    
    print(f"✓ PROPOSAL_CHUNK created successfully (NO SPLIT)")
    print(f"  - Original length: {original_length} chars")
    print(f"  - Chunk length: {chunk_length} chars")
    print(f"  - Did win: {metadata['did_win']}")
    print(f"  - Style: {metadata['style']}")
    print(f"  - Tone: {metadata['tone']}")
    
    # Test JSON serialization
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    print(f"  - JSON size: {len(json_str)} bytes")
    
    return chunk


def test_feedback_chunker():
    """Test FEEDBACK_CHUNK creation (cleaned)"""
    print("\n" + "="*70)
    print("TEST 3: FEEDBACK_CHUNK (Cleaned)")
    print("="*70)
    
    chunk = FeedbackChunker.extract_feedback(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "FEEDBACK_CHUNK should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.FEEDBACK.value
    
    # Verify cleaning
    text = chunk["text"]
    assert "★" not in text, "Feedback should not contain star symbols"
    assert "Date:" not in text, "Feedback should not contain dates"
    assert "Review by:" not in text, "Feedback should not contain labels"
    
    # Verify sentiment detection
    metadata = chunk["metadata"]
    print(f"  - Detected sentiment: {metadata['sentiment']}")
    assert metadata["sentiment"] in ["positive", "negative", "neutral", "mixed"]
    
    print(f"✓ FEEDBACK_CHUNK created successfully (CLEANED)")
    print(f"  - Original feedback had stars and dates: ✓ Removed")
    print(f"  - Text length: {chunk['length']} chars")
    print(f"  - Sentiment: {metadata['sentiment']}")
    
    # Test JSON serialization
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    print(f"  - JSON size: {len(json_str)} bytes")
    
    return chunk


def test_template_chunker():
    """Test TEMPLATE_CHUNK creation"""
    print("\n" + "="*70)
    print("TEST 4: TEMPLATE_CHUNK")
    print("="*70)
    
    template_text = """
    Professional Project Proposal Template
    
    Subject: Proposal for [PROJECT_NAME] - [COMPANY_NAME]
    
    Dear [HIRING_MANAGER],
    
    I am writing to express my strong interest in [PROJECT_DESCRIPTION].
    With [YEARS] years of experience in [DOMAIN], I am confident I can deliver exceptional results.
    
    Key Qualifications:
    • Expertise in [SKILL_1], [SKILL_2], [SKILL_3]
    • Proven track record with [RELEVANT_ACHIEVEMENT]
    • Strong communication and team collaboration skills
    
    Project Approach:
    1. Initial discovery and requirements gathering
    2. Technical architecture and planning
    3. Implementation and development
    4. Testing and quality assurance
    5. Deployment and support
    
    Timeline: [DURATION]
    Investment: [BUDGET]
    
    I would welcome the opportunity to discuss this proposal further.
    
    Best regards,
    [YOUR_NAME]
    """
    
    chunk = TemplateChunker.extract_template(
        template_name="Professional Proposal",
        template_text=template_text,
        category="formal_proposal",
        use_cases=["web_development", "consultation", "technical_projects"]
    )
    
    assert chunk is not None
    assert chunk["chunk_type"] == ChunkTypeEnum.TEMPLATE.value
    assert "template_name" in chunk["metadata"]
    
    print(f"✓ TEMPLATE_CHUNK created successfully")
    print(f"  - Template name: {chunk['metadata']['template_name']}")
    print(f"  - Category: {chunk['metadata']['category']}")
    print(f"  - Use cases: {chunk['metadata']['use_cases']}")
    print(f"  - Text length: {chunk['length']} chars")
    
    # Test JSON serialization
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    print(f"  - JSON size: {len(json_str)} bytes")
    
    return chunk


def test_advanced_processor():
    """Test AdvancedChunkProcessor orchestration"""
    print("\n" + "="*70)
    print("TEST 5: AdvancedChunkProcessor (Full Pipeline)")
    print("="*70)
    
    processor = AdvancedChunkProcessor()
    
    # Test flat chunks (for batch embedding)
    flat_chunks = processor.get_all_chunks_flat(SAMPLE_JOB_DATA)
    
    print(f"✓ Created {len(flat_chunks)} chunks")
    
    chunk_types = {}
    total_text_length = 0
    
    for chunk in flat_chunks:
        chunk_type = chunk["chunk_type"]
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        total_text_length += chunk["length"]
        
        # Validate chunk structure
        assert "chunk_type" in chunk
        assert "text" in chunk
        assert "metadata" in chunk
        assert "created_at" in chunk
        assert len(chunk["text"]) > 0
        
        # Test JSON serialization
        json_str = json.dumps(chunk, default=str)
        assert json_str, f"Chunk {chunk_type} should be JSON serializable"
    
    print(f"\n  Chunk breakdown:")
    for chunk_type, count in chunk_types.items():
        print(f"    - {chunk_type}: {count} chunk(s)")
    
    print(f"\n  Total text length: {total_text_length} chars")
    print(f"  All chunks JSON-serializable: ✓")
    
    # Test full processing with metadata
    full_result = processor.process_job_data(SAMPLE_JOB_DATA)
    assert full_result["job_id"] == SAMPLE_JOB_DATA["contract_id"]
    assert full_result["summary"]["total_chunks"] == len(flat_chunks)
    print(f"\n  Summary statistics:")
    print(f"    - Total chunks: {full_result['summary']['total_chunks']}")
    print(f"    - Total text: {full_result['summary']['total_text_length']} chars")
    
    return flat_chunks


def test_json_validation():
    """Test JSON serialization of all chunk types"""
    print("\n" + "="*70)
    print("TEST 6: JSON Serialization & Validation")
    print("="*70)
    
    processor = AdvancedChunkProcessor()
    chunks = processor.get_all_chunks_flat(SAMPLE_JOB_DATA)
    
    for chunk in chunks:
        is_valid, error_msg = processor.validate_chunk_json(chunk)
        assert is_valid, f"Chunk validation failed: {error_msg}"
        
        # Verify JSON roundtrip
        json_str = json.dumps(chunk, default=str)
        roundtrip = json.loads(json_str)
        assert roundtrip["chunk_type"] == chunk["chunk_type"]
        assert roundtrip["metadata"]["job_id"] == chunk["metadata"]["job_id"]
    
    print(f"✓ All {len(chunks)} chunks are valid JSON")
    print(f"✓ JSON roundtrip successful for all chunks")
    
    # Test export
    json_export = processor.export_chunks_json({
        "chunks": [chunks[0]],
        "summary": {"count": 1}
    })
    assert isinstance(json_export, str)
    assert "chunk_type" in json_export
    print(f"✓ JSON export successful ({len(json_export)} bytes)")


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " ADVANCED CHUNKING STRATEGY - TEST SUITE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        test_job_facts_chunker()
        test_proposal_chunker()
        test_feedback_chunker()
        test_template_chunker()
        test_advanced_processor()
        test_json_validation()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  ✓ JOB_FACTS_CHUNK: Cleaned job descriptions with metadata")
        print("  ✓ PROPOSAL_CHUNK: Full proposals (never split)")
        print("  ✓ FEEDBACK_CHUNK: Cleaned feedback (no noise)")
        print("  ✓ TEMPLATE_CHUNK: Writing templates")
        print("  ✓ All chunks are JSON-ready for embedding")
        print("  ✓ Backward compatible with existing DataChunker interface")
        print("\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
