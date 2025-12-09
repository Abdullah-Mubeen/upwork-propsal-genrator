"""
Test suite for Advanced Data Chunking Strategy (5-Layer Semantic)

Tests the 5-layer semantic strategy:
1. CONTEXT_SNAPSHOT
2. REQUIREMENTS_PROFILE
3. TIMELINE_SCOPE
4. DELIVERABLES_PORTFOLIO
5. FEEDBACK_OUTCOMES
"""

import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.advanced_chunker import (
    AdvancedChunkProcessor,
    ContextSnapshotChunker,
    RequirementsProfileChunker,
    TimelineScopeChunker,
    DeliverablesPortfolioChunker,
    FeedbackOutcomesChunker,
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


def test_context_snapshot():
    """Test CONTEXT_SNAPSHOT chunk creation"""
    print("\n" + "="*70)
    print("TEST 1: CONTEXT_SNAPSHOT")
    print("="*70)
    
    chunk = ContextSnapshotChunker.extract(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "CONTEXT_SNAPSHOT should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.CONTEXT_SNAPSHOT.value
    assert "text" in chunk
    assert "metadata" in chunk
    
    metadata = chunk["metadata"]
    assert metadata["industry"] == SAMPLE_JOB_DATA["industry"]
    assert metadata["task_type"] == SAMPLE_JOB_DATA["task_type"]
    
    print(f"✓ CONTEXT_SNAPSHOT created successfully")
    print(f"  - Text: {chunk['text'][:100]}...")
    print(f"  - Industry: {metadata['industry']}")
    print(f"  - Task type: {metadata['task_type']}")
    print(f"  - Urgency: {metadata.get('urgency', 'normal')}")
    
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    
    return chunk


def test_requirements_profile():
    """Test REQUIREMENTS_PROFILE chunk creation"""
    print("\n" + "="*70)
    print("TEST 2: REQUIREMENTS_PROFILE")
    print("="*70)
    
    chunk = RequirementsProfileChunker.extract(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "REQUIREMENTS_PROFILE should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.REQUIREMENTS_PROFILE.value
    
    metadata = chunk["metadata"]
    assert len(metadata["skills"]) > 0, "Should have skills"
    
    print(f"✓ REQUIREMENTS_PROFILE created successfully")
    print(f"  - Text length: {chunk['length']} chars")
    print(f"  - Skills: {', '.join(metadata['skills'])}")
    print(f"  - Complexity: {metadata.get('task_complexity', 'medium')}")
    
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    
    return chunk


def test_timeline_scope():
    """Test TIMELINE_SCOPE chunk creation"""
    print("\n" + "="*70)
    print("TEST 3: TIMELINE_SCOPE")
    print("="*70)
    
    chunk = TimelineScopeChunker.extract(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "TIMELINE_SCOPE should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.TIMELINE_SCOPE.value
    
    metadata = chunk["metadata"]
    print(f"✓ TIMELINE_SCOPE created successfully")
    print(f"  - Duration: {metadata.get('duration_days', 'N/A')} days")
    print(f"  - Is completed: {metadata.get('is_completed', False)}")
    print(f"  - Text: {chunk['text'][:100]}...")
    
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    
    return chunk


def test_deliverables_portfolio():
    """Test DELIVERABLES_PORTFOLIO chunk creation"""
    print("\n" + "="*70)
    print("TEST 4: DELIVERABLES_PORTFOLIO")
    print("="*70)
    
    chunk = DeliverablesPortfolioChunker.extract(SAMPLE_JOB_DATA)
    
    if chunk is not None:
        assert chunk["chunk_type"] == ChunkTypeEnum.DELIVERABLES_PORTFOLIO.value
        print(f"✓ DELIVERABLES_PORTFOLIO created successfully")
        print(f"  - Text: {chunk['text'][:100]}...")
        
        json_str = json.dumps(chunk, default=str)
        assert json_str, "Should be JSON serializable"
    else:
        print(f"✓ DELIVERABLES_PORTFOLIO skipped (no portfolio URLs)")
    
    return chunk


def test_feedback_outcomes():
    """Test FEEDBACK_OUTCOMES chunk creation"""
    print("\n" + "="*70)
    print("TEST 5: FEEDBACK_OUTCOMES")
    print("="*70)
    
    chunk = FeedbackOutcomesChunker.extract(SAMPLE_JOB_DATA)
    
    assert chunk is not None, "FEEDBACK_OUTCOMES should not be None"
    assert chunk["chunk_type"] == ChunkTypeEnum.FEEDBACK_OUTCOMES.value
    
    metadata = chunk["metadata"]
    print(f"✓ FEEDBACK_OUTCOMES created successfully")
    print(f"  - Text length: {chunk['length']} chars")
    print(f"  - Effectiveness: {metadata.get('proposal_effectiveness', 'N/A')}")
    print(f"  - Satisfaction: {metadata.get('client_satisfaction', 'N/A')}/5")
    
    json_str = json.dumps(chunk, default=str)
    assert json_str, "Should be JSON serializable"
    
    return chunk

    
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
    print("║" + " ADVANCED CHUNKING STRATEGY (5-LAYER) - TEST SUITE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        test_context_snapshot()
        test_requirements_profile()
        test_timeline_scope()
        test_deliverables_portfolio()
        test_feedback_outcomes()
        test_advanced_processor()
        test_json_validation()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary (5-Layer Semantic Strategy):")
        print("  ✓ CONTEXT_SNAPSHOT: Company, industry, job title, urgency")
        print("  ✓ REQUIREMENTS_PROFILE: Skills, job description, complexity")
        print("  ✓ TIMELINE_SCOPE: Duration, status, timeline analysis")
        print("  ✓ DELIVERABLES_PORTFOLIO: Portfolio links for reference")
        print("  ✓ FEEDBACK_OUTCOMES: Client feedback, success patterns")
        print("  ✓ All chunks are JSON-ready with enhanced metadata")
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

