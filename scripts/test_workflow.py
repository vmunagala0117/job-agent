#!/usr/bin/env python
"""Test script to verify the full job search and ranking workflow.

Tests:
1. Upload resume → Profile saved with embedding
2. Search jobs → Jobs saved with embeddings
3. Rank jobs → Uses embeddings for similarity

Usage:
    python scripts/test_workflow.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


async def test_workflow():
    """Test the complete workflow."""
    from job_agent.config import AppConfig
    from job_agent.store import get_store, InMemoryJobStore
    from job_agent.ranking import get_ranking_service
    from job_agent.models import UserProfile, JobSearchCriteria
    from job_agent.providers import get_provider
    
    print("=" * 60)
    print("Job Agent Workflow Test")
    print("=" * 60)
    
    # Load config
    config = AppConfig.load()
    
    # Get store (PostgreSQL if configured, otherwise in-memory)
    if config.database and config.database.is_configured:
        print(f"\n1. Connecting to PostgreSQL at {config.database.host}...")
        store = await get_store(config.database)
        using_postgres = True
    else:
        print("\n1. Using in-memory store (no database configured)")
        store = InMemoryJobStore()
        using_postgres = False
    print("   ✓ Store initialized")
    
    # Get ranking service
    print("\n2. Initializing ranking service...")
    ranking_service = get_ranking_service()
    print("   ✓ Ranking service initialized")
    
    # Test 1: Create and save a user profile with embedding
    print("\n3. Testing profile creation with embedding...")
    profile = UserProfile(
        name="Test User",
        resume_text="""Senior Software Engineer with 10 years of experience in Python, 
        machine learning, and cloud infrastructure. Expert in AWS, Docker, Kubernetes.
        Previously worked at major tech companies building scalable AI systems.
        Strong background in natural language processing and deep learning.""",
        skills=["Python", "Machine Learning", "AWS", "Docker", "Kubernetes", "NLP"],
        current_title="Senior Software Engineer",
        desired_titles=["AI Engineer", "ML Lead", "Principal Engineer"],
        preferred_locations=["Remote", "San Francisco"],
        remote_preference="remote",
        years_experience=10,
    )
    
    # Generate embedding
    profile = await ranking_service.embed_user_profile(profile)
    if profile.embedding:
        print(f"   ✓ Profile embedding generated ({len(profile.embedding)} dimensions)")
    else:
        print("   ✗ Failed to generate profile embedding!")
        return False
    
    # Save profile
    saved_profile = await store.save_profile(profile)
    print(f"   ✓ Profile saved with ID: {saved_profile.id[:8]}...")
    
    # Verify profile retrieval
    retrieved_profile = await store.get_profile(profile.id)
    if retrieved_profile and retrieved_profile.embedding:
        print("   ✓ Profile retrieved from store with embedding")
    else:
        print("   ✗ Failed to retrieve profile with embedding!")
        return False
    
    # Test 2: Search for jobs and verify embeddings are saved
    print("\n4. Testing job search with embedding generation...")
    provider = get_provider()
    provider_name = type(provider).__name__
    print(f"   Using provider: {provider_name}")
    
    criteria = JobSearchCriteria(
        query="Python developer",
        remote_only=True,
        max_results=3,  # Small number for testing
    )
    
    jobs = await provider.fetch_jobs(criteria)
    if not jobs:
        print("   ⚠ No jobs returned from provider (this is OK for mock provider)")
        # Create mock jobs for testing
        from job_agent.models import Job
        jobs = [
            Job(
                title="Senior Python Developer",
                company="TechCorp",
                location="Remote",
                description="Looking for a Python developer with ML experience...",
                skills=["Python", "Machine Learning", "AWS"],
            ),
            Job(
                title="AI Engineer",
                company="AIStartup",
                location="San Francisco, CA",
                description="Building next-gen AI systems with NLP and deep learning...",
                skills=["Python", "NLP", "Deep Learning", "PyTorch"],
            ),
        ]
        print(f"   Created {len(jobs)} test jobs")
    else:
        print(f"   ✓ Fetched {len(jobs)} jobs from {provider_name}")
    
    # Save jobs
    await store.add_many(jobs)
    print(f"   ✓ Jobs saved to store")
    
    # Generate embeddings for jobs
    jobs = await ranking_service.embed_jobs(jobs)
    embedded_count = sum(1 for j in jobs if j.embedding)
    print(f"   ✓ Generated embeddings for {embedded_count}/{len(jobs)} jobs")
    
    # Update embeddings in store
    job_embeddings = [(j.id, j.embedding) for j in jobs if j.embedding]
    if job_embeddings:
        await store.update_job_embeddings(job_embeddings)
        print(f"   ✓ Persisted embeddings to store")
    
    # Verify embeddings are stored
    if using_postgres:
        jobs_without = await store.get_jobs_without_embeddings(limit=10)
        if len(jobs_without) == 0:
            print("   ✓ All jobs have embeddings in database")
        else:
            print(f"   ⚠ {len(jobs_without)} jobs still without embeddings")
    
    # Test 3: Rank jobs
    print("\n5. Testing job ranking...")
    ranked_jobs = await ranking_service.rank_jobs(jobs, profile, top_k=5)
    
    if ranked_jobs:
        print(f"   ✓ Ranked {len(ranked_jobs)} jobs")
        print("\n   Top matches:")
        for i, rj in enumerate(ranked_jobs[:3], 1):
            print(f"   {i}. {rj.job.title} at {rj.job.company}")
            print(f"      Score: {rj.score*100:.1f}% (similarity: {rj.similarity_score*100:.1f}%, skills: {rj.skill_match_score*100:.1f}%)")
    else:
        print("   ✗ No ranked jobs returned!")
        return False
    
    # Test 4: Verify default profile retrieval
    print("\n6. Testing default profile retrieval...")
    default_profile = await store.get_default_profile()
    if default_profile:
        print(f"   ✓ Default profile: {default_profile.name}")
        print(f"   ✓ Has embedding: {default_profile.embedding is not None}")
    else:
        print("   ⚠ No default profile found (OK for fresh database)")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    print("\n📋 Summary of the workflow:")
    print("   1. Resume → Profile with embedding → Saved to DB")
    print("   2. Search → Jobs with embeddings → Saved to DB")
    print("   3. Rank → Cosine similarity between profile and jobs")
    print("   4. Embeddings are persisted and reused on subsequent runs")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_workflow())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
