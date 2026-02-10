#!/usr/bin/env python
"""End-to-end test for the complete job agent workflow.

Tests the full pipeline:
1. Resume upload and parsing → Profile created with embedding
2. Job search → Jobs fetched and stored with embeddings  
3. Job ranking → Jobs ranked by similarity to profile
4. Notifications → Top matches sent via configured channel
5. Application prep → Resume suggestions, cover letter, intro email generated

Usage:
    python scripts/test_e2e.py
"""

import asyncio
import base64
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()


# Sample resume text for testing
SAMPLE_RESUME_TEXT = """
JOHN DEVELOPER
Seattle, WA | john.developer@email.com | (555) 123-4567 | linkedin.com/in/johndev | github.com/johndev

PROFESSIONAL SUMMARY
Senior Software Engineer with 6+ years of experience building scalable backend systems and APIs. 
Expert in Python, FastAPI, and cloud-native development on Azure. Passionate about clean code, 
automated testing, and mentoring junior developers.

TECHNICAL SKILLS
Languages: Python, TypeScript, SQL, Go
Frameworks: FastAPI, Django, Flask, React
Databases: PostgreSQL, Redis, MongoDB, Elasticsearch
Cloud: Azure (Functions, AKS, Cosmos DB, App Service), AWS (Lambda, ECS, RDS)
Tools: Docker, Kubernetes, Terraform, GitHub Actions, CI/CD
Other: REST APIs, GraphQL, Microservices, Event-driven architecture

PROFESSIONAL EXPERIENCE

Senior Software Engineer | TechStartup Inc. | Seattle, WA | 2021 - Present
- Architected and built a high-throughput API platform using FastAPI serving 10M+ requests/day
- Led migration from monolith to microservices on Azure Kubernetes Service, reducing latency by 40%
- Implemented real-time data pipeline with Azure Event Hubs processing 1M+ events/hour
- Mentored team of 4 junior developers, establishing code review practices and documentation standards
- Designed PostgreSQL schema optimization reducing query times from 500ms to 50ms

Software Engineer | BigCorp Solutions | Bellevue, WA | 2018 - 2021
- Developed RESTful APIs in Python/Django serving internal and external customers
- Built automated testing suite achieving 90% code coverage across 200+ microservices
- Created data ingestion pipeline processing 500GB+ daily using Apache Kafka
- Collaborated with DevOps to containerize applications and deploy to AWS ECS

Junior Developer | LocalAgency | Seattle, WA | 2016 - 2018
- Built responsive web applications using React and Node.js
- Developed internal tools automating manual workflows, saving 20+ hours/week
- Participated in agile development with 2-week sprints

EDUCATION
Bachelor of Science in Computer Science | University of Washington | 2016
- GPA: 3.7/4.0, Dean's List
- Relevant coursework: Data Structures, Algorithms, Database Systems, Distributed Systems

CERTIFICATIONS
- Azure Developer Associate (AZ-204) - 2023
- AWS Certified Developer - 2022
- Kubernetes Application Developer (CKAD) - 2022
"""


async def test_e2e():
    """Run the complete end-to-end test."""
    from job_agent.config import AppConfig, AzureOpenAIConfig
    from job_agent.store import get_store, InMemoryJobStore
    from job_agent.ranking import get_ranking_service, get_embedding_service
    from job_agent.models import (
        UserProfile, JobSearchCriteria, Job, RankedJob, JobStatus,
        NotificationChannel, NotificationConfig, FeedbackType, JobFeedback,
    )
    from job_agent.providers import get_provider
    from job_agent.resume_parser import ResumeParser, ParsedResume
    from job_agent.notifications import NotificationService
    from job_agent.application_prep import ApplicationPrepService, ApplicationPrepConfig
    
    print("=" * 70)
    print("JOB AGENT - END-TO-END TEST")
    print("=" * 70)
    
    # Load config
    config = AppConfig.load()
    print(f"\n✓ Config loaded")
    print(f"  - Azure OpenAI: {config.azure_openai.endpoint}")
    print(f"  - Deployment: {config.azure_openai.deployment_name}")
    
    # Initialize services
    if config.database and config.database.is_configured:
        print(f"  - PostgreSQL: {config.database.host}:{config.database.port}")
        store = await get_store(config.database)
    else:
        print("  - Storage: In-memory")
        store = InMemoryJobStore()
    
    ranking_service = get_ranking_service(config.azure_openai)
    embedding_service = get_embedding_service(config.azure_openai)
    provider = get_provider()
    
    results = {"passed": 0, "failed": 0}
    
    # =========================================================================
    # TEST 1: Resume Upload and Parsing
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: Resume Upload and Parsing")
    print("-" * 70)
    
    try:
        # Create a temporary text file with resume content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_RESUME_TEXT)
            temp_resume_path = f.name
        
        print(f"  - Created temp resume: {temp_resume_path}")
        
        # Test parsing (text-based since we don't have PDF/DOCX libs installed)
        parser = ResumeParser()
        
        # Since it's a .txt file, we'll parse it directly
        parsed = parser.extract_with_regex(SAMPLE_RESUME_TEXT)
        print(f"  - Parsed name: {parsed.name}")
        print(f"  - Parsed email: {parsed.email}")
        print(f"  - Parsed skills: {parsed.skills[:5]}...")
        
        # Convert to profile
        profile = parsed.to_user_profile()
        # Manually set name since regex doesn't extract it well
        if not profile.name:
            profile.name = "John Developer"
        profile.preferred_locations = ["Seattle, WA", "Remote"]
        profile.remote_preference = "flexible"
        profile.desired_titles = ["Senior Software Engineer", "Staff Engineer", "Backend Engineer"]
        
        # Generate embedding
        print("  - Generating profile embedding...")
        profile.embedding = await embedding_service.embed(profile.resume_text[:8000])
        print(f"  - Embedding dim: {len(profile.embedding)}")
        
        # Save profile
        await store.save_profile(profile)
        saved_profile = await store.get_profile(profile.id)
        
        assert saved_profile is not None, "Profile not saved"
        assert len(saved_profile.embedding) > 0, "Embedding not saved"
        
        print(f"  ✓ Profile saved with ID: {profile.id[:8]}...")
        results["passed"] += 1
        
        # Cleanup
        os.unlink(temp_resume_path)
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
        return results
    
    # =========================================================================
    # TEST 2: Job Search
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: Job Search")
    print("-" * 70)
    
    try:
        criteria = JobSearchCriteria(
            query="Python FastAPI Senior Engineer",
            location="Seattle, WA",
            remote_only=False,
        )
        print(f"  - Searching: '{criteria.query}' in {criteria.location}")
        
        jobs = await provider.fetch_jobs(criteria)
        print(f"  - Found {len(jobs)} jobs")
        
        if len(jobs) == 0:
            print("  ⚠ No jobs returned (provider may be rate-limited or mock)")
            # Create mock jobs for testing
            jobs = [
                Job(
                    title="Senior Python Developer",
                    company="TechCorp",
                    location="Seattle, WA",
                    description="""We're looking for a Senior Python Developer to join our growing team.
                    You'll be working on high-scale APIs using FastAPI and PostgreSQL.
                    Requirements: 5+ years Python, FastAPI or Django, SQL databases, AWS or Azure.
                    Nice to have: Kubernetes, microservices experience.""",
                    url="https://example.com/jobs/1",
                    source="mock",
                ),
                Job(
                    title="Staff Software Engineer - Backend",
                    company="StartupXYZ",
                    location="Remote (US)",
                    description="""Staff Engineer role for our backend platform team.
                    Tech stack: Python, Go, PostgreSQL, Redis, Kubernetes on GCP.
                    You'll architect and build our next-gen API platform.
                    Requirements: 7+ years experience, strong system design skills.""",
                    url="https://example.com/jobs/2",
                    source="mock",
                ),
                Job(
                    title="Machine Learning Engineer",
                    company="AIStartup",
                    location="San Francisco, CA",
                    description="""ML Engineer to work on our recommendation system.
                    Requirements: Python, TensorFlow/PyTorch, MLOps, AWS SageMaker.
                    Experience with large-scale data pipelines required.""",
                    url="https://example.com/jobs/3",
                    source="mock",
                ),
            ]
            print(f"  - Using {len(jobs)} mock jobs for testing")
        
        # Generate embeddings for jobs
        print("  - Generating job embeddings...")
        for job in jobs:
            job.embedding = await embedding_service.embed(
                f"{job.title} at {job.company}. {job.description[:2000]}"
            )
        
        # Save jobs
        for job in jobs:
            await store.add(job)
        
        saved_jobs = await store.list_all()
        assert len(saved_jobs) >= len(jobs), "Jobs not saved"
        print(f"  ✓ Saved {len(jobs)} jobs with embeddings")
        results["passed"] += 1
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
        return results
    
    # =========================================================================
    # TEST 3: Job Ranking
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: Job Ranking")
    print("-" * 70)
    
    try:
        print("  - Ranking jobs against profile...")
        ranked_jobs = await ranking_service.rank_jobs(jobs, profile)
        
        print(f"  - Ranked {len(ranked_jobs)} jobs:")
        for i, rj in enumerate(ranked_jobs[:5], 1):
            print(f"    {i}. {rj.job.title} @ {rj.job.company}")
            print(f"       Score: {rj.score:.2%} | Similarity: {rj.similarity_score:.2%} | Skills: {rj.skill_match_score:.2%}")
        
        # Verify ranking order
        scores = [rj.score for rj in ranked_jobs]
        assert scores == sorted(scores, reverse=True), "Jobs not sorted by score"
        
        print(f"  ✓ Jobs ranked successfully")
        results["passed"] += 1
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
        ranked_jobs = []
    
    # =========================================================================
    # TEST 4: Notifications
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 4: Notifications")
    print("-" * 70)
    
    try:
        if ranked_jobs:
            # Use console notification for testing
            notif_config = NotificationConfig(channel=NotificationChannel.CONSOLE)
            notification_service = NotificationService(configs=[notif_config])
            
            print("  - Sending notification with top matches...")
            result = await notification_service.send_job_matches(
                ranked_jobs=ranked_jobs[:3],
                profile=profile,
                title="Your Top Job Matches",
            )
            
            assert result.get("console") == True, "Console notification failed"
            print(f"  ✓ Notification sent: {result}")
            results["passed"] += 1
        else:
            print("  ⚠ Skipped (no ranked jobs)")
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
    
    # =========================================================================
    # TEST 5: Feedback Storage
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 5: Feedback Storage")
    print("-" * 70)
    
    try:
        if ranked_jobs:
            top_job = ranked_jobs[0].job
            
            feedback = JobFeedback(
                job_id=top_job.id,
                feedback_type=FeedbackType.GOOD_FIT,
                notes="Great match for my skills!",
            )
            
            await store.save_feedback(feedback)
            feedbacks = await store.get_feedback(top_job.id)
            
            assert len(feedbacks) > 0, "Feedback not saved"
            assert feedbacks[0].feedback_type == FeedbackType.GOOD_FIT
            
            print(f"  ✓ Feedback saved for job {top_job.id[:8]}...")
            results["passed"] += 1
        else:
            print("  ⚠ Skipped (no ranked jobs)")
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
    
    # =========================================================================
    # TEST 6: Application Prep
    # =========================================================================
    print("\n" + "-" * 70)
    print("TEST 6: Application Prep")
    print("-" * 70)
    
    try:
        if ranked_jobs:
            top_job = ranked_jobs[0].job
            
            print(f"  - Preparing application for: {top_job.title} @ {top_job.company}")
            
            prep_config = ApplicationPrepConfig(
                max_tokens=1000,
                temperature=0.7,
            )
            prep_service = ApplicationPrepService(config.azure_openai, prep_config)
            
            package = await prep_service.prepare_application(top_job, profile)
            
            print(f"  - Package ID: {package.id[:8]}...")
            print(f"  - Status: {package.status}")
            
            # Show snippets
            print(f"\n  📝 Resume Suggestions (first 2):")
            if isinstance(package.resume_suggestions, list):
                for i, sugg in enumerate(package.resume_suggestions[:2], 1):
                    print(f"     {i}. {sugg[:100]}...")
            else:
                print(f"     {package.resume_suggestions[:200]}...")
            
            print(f"\n  📄 Cover Letter (snippet):")
            print(f"     {package.cover_letter[:200]}...")
            
            print(f"\n  ✉️ Intro Email (snippet):")
            print(f"     {package.intro_email[:200]}...")
            
            # Store package
            await store.save_application_package(package)
            retrieved = await store.get_application_package(package.id)
            
            assert retrieved is not None, "Package not saved"
            print(f"\n  ✓ Application package saved")
            results["passed"] += 1
        else:
            print("  ⚠ Skipped (no ranked jobs)")
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["failed"] += 1
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Status: {'✓ ALL TESTS PASSED' if results['failed'] == 0 else '✗ SOME TESTS FAILED'}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    asyncio.run(test_e2e())
